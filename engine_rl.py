import math
import inspect
from contextlib import nullcontext
from typing import Iterable, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn

import util.misc as misc


# ==========================================================
# Helpers
# ==========================================================

def update_ema(target_params, source_params, rate=0.99):
    """Exponential moving average update."""
    rate = float(rate)
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1.0 - rate)


def _safe_to_device(x, device, non_blocking: bool = True):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)
    return x


def _filter_kwargs_by_signature(fn, kwargs: dict) -> dict:
    """Drop unexpected kwargs to avoid TypeError across different implementations."""
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _autocast_ctx(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()


def _unpack_batch(batch):
    """
    Returns: (samples, captions, layout, layout_mask, meta)
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    if len(batch) < 2:
        raise ValueError(f"Unexpected batch len={len(batch)}; need at least (samples, captions, ...)")

    samples = batch[0]
    captions = batch[1]
    layout = batch[2] if len(batch) >= 3 else None
    layout_mask = batch[3] if len(batch) >= 4 else None
    meta = batch[4] if len(batch) >= 5 else None

    if isinstance(captions, (tuple, list)):
        captions = list(captions)
    else:
        captions = [captions]

    return samples, captions, layout, layout_mask, meta


def _parse_text_encoder_output(enc_out) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Make engine compatible with multiple text_encoder.encode() variants
    """
    if torch.is_tensor(enc_out):
        return enc_out, None

    if isinstance(enc_out, (tuple, list)):
        if len(enc_out) == 0:
            raise ValueError("text_encoder.encode() returned empty tuple/list")
        emb = enc_out[0]
        mask = enc_out[1] if len(enc_out) > 1 else None
        return emb, mask

    if isinstance(enc_out, dict):
        emb = enc_out.get("emb", None)
        mask = enc_out.get("mask", None)
        return emb, mask

    raise TypeError(f"Unsupported text_encoder.encode() output type: {type(enc_out)}")


def _reduce_model_loss_to_per_sample(
    loss_vec: torch.Tensor,
    batch_size: int,
    model: nn.Module,
) -> torch.Tensor:
    """
    Convert model(reduction="none") outputs into per-sample loss [B].
    """
    loss_vec = loss_vec.view(-1)

    if loss_vec.numel() == batch_size:
        return loss_vec

    if loss_vec.numel() % batch_size != 0:
        raise RuntimeError(
            f"Cannot reduce loss: got numel={loss_vec.numel()}, expected multiple of B({batch_size})."
        )

    mul = loss_vec.numel() // batch_size
    return loss_vec.view(mul, batch_size).mean(dim=0)


@torch.no_grad()
def _moments_to_latents_if_needed(
    samples: torch.Tensor,
    vae_embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = samples.to(device=device, dtype=dtype)
    if x.ndim != 4:
        return x
    C = int(x.shape[1])
    if C == int(vae_embed_dim):
        return x
    if C == int(2 * vae_embed_dim):
        mean = x[:, :vae_embed_dim]
        logvar = x[:, vae_embed_dim:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return z
    return x


@torch.no_grad()
def _vae_encode_to_latents(
    vae: nn.Module,
    images: torch.Tensor,
    vae_scale: float,
) -> torch.Tensor:
    enc = vae.encode(images)
    z = None

    if hasattr(enc, "sample") and callable(getattr(enc, "sample")):
        z = enc.sample()
    elif hasattr(enc, "mode") and callable(getattr(enc, "mode")):
        z = enc.mode()
    elif torch.is_tensor(enc):
        z = enc
    elif isinstance(enc, (tuple, list)) and len(enc) > 0 and torch.is_tensor(enc[0]):
        z = enc[0]
    elif isinstance(enc, dict):
        for k in ["latent", "latents", "z", "moments"]:
            if k in enc and torch.is_tensor(enc[k]):
                z = enc[k]
                break

    if z is None:
        raise RuntimeError(f"Unsupported vae.encode() output type: {type(enc)}")

    return z * float(vae_scale)


def _normalize_advantages(
    rewards_grouped: torch.Tensor,  # [B, G]
    mode: str,
) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "none":
        return rewards_grouped
    mean = rewards_grouped.mean(dim=1, keepdim=True)
    if mode == "center":
        return rewards_grouped - mean
    if mode == "zscore":
        std = rewards_grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        return (rewards_grouped - mean) / std
    raise ValueError(f"Unknown adv_norm mode: {mode}")


def _compose_reward(
    reward_out: Union[torch.Tensor, Tuple, Dict[str, Any]],
    args,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward composition priority:
      1) Tensor -> reward directly
      2) dict -> 'reward' key OR weighted sum of components
    """
    comps: Dict[str, torch.Tensor] = {}

    if torch.is_tensor(reward_out):
        comps["reward"] = reward_out
        return reward_out, comps

    if not isinstance(reward_out, dict):
        raise TypeError(f"Unsupported reward_model output type: {type(reward_out)}")

    iou = reward_out.get("iou", torch.tensor(0.0, device=device))
    conf = reward_out.get("conf", torch.tensor(0.0, device=device))
    center = reward_out.get("center", None)
    if center is None:
        center = reward_out.get("centroid", torch.tensor(0.0, device=device))

    comps["iou"] = iou
    comps["conf"] = conf
    comps["center"] = center

    if "reward" in reward_out and torch.is_tensor(reward_out["reward"]):
        r = reward_out["reward"]
    else:
        lam_iou = float(getattr(args, "reward_lambda_iou", 1.0))
        lam_conf = float(getattr(args, "reward_lambda_conf", 0.0))
        lam_center = float(getattr(args, "reward_lambda_center",
                                   getattr(args, "reward_lambda_centroid", 0.0)))
        r = lam_iou * iou + lam_conf * conf + lam_center * center

    comps["reward"] = r
    return r, comps


def _compute_kl_penalty(
    ref_loss: torch.Tensor,            # [BG]
    cur_loss_detached: torch.Tensor,   # [BG]
    args,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate log_ratio using loss difference:
        log_ratio ~ (ref_loss - cur_loss)
    """
    log_ratio = (ref_loss - cur_loss_detached)
    kl_mode = str(getattr(args, "kl_mode", "mse")).lower()
    kl_scale = float(getattr(args, "kl_scale", 1.0))

    if kl_mode == "mse":
        kl_pen = (log_ratio ** 2)
    elif kl_mode == "abs":
        kl_pen = log_ratio.abs()
    elif kl_mode == "linear":
        kl_pen = log_ratio
    else:
        raise ValueError(f"Unknown kl_mode: {kl_mode}")

    if kl_scale != 1.0:
        kl_pen = kl_pen * kl_scale

    return kl_pen.detach(), log_ratio.detach()


# ==========================================================
# GRPO RL Training Loop
# ==========================================================

def train_one_epoch_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    vae: nn.Module,
    reward_model: nn.Module,
    model_params,
    ema_params,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    text_encoder,
    log_writer=None,
    args=None,
):
    if args is None:
        raise ValueError("args must not be None for RL training")

    use_amp = bool(getattr(args, "amp", False)) and (device.type == "cuda")

    vae.eval()
    reward_model.eval()
    if ref_model is not None:
        ref_model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("reward", misc.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("kl_pen", misc.SmoothedValue(window_size=20, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_rl", misc.SmoothedValue(window_size=20, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_sft", misc.SmoothedValue(window_size=20, fmt="{value:.6f}"))

    header = f"Epoch: [{epoch}] [RL-GRPO]"
    print_freq = 10

    group_size = int(getattr(args, "group_size", 4))
    accum_iter = max(1, int(getattr(args, "accum_iter", 1)))
    rl_step_ratio = float(getattr(args, "rl_step_ratio", 1.0))
    adv_norm = str(getattr(args, "adv_norm", "zscore"))
    reward_clip = float(getattr(args, "reward_clip", 10.0))
    kl_coef = float(getattr(args, "kl_coef", 0.0))
    use_kl = (ref_model is not None) and (kl_coef > 0.0)

    num_iter = int(getattr(args, "num_iter", 256))
    cfg = float(getattr(args, "cfg", 1.0))
    cfg_schedule = str(getattr(args, "cfg_schedule", "linear"))
    temperature = float(getattr(args, "temperature", 1.0))
    vae_scale = float(getattr(vae, "scaling_factor", 0.18215))
    vae_embed_dim = int(getattr(args, "vae_embed_dim", 16))

    LOSS_SCALE = float(getattr(args, "loss_scale", 10000.0))

    optimizer.zero_grad(set_to_none=True)

    if misc.is_main_process() and epoch == 0 and ref_model is not None:
        with torch.inference_mode():
            real_model = model.module if hasattr(model, "module") else model
            if hasattr(real_model, "encoder_blocks"):
                w1 = real_model.encoder_blocks[0].attn.qkv.weight.mean().item()
                w2 = ref_model.encoder_blocks[0].attn.qkv.weight.mean().item()
                if abs(w1 - w2) > 1e-5:
                    print("  [WARN] Reference Model weights DO NOT match! KL may be unstable.")

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, captions, layout, layout_mask, _ = _unpack_batch(batch)

        enc_out = text_encoder.encode(captions)
        text_emb, text_mask = _parse_text_encoder_output(enc_out)
        text_emb = _safe_to_device(text_emb, device)
        text_mask = _safe_to_device(text_mask, device) if text_mask is not None else None

        if layout is None:
            raise ValueError("RL training requires layout.")
        layout = _safe_to_device(layout, device).float()

        if layout_mask is None:
            layout_mask = torch.ones(layout.shape[:2], device=layout.device, dtype=torch.bool)
        else:
            layout_mask = _safe_to_device(layout_mask, device).bool()

        bsz = int(text_emb.shape[0])
        BG = int(bsz * group_size)

        # ======================================================
        # 1) Rollout (no grad)
        # ======================================================
        model.eval()
        with torch.inference_mode():
            text_emb_rep = text_emb.repeat_interleave(group_size, dim=0)
            text_mask_rep = text_mask.repeat_interleave(group_size, dim=0) if text_mask is not None else None
            layout_rep = layout.repeat_interleave(group_size, dim=0)
            layout_mask_rep = layout_mask.repeat_interleave(group_size, dim=0)

            with _autocast_ctx(device, use_amp):
                sample_kwargs = dict(
                    bsz=BG, num_iter=num_iter, cfg=cfg, cfg_schedule=cfg_schedule, temperature=temperature,
                    text_emb=text_emb_rep, text_mask=text_mask_rep,
                    layout=layout_rep, layout_mask=layout_mask_rep,
                )
                sample_kwargs = _filter_kwargs_by_signature(model.sample_tokens, sample_kwargs)
                gen_latents = model.sample_tokens(**sample_kwargs).float()

                gen_images = vae.decode(gen_latents / vae_scale).clamp(-1, 1)
                gen_images_norm = (gen_images + 1.0) * 0.5

            reward_out = reward_model(gen_images_norm, layout_rep, layout_mask_rep)
            rewards, reward_comps = _compose_reward(reward_out, args, device)
            rewards = rewards.view(-1).float()

            if reward_clip > 0:
                rewards = rewards.clamp(-reward_clip, reward_clip)

        # ======================================================
        # 2) Policy update (with grad)
        # ======================================================
        model.train()
        do_step = ((data_iter_step + 1) % accum_iter == 0)

        with torch.inference_mode():
            real_model = model.module if hasattr(model, "module") else model
            x_dummy = real_model.patchify(gen_latents)
            orders = real_model.sample_orders(bsz=BG, device=device)
            common_mask = real_model.random_masking(x_dummy, orders)

        with _autocast_ctx(device, use_amp):
            raw_loss_vec = model(
                latents=gen_latents,
                text_emb=text_emb_rep,
                text_mask=text_mask_rep,
                layout=layout_rep,
                layout_mask=layout_mask_rep,
                reduction="none",
                external_mask=common_mask
            )
            raw_loss_vec = raw_loss_vec.float().view(-1)
            per_sample_loss = _reduce_model_loss_to_per_sample(raw_loss_vec, BG, model)

            if use_kl:
                with torch.inference_mode():
                    with _autocast_ctx(device, use_amp):
                        ref_raw = ref_model(
                            latents=gen_latents,
                            text_emb=text_emb_rep,
                            text_mask=text_mask_rep,
                            layout=layout_rep,
                            layout_mask=layout_mask_rep,
                            reduction="none",
                            external_mask=common_mask
                        ).float().view(-1)

                    ref_loss = _reduce_model_loss_to_per_sample(ref_raw, BG, ref_model)
                kl_pen, log_ratio = _compute_kl_penalty(ref_loss, per_sample_loss.detach(), args)
            else:
                kl_pen = torch.zeros_like(per_sample_loss.detach())

            rewards_grouped = rewards.view(bsz, group_size)
            kl_grouped = kl_pen.view(bsz, group_size)

            shaped_rewards_grouped = rewards_grouped - (kl_coef * kl_grouped if use_kl else 0.0)
            adv_grouped = _normalize_advantages(shaped_rewards_grouped, mode=adv_norm)
            advantages = adv_grouped.reshape(-1).detach()

            rl_loss = (per_sample_loss * advantages).mean()

            if rl_step_ratio < 1.0:
                with torch.inference_mode():
                    if not torch.is_tensor(samples):
                        raise TypeError(f"samples must be Tensor, got {type(samples)}")
                    samples_t = _safe_to_device(samples, device).float()

                    if samples_t.ndim == 4 and int(samples_t.shape[1]) == 3:
                        gt_latents = _vae_encode_to_latents(vae, samples_t, vae_scale=vae_scale)
                    else:
                        gt_latents = _moments_to_latents_if_needed(
                            samples_t, vae_embed_dim=vae_embed_dim,
                            device=device, dtype=torch.float32
                        )

                sft_raw_loss = model(
                    imgs=None,
                    latents=gt_latents,
                    text_emb=text_emb, text_mask=text_mask,
                    layout=layout, layout_mask=layout_mask,
                    reduction="mean",
                ).float()

                sft_loss = sft_raw_loss / LOSS_SCALE
                total_loss = (1.0 - rl_step_ratio) * sft_loss + rl_step_ratio * rl_loss
            else:
                sft_loss = torch.zeros((), device=device, dtype=torch.float32)
                total_loss = rl_loss

            total_loss = total_loss / float(accum_iter)

        # 3) Optimize
        loss_value = float((total_loss * float(accum_iter)).item())
        if not math.isfinite(loss_value):
            if misc.is_main_process():
                print(f"[WARN] Loss is {loss_value}, skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_scaler(
            total_loss,
            optimizer,
            clip_grad=float(getattr(args, "grad_clip", 1.0)),
            parameters=model.parameters(),
            update_grad=do_step,
        )

        if do_step:
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            update_ema(ema_params, model_params, rate=float(getattr(args, "ema_rate", 0.9999)))

        # 4) Logging
        rl_loss_log = float(rl_loss.item())
        sft_loss_log = float(sft_loss.item())
        kl_pen_log = float(kl_pen.mean().item()) if use_kl else 0.0

        metric_logger.update(loss_total=loss_value)
        metric_logger.update(loss_rl=rl_loss_log)
        metric_logger.update(loss_sft=sft_loss_log)
        metric_logger.update(kl_pen=kl_pen_log)
        metric_logger.update(reward=rewards.mean().item())
        metric_logger.update(lr=float(optimizer.param_groups[0]["lr"]))

        if log_writer is not None and misc.is_main_process():
            step_idx = data_iter_step + epoch * len(data_loader)
            log_writer.add_scalar("train/loss_total", loss_value, step_idx)
            log_writer.add_scalar("train/loss_rl", rl_loss_log, step_idx)
            log_writer.add_scalar("train/reward_mean", rewards.mean().item(), step_idx)
            if use_kl:
                log_writer.add_scalar("train/kl_pen_mean", kl_pen_log, step_idx)

    metric_logger.synchronize_between_processes()
    if misc.is_main_process():
        print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}