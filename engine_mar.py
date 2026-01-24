# engine_mar.py
import math
import os
import sys
import time
import copy
import shutil
import inspect
from contextlib import nullcontext
from typing import Iterable, Tuple, Optional

import numpy as np
import torch
import cv2
import torch_fidelity

import util.misc as misc
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution


# ------------------------------------------------------------
# One-time warnings
# ------------------------------------------------------------
_WARNED_TEXT_POOLED_FALLBACK = False
_WARNED_TEXT_MASK_FALLBACK = False


def update_ema(target_params, source_params, rate=0.99):
    """Exponential moving average update."""
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def _is_dist_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _barrier():
    if _is_dist_ready():
        torch.distributed.barrier()


def _safe_to_device(x, device, non_blocking: bool = True):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)
    return x


def _supports_kwarg(fn, kw: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return kw in sig.parameters
    except Exception:
        return False


def _filter_kwargs_by_signature(fn, kwargs: dict) -> dict:
    """Drop unexpected kwargs to avoid TypeError across different implementations."""
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _autocast_ctx(device: torch.device, enabled: bool):
    """
    Unified autocast context.
    - CUDA: torch.amp.autocast("cuda", enabled=...)
    - CPU: no-op
    """
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()


def _unpack_batch(batch):
    """
    兼容多种 DataLoader 输出格式，返回：
      samples: Tensor [B, 3, H, W] 或 cached moments Tensor
      captions: list[str] (len=B)
      layout: Tensor [B, N, 5] or None
      layout_mask: Tensor [B, N] bool or None
      meta: list[dict] or None
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    if len(batch) < 2:
        raise ValueError(f"Unexpected batch len: {len(batch)} (need at least samples, captions)")

    samples = batch[0]
    captions = batch[1]

    layout = None
    layout_mask = None
    meta = None

    # COCO: (samples, captions, layout, layout_mask, meta)
    if len(batch) >= 4:
        layout = batch[2]
        layout_mask = batch[3]
    if len(batch) >= 5:
        meta = batch[4]

    # captions -> list[str]
    if isinstance(captions, (tuple, list)):
        captions = list(captions)
    else:
        captions = [captions]

    # light sanity
    if layout is not None:
        if not torch.is_tensor(layout):
            raise TypeError(f"layout must be a torch.Tensor, got {type(layout)}")
        if layout.dim() != 3 or layout.size(-1) != 5:
            raise ValueError(f"layout must be [B, N, 5], got {tuple(layout.shape)}")

    if layout_mask is not None:
        if not torch.is_tensor(layout_mask):
            raise TypeError(f"layout_mask must be a torch.Tensor, got {type(layout_mask)}")
        if layout_mask.dim() != 2:
            raise ValueError(f"layout_mask must be [B, N], got {tuple(layout_mask.shape)}")

    if meta is not None and (not isinstance(meta, (list, tuple))):
        raise TypeError(f"meta must be list/tuple if provided, got {type(meta)}")

    return samples, captions, layout, layout_mask, meta


def _boxes_to_xyxy(boxes: torch.Tensor, box_format: str) -> torch.Tensor:
    """
    boxes: [B, N, 4] assumed normalized to [0,1] (planner side)
    box_format:
      - 'cxcywh' (DETR default)
      - 'xywh'
      - 'xyxy'
    return: [B, N, 4] in xyxy, clamped to [0,1]
    """
    if box_format == "cxcywh":
        cx, cy, w, h = boxes.unbind(dim=-1)
        x0 = cx - 0.5 * w
        y0 = cy - 0.5 * h
        x1 = cx + 0.5 * w
        y1 = cy + 0.5 * h
        out = torch.stack([x0, y0, x1, y1], dim=-1)
    elif box_format == "xywh":
        x, y, w, h = boxes.unbind(dim=-1)
        out = torch.stack([x, y, x + w, y + h], dim=-1)
    elif box_format == "xyxy":
        out = boxes
    else:
        raise ValueError(f"Unknown planner_box_format={box_format}, expected cxcywh/xywh/xyxy")

    return out.clamp(0.0, 1.0)


@torch.no_grad()
def _detr_outputs_to_layout(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    num_classes: int,
    score_thresh: float,
    box_format: str,
    force_one: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DETR 风格输出 -> layout/layout_mask
      pred_logits: [B, N, C(+1)] (最后一类通常是 no-object)
      pred_boxes:  [B, N, 4]   (normalized)
    return:
      layout:      [B, N, 5] float32 => (cls, x0,y0,x1,y1)
      layout_mask: [B, N] bool
    """
    if pred_logits.dim() != 3:
        raise ValueError(f"pred_logits must be [B,N,C], got {tuple(pred_logits.shape)}")
    if pred_boxes.dim() != 3 or pred_boxes.size(-1) != 4:
        raise ValueError(f"pred_boxes must be [B,N,4], got {tuple(pred_boxes.shape)}")

    B, N, C = pred_logits.shape
    if num_classes <= 0:
        raise ValueError(f"num_classes must be >0, got {num_classes}")

    probs = pred_logits.float().softmax(dim=-1)  # [B,N,C]

    has_noobj = (C == num_classes + 1)

    if has_noobj:
        scores, labels = probs[..., :num_classes].max(dim=-1)  # [B,N]
        keep = scores >= float(score_thresh)
    else:
        scores, labels = probs.max(dim=-1)
        keep = scores >= float(score_thresh)

    if force_one:
        top1 = scores.argmax(dim=-1)  # [B]
        b = torch.arange(B, device=scores.device)
        keep[b, top1] = True

    xyxy = _boxes_to_xyxy(pred_boxes, box_format=box_format)  # [B,N,4]
    layout = torch.cat([labels.float().unsqueeze(-1), xyxy], dim=-1).contiguous()  # [B,N,5]
    layout_mask = keep.bool().contiguous()
    return layout, layout_mask


@torch.no_grad()
def _maybe_predict_layout_with_planner(args, text_emb_pooled: torch.Tensor, device: torch.device):
    """
    注意：planner 默认只吃 pooled text embedding: [B, D]。
    """
    planner = getattr(args, "planner", None)
    if planner is None:
        return None, None

    planner.eval()
    out = planner(text_emb_pooled.to(device, non_blocking=True))

    # Case 1: (layout, layout_mask)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        layout, layout_mask = out[0], out[1]
        layout = layout.to(device, non_blocking=True)
        layout_mask = layout_mask.to(device, non_blocking=True).bool()
        return layout, layout_mask

    # Case 2: dict
    if isinstance(out, dict):
        if "layout" in out:
            layout = out["layout"].to(device, non_blocking=True)
            layout_mask = out.get("layout_mask", None)
            if layout_mask is None:
                layout_mask = torch.ones(layout.shape[0], layout.shape[1], device=device, dtype=torch.bool)
            else:
                layout_mask = layout_mask.to(device, non_blocking=True).bool()
            return layout, layout_mask

        if ("pred_logits" in out) and ("pred_boxes" in out):
            logits = out["pred_logits"].to(device, non_blocking=True)
            boxes = out["pred_boxes"].to(device, non_blocking=True)

            C = logits.size(-1)
            cfg_num_classes = int(getattr(args, "planner_num_classes", max(C - 1, 1)))
            if C == cfg_num_classes + 1 or C == cfg_num_classes:
                num_classes = cfg_num_classes
            else:
                num_classes = max(C - 1, 1)

            score_thresh = float(getattr(args, "planner_score_thresh", 0.0))
            box_format = str(getattr(args, "planner_box_format", "xyxy"))
            force_one = bool(getattr(args, "planner_force_one", True))

            layout, layout_mask = _detr_outputs_to_layout(
                pred_logits=logits,
                pred_boxes=boxes,
                num_classes=num_classes,
                score_thresh=score_thresh,
                box_format=box_format,
                force_one=force_one,
            )
            return layout, layout_mask

    # Case 3: raw tensor fallback
    if torch.is_tensor(out):
        layout = out.to(device, non_blocking=True)
        layout_mask = torch.ones(layout.shape[0], layout.shape[1], device=device, dtype=torch.bool)
        return layout, layout_mask

    return None, None


@torch.no_grad()
def _encode_text_for_model_and_planner(text_encoder, captions, device, args):
    """
    统一处理 text_encoder 输出，支持：
      - pooled: [B, D]
      - tokens: [B, T, D]
    并尽量获取 attention_mask: [B,T]（token-level cross-attn 需要）

    返回：
      text_for_model: Tensor (2D 或 3D，取决于 args.clip_return_pooled / args.use_text_cross_attn)
      text_pooled:    Tensor [B, D] (planner 用；若不需要 planner 也可 None)
      text_mask:      Optional[Tensor] [B,T] bool (仅 token-level 时返回；否则 None)
    """
    global _WARNED_TEXT_POOLED_FALLBACK
    global _WARNED_TEXT_MASK_FALLBACK

    use_text_cross_attn = bool(getattr(args, "use_text_cross_attn", False))
    clip_return_pooled = bool(getattr(args, "clip_return_pooled", False))
    use_planner_for_train = bool(getattr(args, "use_planner_for_train", False)) and (getattr(args, "planner", None) is not None)

    if use_text_cross_attn and clip_return_pooled:
        raise ValueError("--use_text_cross_attn requires token-level CLIP; remove --clip_return_pooled.")

    want_tokens_for_model = (not clip_return_pooled)  # 默认 tokens；显式 pooled 则返回 pooled

    text_out = None
    text_mask = None

    if hasattr(text_encoder, "encode") and callable(getattr(text_encoder, "encode")):
        enc = text_encoder.encode

        enc_kwargs = {}
        if _supports_kwarg(enc, "return_tokens"):
            enc_kwargs["return_tokens"] = want_tokens_for_model
        if _supports_kwarg(enc, "return_attention_mask"):
            enc_kwargs["return_attention_mask"] = True

        # 有的旧实现不支持 kwargs
        try:
            text_out = enc(captions, **enc_kwargs) if len(enc_kwargs) > 0 else enc(captions)
        except TypeError:
            text_out = enc(captions)

        if isinstance(text_out, (tuple, list)) and len(text_out) >= 2:
            text_out, text_mask = text_out[0], text_out[1]
    else:
        raise TypeError("text_encoder must have callable .encode(captions)")

    if not torch.is_tensor(text_out):
        raise TypeError(f"text_encoder.encode returned {type(text_out)}, expected torch.Tensor")

    text_tokens = None
    text_pooled = None

    if text_out.ndim == 3:
        text_tokens = text_out
    elif text_out.ndim == 2:
        text_pooled = text_out
    else:
        raise RuntimeError(f"Unexpected text embedding ndim={text_out.ndim}, shape={tuple(text_out.shape)}")

    # sanitize mask
    if text_tokens is not None:
        if text_mask is not None and torch.is_tensor(text_mask) and text_mask.ndim == 2:
            if text_mask.shape[0] == text_tokens.shape[0] and text_mask.shape[1] == text_tokens.shape[1]:
                text_mask = text_mask.to(device=device, non_blocking=True).bool()
            else:
                text_mask = None
        else:
            text_mask = None

        if use_text_cross_attn and text_mask is None and (not _WARNED_TEXT_MASK_FALLBACK):
            print(
                "[WARN] text_encoder did not provide attention_mask. "
                "Cross-attn will treat all tokens as valid; padding may hurt. "
                "Consider upgrading encoder to return (tokens, attention_mask)."
            )
            _WARNED_TEXT_MASK_FALLBACK = True
    else:
        text_mask = None  # pooled 不需要 mask

    # 2) 如果启用 planner：强制拿 pooled
    if use_planner_for_train:
        if text_pooled is None:
            if hasattr(text_encoder, "encode_pooled") and callable(getattr(text_encoder, "encode_pooled")):
                text_pooled = text_encoder.encode_pooled(captions)
            else:
                # fallback：用 tokens 近似 pooled（用 mask 做加权 mean）
                if text_tokens is None and hasattr(text_encoder, "encode") and _supports_kwarg(text_encoder.encode, "return_tokens"):
                    tmp = text_encoder.encode(captions, return_tokens=True, return_attention_mask=True) \
                        if _supports_kwarg(text_encoder.encode, "return_attention_mask") else text_encoder.encode(captions, return_tokens=True)
                    if isinstance(tmp, (tuple, list)) and len(tmp) >= 2:
                        text_tokens = tmp[0]
                        text_mask = tmp[1].bool()
                    else:
                        text_tokens = tmp

                if text_tokens is None:
                    raise RuntimeError("Planner needs pooled embedding but cannot obtain it from text_encoder.")

                if not _WARNED_TEXT_POOLED_FALLBACK:
                    print("[WARN] text_encoder has no encode_pooled(); using masked-mean(tokens) as pooled for planner. "
                          "This may hurt planner quality.")
                    _WARNED_TEXT_POOLED_FALLBACK = True

                if text_mask is None:
                    text_pooled = text_tokens.mean(dim=1)
                else:
                    m = text_mask.to(dtype=text_tokens.dtype)  # [B,T]
                    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
                    text_pooled = (text_tokens * m.unsqueeze(-1)).sum(dim=1) / denom

    # 3) 选择 text_for_model
    if clip_return_pooled:
        if text_pooled is None:
            if hasattr(text_encoder, "encode_pooled") and callable(getattr(text_encoder, "encode_pooled")):
                text_pooled = text_encoder.encode_pooled(captions)
            else:
                if text_tokens is None:
                    raise RuntimeError("clip_return_pooled=True but no pooled embedding available.")
                if text_mask is None:
                    text_pooled = text_tokens.mean(dim=1)
                else:
                    m = text_mask.to(dtype=text_tokens.dtype)
                    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
                    text_pooled = (text_tokens * m.unsqueeze(-1)).sum(dim=1) / denom
        text_for_model = text_pooled
        text_mask_for_model = None
    else:
        text_for_model = text_tokens if text_tokens is not None else text_pooled
        if text_for_model is None:
            raise RuntimeError("No valid text embedding produced for model.")
        text_mask_for_model = text_mask  # only for tokens

    # move to device
    text_for_model = _safe_to_device(text_for_model, device, non_blocking=True)
    if text_pooled is not None:
        text_pooled = _safe_to_device(text_pooled, device, non_blocking=True)

    return text_for_model, text_pooled, text_mask_for_model


def train_one_epoch(
    model,
    vae,
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
    """
    - 默认用 GT layout（batch 内 layout/layout_mask）
    - 若 args.use_planner_for_train=True 且 args.planner 存在，则用 planner 预测覆盖 GT
    - 若 args.use_cached=True：batch[0] 应该是 cached moments（DiagonalGaussianDistribution 的参数）

    text_encoder:
      - 新版：encode() 可返回 tokens [B,T,D] 或 pooled [B,D]
      - 旧版：encode() 返回 pooled [B,D]
      - 新版可返回 attention_mask [B,T]（建议）
    """
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, captions, layout, layout_mask, _meta = _unpack_batch(batch)

        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        use_cached = bool(getattr(args, "use_cached", False))

        # NOTE: cached moments 也需要放到 device
        samples = _safe_to_device(samples, device, non_blocking=True)
        if layout is not None:
            layout = _safe_to_device(layout, device, non_blocking=True)
        if layout_mask is not None:
            layout_mask = _safe_to_device(layout_mask, device, non_blocking=True).bool()

        with torch.no_grad():
            if use_cached:
                moments = samples
                posterior = DiagonalGaussianDistribution(moments)
            else:
                posterior = vae.encode(samples)

            # SD-style latent scaling
            x = posterior.sample().mul_(0.18215)

            # debug probe
            if data_iter_step % 20 == 0:
                l_std = x.std().item()
                l_mean = x.mean().item()
                print(f"\n[VAE Check step={data_iter_step}] Mean: {l_mean:.4f} | Std: {l_std:.4f} (Expect ~1.0)")
                if torch.isnan(x).any():
                    print("❌❌❌ CRITICAL: VAE Latents contain NaN!")
                    sys.exit(1)

            # -------- text encode (model + planner) --------
            text_for_model, text_pooled, text_mask = _encode_text_for_model_and_planner(
                text_encoder=text_encoder,
                captions=captions,
                device=device,
                args=args,
            )

            # -------- optional planner overwrite GT layout --------
            if bool(getattr(args, "use_planner_for_train", False)):
                if text_pooled is None:
                    raise RuntimeError("use_planner_for_train=True but pooled text embedding is None.")
                pred_layout, pred_mask = _maybe_predict_layout_with_planner(
                    args, text_emb_pooled=text_pooled, device=device
                )
                if pred_layout is not None:
                    layout, layout_mask = pred_layout, pred_mask

        # -------------------------
        # forward
        # -------------------------
        use_latents_kw = _supports_kwarg(model.forward, "latents")
        if use_latents_kw:
            model_kwargs = dict(
                imgs=None,
                latents=x,
                text_emb=text_for_model,
                text_mask=text_mask,          # ✅ NEW
                layout=layout,
                layout_mask=layout_mask,
            )
        else:
            model_kwargs = dict(
                imgs=x,
                text_emb=text_for_model,
                text_mask=text_mask,          # ✅ NEW
                layout=layout,
                layout_mask=layout_mask,
            )

        model_kwargs = _filter_kwargs_by_signature(model.forward, model_kwargs)

        use_amp = bool(getattr(args, "amp", False)) and (device.type == "cuda")
        with _autocast_ctx(device, use_amp):
            loss = model(**model_kwargs)

        if isinstance(loss, dict):
            loss_total = loss.get("loss", None)
            if loss_total is None:
                raise ValueError("Model returned dict but missing key 'loss'.")
            loss_value = float(loss_total.item())
        else:
            loss_total = loss
            loss_value = float(loss_total.item())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss_scaler(
            loss_total,
            optimizer,
            clip_grad=getattr(args, "grad_clip", 0.0),
            parameters=model.parameters(),
            update_grad=True,
        )
        optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=getattr(args, "ema_rate", 0.9999))

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        avg_obj = None
        if layout_mask is not None:
            with torch.no_grad():
                avg_obj_local = float(layout_mask.sum(dim=1).float().mean().item())
            avg_obj = misc.all_reduce_mean(avg_obj_local)
            metric_logger.update(avg_obj=avg_obj)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
            if avg_obj is not None:
                log_writer.add_scalar("avg_obj", avg_obj, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(
    model_without_ddp,
    vae,
    ema_params,
    args,
    epoch,
    batch_size=16,
    log_writer=None,
    cfg=1.0,
    use_ema=True,
):
    """
    原 repo 的 class-conditional(ImageNet) 评估逻辑。
    你当前任务是 text/layout conditioning（COCO），这里保持原逻辑，仅做健壮性处理。
    """
    model_without_ddp.eval()

    world_size = misc.get_world_size()
    local_rank = misc.get_rank()

    num_steps = args.num_images // (batch_size * world_size) + 1
    save_folder = os.path.join(
        args.output_dir,
        "ariter{}-diffsteps{}-temp{}-{}cfg{}-image{}".format(
            args.num_iter,
            args.num_sampling_steps,
            args.temperature,
            args.cfg_schedule,
            cfg,
            args.num_images,
        ),
    )
    if use_ema:
        save_folder = save_folder + "_ema"
    if getattr(args, "evaluate", False):
        save_folder = save_folder + "_evaluate"

    if misc.is_main_process():
        os.makedirs(save_folder, exist_ok=True)
    _barrier()

    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    class_num = getattr(args, "class_num", None)
    if class_num is None:
        raise ValueError("args.class_num is required for class-conditional evaluate()")

    assert args.num_images % class_num == 0
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    used_time = 0.0
    gen_img_cnt = 0

    device = next(model_without_ddp.parameters()).device
    use_amp = bool(getattr(args, "amp", False)) and (device.type == "cuda")

    for i in range(num_steps):
        if misc.is_main_process():
            print(f"Generation step {i}/{num_steps}")

        start = world_size * batch_size * i + local_rank * batch_size
        end = world_size * batch_size * i + (local_rank + 1) * batch_size
        labels_gen = class_label_gen_world[start:end]
        labels_gen = torch.tensor(labels_gen, device=device).long()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            with _autocast_ctx(device, use_amp):
                sampled_tokens = model_without_ddp.sample_tokens(
                    bsz=batch_size,
                    num_iter=args.num_iter,
                    cfg=cfg,
                    cfg_schedule=args.cfg_schedule,
                    labels=labels_gen,
                    temperature=args.temperature,
                )
                sampled_images = vae.decode(sampled_tokens / 0.18215)

        if i >= 1:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            if misc.is_main_process():
                print(
                    f"Generating {gen_img_cnt} images takes {used_time:.5f} seconds, "
                    f"{used_time / gen_img_cnt:.5f} sec per image"
                )

        _barrier()

        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2

        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, f"{str(img_id).zfill(5)}.png"), gen_img)

    _barrier()
    time.sleep(2)

    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    if (log_writer is not None) and misc.is_main_process():
        if args.img_size == 256:
            input2 = None
            fid_statistics_file = "fid_stats/adm_in256_stats.npz"
        else:
            raise NotImplementedError

        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict["frechet_inception_distance"]
        inception_score = metrics_dict["inception_score_mean"]

        postfix = ""
        if use_ema:
            postfix += "_ema"
        if cfg != 1.0:
            postfix += f"_cfg{cfg}"

        log_writer.add_scalar(f"fid{postfix}", fid, epoch)
        log_writer.add_scalar(f"is{postfix}", inception_score, epoch)
        print(f"FID: {fid:.4f}, Inception Score: {inception_score:.4f}")

        try:
            shutil.rmtree(save_folder)
        except Exception:
            pass

    _barrier()
    time.sleep(2)


def cache_latents(vae, data_loader: Iterable, device: torch.device, args=None):
    """
    为 COCO caption-layout dataset 缓存 VAE moments（以及水平翻转的 moments_flip）。
    """
    if args is None:
        raise ValueError("cache_latents requires args with fields: cached_path, split (or args.data_split)")

    cached_root = getattr(args, "cached_path", None)
    if not cached_root:
        raise ValueError("args.cached_path is required for cache_latents()")

    split = getattr(args, "split", None) or getattr(args, "data_split", None) or getattr(args, "coco_split", None) or "train2017"

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"Caching({split}): "
    print_freq = 20

    vae.eval()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if not isinstance(batch, (tuple, list)) or len(batch) < 1:
            raise ValueError("cache_latents expects batch to be tuple/list with samples at index 0")

        meta = None
        if len(batch) >= 5:
            meta = batch[4]

        samples = batch[0]
        if not torch.is_tensor(samples):
            raise TypeError(f"cache_latents expects samples to be torch.Tensor image batch, got {type(samples)}")
        samples = samples.to(device, non_blocking=True)

        if meta is None:
            raise ValueError(
                "cache_latents requires meta to name files. "
                "Please make your dataset return meta with 'file_name' (recommended)."
            )
        if not isinstance(meta, (list, tuple)):
            raise TypeError(f"meta must be list/tuple, got {type(meta)}")
        if len(meta) != samples.size(0):
            raise ValueError(f"meta length {len(meta)} must equal batch size {samples.size(0)}")

        with torch.no_grad():
            posterior = vae.encode(samples)
            moments = posterior.parameters
            posterior_flip = vae.encode(samples.flip(dims=[3]))
            moments_flip = posterior_flip.parameters

        bsz = samples.size(0)

        for i in range(bsz):
            m = meta[i]
            if not isinstance(m, dict):
                raise TypeError(f"meta[{i}] must be dict, got {type(m)}")

            key = m.get("cache_key", None) or m.get("file_name", None)
            if key is None:
                raise KeyError("meta dict must contain 'cache_key' or 'file_name'")

            out_dir = os.path.join(cached_root, split)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, key + ".npz")

            np.savez(
                out_path,
                moments=moments[i].detach().float().cpu().numpy(),
                moments_flip=moments_flip[i].detach().float().cpu().numpy(),
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return
