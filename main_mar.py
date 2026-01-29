import argparse
import copy
import datetime
import inspect
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.vae import AutoencoderKL
from models import mar
from engine_mar import train_one_epoch, evaluate
from text_encoder_clip import HFCLIPTextEncoder

# Dataset
try:
    from coco_mini_dataset import build_coco_mini_dataset, coco_collate_fn  # noqa
except Exception:
    from coco_mini_dataset import build_coco_mini_dataset  # noqa

    def coco_collate_fn(batch):
        imgs, caps, layouts, masks, metas = zip(*batch)
        return (
            torch.stack(imgs, 0),
            list(caps),
            torch.stack(layouts, 0),
            torch.stack(masks, 0),
            list(metas),
        )


def get_args_parser():
    parser = argparse.ArgumentParser("MAR training with Diffusion Loss (COCO caption+layout)", add_help=False)

    # -------------------------
    # Core train
    # -------------------------
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=2, type=int)

    # Dataset subset (sanity/overfit). 0 = use full dataset
    parser.add_argument(
        "--num_train_samples",
        default=0,
        type=int,
        help="If >0: randomly subsample this many dataset items. 0 = use full dataset.",
    )
    parser.add_argument("--subset_seed", default=0, type=int)

    # Train mode
    parser.add_argument(
        "--train_mode",
        type=str,
        default="full",
        choices=["text_proj", "text_layout", "full"],
        help=(
            "text_proj: train only text-conditioning modules\n"
            "text_layout: train text + layout related modules\n"
            "full: train all parameters"
        ),
    )

    # Model
    parser.add_argument("--model", default="mar_base", type=str)

    # VAE
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument(
        "--vae_path",
        default="pretrained_models/vae/kl16.ckpt",
        type=str,
    )
    parser.add_argument("--vae_embed_dim", default=16, type=int)
    parser.add_argument("--vae_stride", default=16, type=int)
    parser.add_argument("--patch_size", default=1, type=int)

    # Eval / sampling
    parser.add_argument("--online_eval", action="store_true", help="Run class-conditional eval (legacy).")
    parser.add_argument("--evaluate", action="store_true", help="Run class-conditional eval (legacy).")
    parser.add_argument("--eval_freq", type=int, default=40)
    parser.add_argument("--eval_bsz", type=int, default=64)
    parser.add_argument("--num_iter", default=256, type=int)
    parser.add_argument("--num_images", default=1000, type=int)
    parser.add_argument("--cfg", default=2.9, type=float)
    parser.add_argument("--cfg_schedule", default="linear", type=str)
    parser.add_argument("--temperature", default=1.0, type=float)

    # Optim
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--lr", type=float, default=None, help="If set, override lr computed from blr.")
    parser.add_argument("--blr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, default="constant")
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--ema_rate", default=0.9999, type=float)

    # MAR params
    parser.add_argument("--mask_ratio_min", type=float, default=0.7)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--buffer_size", type=int, default=64)
    parser.add_argument("--label_drop_prob", default=0.0, type=float)

    # DiffLoss params
    parser.add_argument("--diffloss_d", type=int, default=6)
    parser.add_argument("--diffloss_w", type=int, default=1024)
    parser.add_argument("--num_sampling_steps", type=str, default="100")
    parser.add_argument("--diffusion_batch_mul", type=int, default=1)

    # -------------------------
    # CLIP text encoder
    # -------------------------
    parser.add_argument(
        "--clip_model_dir",
        default="pretrained_models/clip-vit-base-patch32",
        type=str,
    )

    parser.add_argument(
        "--clip_return_pooled",
        action="store_true",
        help="If set, CLIP encode() returns pooled [B,D]. Default: token-level [B,T,D].",
    )

    parser.add_argument(
        "--clip_l2_normalize",
        action="store_true",
        help="If set, L2-normalize CLIP outputs. Default: False.",
    )

    # ---------------------------
    # Architecture switches
    # ---------------------------

    # 2D RoPE
    parser.add_argument("--use_2d_rope", dest="use_2d_rope", action="store_true")
    parser.add_argument("--no_use_2d_rope", dest="use_2d_rope", action="store_false")
    parser.set_defaults(use_2d_rope=True)
    parser.add_argument("--rope_base", default=10000.0, type=float)

    # Text cross-attn
    parser.add_argument("--use_text_cross_attn", dest="use_text_cross_attn", action="store_true")
    parser.add_argument("--no_use_text_cross_attn", dest="use_text_cross_attn", action="store_false")
    parser.set_defaults(use_text_cross_attn=True)
    parser.add_argument("--text_cross_every_n_layers", default=1, type=int)

    # Layout-guided attention bias
    parser.add_argument("--use_layout_bias", dest="use_layout_bias", action="store_true")
    parser.add_argument("--no_use_layout_bias", dest="use_layout_bias", action="store_false")
    parser.set_defaults(use_layout_bias=True)
    parser.add_argument("--layout_bias_value", default=10000.0, type=float)

    parser.add_argument("--layout_bias_on_encoder", dest="layout_bias_on_encoder", action="store_true")
    parser.add_argument("--no_layout_bias_on_encoder", dest="layout_bias_on_encoder", action="store_false")
    parser.set_defaults(layout_bias_on_encoder=True)

    parser.add_argument("--layout_bias_on_decoder", dest="layout_bias_on_decoder", action="store_true")
    parser.add_argument("--no_layout_bias_on_decoder", dest="layout_bias_on_decoder", action="store_false")
    parser.set_defaults(layout_bias_on_decoder=True)

    parser.add_argument("--keep_learned_pos_emb", action="store_true")

    # -------------------------
    # Dataset (COCO-style)
    # -------------------------
    parser.add_argument("--coco_root", default="data/coco", type=str)
    parser.add_argument(
        "--coco_split",
        default="val2017",
        type=str,
        choices=["val2017", "train2017"],
    )
    parser.add_argument("--max_objects", default=16, type=int)
    parser.add_argument("--random_flip", action="store_true")

    parser.add_argument("--class_num", default=1000, type=int)

    # IO
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument(
        "--resume",
        default="pretrained_models/mar-base.safetensors",
        type=str,
    )
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", type=str)

    # Caching
    parser.add_argument("--use_cached", action="store_true", dest="use_cached")
    parser.set_defaults(use_cached=False)
    parser.add_argument("--cached_path", default="", type=str)

    parser.add_argument("--save_last_freq", type=int, default=5)

    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--amp", action="store_true")

    return parser


def _try_resume(model_without_ddp, args, device):
    if not args.resume:
        return None, None, False

    resume = args.resume

    # A) dir + checkpoint-last.pth
    if os.path.isdir(resume):
        ckpt_path = os.path.join(resume, "checkpoint-last.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
            model_params = list(model_without_ddp.parameters())
            ema_state_dict = checkpoint.get("model_ema", None)
            if ema_state_dict is not None:
                ema_params = [ema_state_dict[name].to(device) for name, _ in model_without_ddp.named_parameters()]
            else:
                ema_params = copy.deepcopy(model_params)
            print(f"Resume checkpoint dir: {resume} (strict=False)")
            return checkpoint, (model_params, ema_params), True

    # B) file
    if os.path.isfile(resume):
        lower = resume.lower()

        if lower.endswith(".pth") or lower.endswith(".pt"):
            ckpt = torch.load(resume, map_location="cpu")
            if isinstance(ckpt, dict) and "model" in ckpt:
                model_without_ddp.load_state_dict(ckpt["model"], strict=False)
            elif isinstance(ckpt, dict):
                model_without_ddp.load_state_dict(ckpt, strict=False)
            else:
                raise ValueError("Unsupported checkpoint format for .pth/.pt")
            print(f"Resume model weights from file: {resume} (strict=False)")
            return ckpt, None, True

        if lower.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(resume)
            model_without_ddp.load_state_dict(sd, strict=False)
            print(f"Resume model weights from safetensors: {resume} (strict=False)")
            return sd, None, True

    print(f"[Resume] Not found or unsupported: {resume}")
    return None, None, False


def _build_dataset_from_signature(args):
    sig = inspect.signature(build_coco_mini_dataset)
    params = sig.parameters

    kwargs = {}
    if "root" in params:
        kwargs["root"] = args.coco_root
    if "split" in params:
        kwargs["split"] = args.coco_split
    if "image_size" in params:
        kwargs["image_size"] = args.img_size

    if "max_objects" in params:
        kwargs["max_objects"] = args.max_objects
    elif "max_objs" in params:
        kwargs["max_objs"] = args.max_objects

    if "random_flip" in params:
        kwargs["random_flip"] = args.random_flip

    if "use_cached" in params:
        kwargs["use_cached"] = bool(getattr(args, "use_cached", False))
    if "cached_path" in params:
        kwargs["cached_path"] = str(getattr(args, "cached_path", "")) if getattr(args, "use_cached", False) else None
    if "cached_use_flip" in params:
        kwargs["cached_use_flip"] = bool(getattr(args, "random_flip", False))

    return build_coco_mini_dataset(**kwargs)


def _maybe_subsample_dataset(dataset, num_train_samples: int, subset_seed: int):
    if num_train_samples is None or num_train_samples <= 0:
        return dataset
    n = len(dataset)
    if num_train_samples >= n:
        print(f"[Dataset] num_train_samples={num_train_samples} >= len(dataset)={n}, use full dataset.")
        return dataset
    rng = np.random.RandomState(int(subset_seed))
    idx = rng.choice(n, size=int(num_train_samples), replace=False)
    idx = np.sort(idx).tolist()
    print(f"[Dataset] Subsample: {num_train_samples}/{n} items (seed={subset_seed})")
    return Subset(dataset, idx)


def _set_trainable_params(model: torch.nn.Module, train_mode: str):
    for p in model.parameters():
        p.requires_grad = False

    if train_mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    # text-only conditioning modules
    allow_keys = [
        "text_proj",
        "text",
        "cross_attn",
        "crossattn",
        "txt_attn",
        "t2i",
        "adapter",
        "adaln",
        "ada_ln",
        "gate",
        "gated",
        "cond",
        "condition",
        "context",
    ]

    # add layout-related modules
    if train_mode == "text_layout":
        allow_keys += [
            "layout",
            "bbox",
            "box",
            "obj",
            "object",
            "bias",
            "attn_bias",
            "layout_bias",
            "layout_gate",
        ]

    for name, p in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in allow_keys):
            p.requires_grad = True


def _print_trainable_stats(model: torch.nn.Module):
    total = 0
    trainable = 0
    names = []
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            names.append(n)
    print(f"[Trainable] {trainable/1e6:.3f}M / {total/1e6:.3f}M trainable.")
    for n in names[:40]:
        print(f"  - {n}")
    if len(names) > 40:
        print(f"  ... ({len(names)-40} more)")


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[TF32] Enabled.")

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.log_dir is None:
        args.log_dir = args.output_dir

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if getattr(args, "use_cached", False) and not getattr(args, "cached_path", ""):
        raise ValueError("--use_cached requires --cached_path to be set.")

    if args.use_text_cross_attn and args.clip_return_pooled:
        raise ValueError(
            "--use_text_cross_attn requires token-level CLIP embeddings. "
            "Please remove --clip_return_pooled or pass --no_use_text_cross_attn."
        )

    # -------------------------
    # Dataset
    # -------------------------
    dataset_train = _build_dataset_from_signature(args)
    dataset_train = _maybe_subsample_dataset(dataset_train, args.num_train_samples, args.subset_seed)
    print(dataset_train)

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=coco_collate_fn,
    )

    # -------------------------
    # VAE + Text Encoder
    # -------------------------
    vae = AutoencoderKL(
        embed_dim=args.vae_embed_dim,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=args.vae_path,
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    te_kwargs = {
        "model_dir": args.clip_model_dir,
        "device": device,
    }
    try:
        te_sig = inspect.signature(HFCLIPTextEncoder.__init__)
        if "return_tokens_default" in te_sig.parameters:
            te_kwargs["return_tokens_default"] = (not args.clip_return_pooled)
        if "l2_normalize_default" in te_sig.parameters:
            te_kwargs["l2_normalize_default"] = bool(args.clip_l2_normalize)
    except Exception:
        pass

    text_encoder = HFCLIPTextEncoder(**te_kwargs)

    if args.use_text_cross_attn:
        try:
            with torch.no_grad():
                _probe = text_encoder.encode(["a photo"])
            if torch.is_tensor(_probe) and _probe.dim() == 2:
                print(
                    "\n[WARN] text_encoder.encode() returned pooled [B,D]. "
                    "Cross-attn will still run (context length=1), but token-level [B,T,D] is recommended.\n"
                )
        except Exception:
            pass

    # -------------------------
    # MAR model
    # -------------------------
    model = mar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,

        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,

        cond_dim=getattr(text_encoder, "text_dim", None),
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,

        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,

        grad_checkpointing=args.grad_checkpointing,

        use_2d_rope=bool(args.use_2d_rope),
        rope_base=float(args.rope_base),
        disable_learned_pos_emb_when_rope=(not bool(args.keep_learned_pos_emb)),

        use_text_cross_attn=bool(args.use_text_cross_attn),
        text_cross_every_n_layers=int(args.text_cross_every_n_layers),

        use_layout_bias=bool(args.use_layout_bias),
        layout_bias_value=float(args.layout_bias_value),
        layout_bias_on_encoder=bool(args.layout_bias_on_encoder),
        layout_bias_on_decoder=bool(args.layout_bias_on_decoder),
    )

    _set_trainable_params(model, args.train_mode)
    _print_trainable_stats(model)

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256.0
        lr_src = "blr_scaled"
    else:
        lr_src = "explicit_lr"
    print(f"[LR] {args.lr:.8f} ({lr_src}), effective batch size={eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler(enabled=args.amp)

    print("[LR check]", [g["lr"] for g in optimizer.param_groups])

    # -------------------------
    # Resume
    # -------------------------
    checkpoint, _maybe_params, resumed = _try_resume(model_without_ddp, args, device)

    if resumed and isinstance(checkpoint, dict) and ("optimizer" in checkpoint) and ("epoch" in checkpoint):
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint.get("model_ema", None)
        if ema_state_dict is not None:
            ema_params = [ema_state_dict[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        else:
            ema_params = copy.deepcopy(model_params)

        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = int(checkpoint["epoch"]) + 1
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        print("Resumed with optimizer & scaler.")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        if resumed:
            print("Resumed model weights only (no optimizer state).")
        else:
            print("Training from scratch.")

    if args.evaluate:
        print(
            "\n[WARN] args.evaluate=True triggers legacy evaluate().\n"
            "       It does NOT evaluate COCO text/layout conditioning.\n"
        )
        torch.cuda.empty_cache()
        evaluate(
            model_without_ddp,
            vae,
            ema_params,
            args,
            0,
            batch_size=args.eval_bsz,
            log_writer=log_writer,
            cfg=args.cfg,
            use_ema=True,
        )
        return

    # Training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            vae,
            model_params,
            ema_params,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            text_encoder=text_encoder,
            log_writer=log_writer,
            args=args,
        )

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                ema_params=ema_params,
                epoch_name="last",
            )

        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            print(
                "\n[WARN] online_eval=True triggers legacy evaluate().\n"
                "       It does NOT evaluate COCO text/layout task.\n"
            )
            torch.cuda.empty_cache()
            evaluate(
                model_without_ddp,
                vae,
                ema_params,
                args,
                epoch,
                batch_size=args.eval_bsz,
                log_writer=log_writer,
                cfg=1.0,
                use_ema=True,
            )
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        args.log_dir = args.output_dir
    main(args)