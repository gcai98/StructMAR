# main_rl.py
import os
# ==============================================================================
# [CRITICAL] MUST BE THE FIRST LINES
# ==============================================================================
os.environ["YOLO_VERBOSE"] = "False"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["ULTRALYTICS_NO_AUTOUPDATES"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

import argparse
import copy
import datetime
import inspect
import numpy as np
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset  # ✅ NEW

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.vae import AutoencoderKL
from models import mar
from text_encoder_clip import HFCLIPTextEncoder

from engine_rl import train_one_epoch_grpo
from rewards.layout_scorer import LayoutScorer

try:
    from coco_mini_dataset import build_coco_mini_dataset, coco_collate_fn
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from coco_mini_dataset import build_coco_mini_dataset, coco_collate_fn


def _supports_kwarg(fn, kw: str) -> bool:
    try:
        return kw in inspect.signature(fn).parameters
    except Exception:
        return False


def _filter_kwargs_by_signature(fn, kwargs: dict) -> dict:
    try:
        allowed = set(inspect.signature(fn).parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def get_args_parser():
    parser = argparse.ArgumentParser("MAR Reinforcement Learning (GRPO)", add_help=False)

    # -------------------------
    # RL / GRPO
    # -------------------------
    parser.add_argument("--group_size", default=4, type=int, help="Number of samples per prompt")
    parser.add_argument("--rl_step_ratio", default=0.5, type=float,
                        help="Mixture: total = (1-rl_step_ratio)*SFT + rl_step_ratio*RL")
    parser.add_argument("--grad_clip", default=1.0, type=float)

    parser.add_argument("--adv_norm", default="zscore", type=str,
                        choices=["none", "center", "zscore"],
                        help="How to normalize advantage within each group")
    parser.add_argument("--reward_clip", default=10.0, type=float,
                        help="Clip rewards to [-reward_clip, reward_clip]")

    # KL to ref (GRPO stability)
    parser.add_argument("--kl_coef", default=0.02, type=float,
                        help="KL coefficient to reference model (0 disables ref/kl)")

    # Sampling knobs (policy rollout)
    parser.add_argument("--num_iter", default=256, type=int)
    parser.add_argument("--cfg", default=1.0, type=float, help="CFG used for rollout sampling (recommend 1.0 for RL)")
    parser.add_argument("--cfg_schedule", default="linear", type=str, choices=["linear", "constant"])
    parser.add_argument("--temperature", default=1.0, type=float)

    # -------------------------
    # Train
    # -------------------------
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)

    # -------------------------
    # MAR
    # -------------------------
    parser.add_argument("--model", default="mar_base", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--vae_embed_dim", default=16, type=int)
    parser.add_argument("--vae_stride", default=16, type=int)
    parser.add_argument("--patch_size", default=1, type=int)

    parser.add_argument("--buffer_size", default=64, type=int)
    parser.add_argument("--diffloss_d", default=3, type=int)
    parser.add_argument("--diffloss_w", default=1024, type=int)
    parser.add_argument("--num_sampling_steps", default="100", type=str)
    parser.add_argument("--diffusion_batch_mul", default=4, type=int)
    parser.add_argument("--grad_checkpointing", action="store_true")

    # RL 默认不要 cond dropout（和你 SFT 默认一致）
    parser.add_argument("--label_drop_prob", default=0.0, type=float)

    # ---------------------------
    # Architecture switches
    # ---------------------------
    parser.add_argument("--use_2d_rope", dest="use_2d_rope", action="store_true")
    parser.add_argument("--no_use_2d_rope", dest="use_2d_rope", action="store_false")
    parser.set_defaults(use_2d_rope=True)
    parser.add_argument("--rope_base", default=10000.0, type=float)
    parser.add_argument("--keep_learned_pos_emb", action="store_true")

    parser.add_argument("--use_text_cross_attn", dest="use_text_cross_attn", action="store_true")
    parser.add_argument("--no_use_text_cross_attn", dest="use_text_cross_attn", action="store_false")
    parser.set_defaults(use_text_cross_attn=True)
    parser.add_argument("--text_cross_every_n_layers", default=1, type=int)

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

    # -------------------------
    # Optim
    # -------------------------
    parser.add_argument("--lr", default=1e-6, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--ema_rate", default=0.9999, type=float)

    # -------------------------
    # IO
    # -------------------------
    parser.add_argument("--output_dir", default="./output_rl_grpo", type=str)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--resume", default="", required=True, help="SFT checkpoint (.pth)")

    # -------------------------
    # Data
    # -------------------------
    parser.add_argument("--coco_root", default=r"C:\caogang\coco-mini", type=str)
    parser.add_argument("--coco_split", default="train2017", type=str)
    parser.add_argument("--max_objects", default=16, type=int)
    parser.add_argument("--random_flip", action="store_true")

    # ✅ NEW: dataset subset (sanity). 0 = full dataset
    parser.add_argument(
        "--num_train_samples",
        default=0,
        type=int,
        help="If >0: randomly subsample this many dataset items. 0 = use full dataset.",
    )
    parser.add_argument("--subset_seed", default=0, type=int)

    # cached latents (optional)
    parser.add_argument("--use_cached", action="store_true", dest="use_cached")
    parser.set_defaults(use_cached=False)
    parser.add_argument("--cached_path", default="", type=str)
    parser.add_argument("--cached_use_flip", action="store_true",
                        help="If cached moments include moments_flip, enable this with random_flip")

    # -------------------------
    # Paths
    # -------------------------
    parser.add_argument("--vae_path",
                        default=r"C:\caogang\MAR\mar-main-DIT\mar-main\pretrained_models\vae\kl16.ckpt",
                        type=str)
    parser.add_argument("--clip_model_dir",
                        default=r"C:\caogang\HuggingFace_model\models--openai--clip-vit-base-patch32",
                        type=str)

    # CLIP output control
    parser.add_argument("--clip_return_pooled", action="store_true")
    parser.add_argument("--clip_l2_normalize", action="store_true")

    # Reward model
    parser.add_argument("--detector_path",
                        default=r"C:\caogang\fasterrcnn_resnet50\fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
                        type=str)

    # runtime
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")

    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--amp", action="store_true")

    # distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser


def _setup_device(args) -> torch.device:
    distributed = bool(getattr(args, "distributed", False))
    if args.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if distributed:
        gpu = getattr(args, "gpu", None)
        if gpu is None or gpu < 0:
            gpu = int(getattr(args, "local_rank", 0))
        torch.cuda.set_device(gpu)
        return torch.device(f"cuda:{gpu}")
    return torch.device("cuda:0")


def main(args):
    misc.init_distributed_mode(args)
    distributed = bool(getattr(args, "distributed", False))
    rank = misc.get_rank()

    if args.log_dir is None:
        args.log_dir = args.output_dir

    if misc.is_main_process():
        print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(", ", ",\n"))

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if misc.is_main_process():
            print("[TF32] Enabled.")

    device = _setup_device(args)

    seed = int(args.seed) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    if misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

    log_writer = None
    if misc.is_main_process() and args.log_dir is not None:
        log_writer = SummaryWriter(log_dir=args.log_dir)

    # cross-attn needs token-level
    if args.use_text_cross_attn and args.clip_return_pooled:
        raise ValueError(
            "--use_text_cross_attn requires token-level CLIP embeddings. "
            "Please remove --clip_return_pooled or pass --no_use_text_cross_attn."
        )

    # -------------------------
    # Dataset
    # -------------------------
    if getattr(args, "use_cached", False) and not getattr(args, "cached_path", ""):
        raise ValueError("--use_cached requires --cached_path")

    ds_kwargs = dict(
        root=args.coco_root,
        split=args.coco_split,
        image_size=args.img_size,
        max_objects=args.max_objects,
        random_flip=bool(args.random_flip),
        use_cached=bool(getattr(args, "use_cached", False)),
        cached_path=str(getattr(args, "cached_path", "")) if getattr(args, "use_cached", False) else None,
        cached_use_flip=bool(getattr(args, "cached_use_flip", False)),
    )
    ds_kwargs = _filter_kwargs_by_signature(build_coco_mini_dataset, ds_kwargs)

    dataset_train = build_coco_mini_dataset(**ds_kwargs)

    # ✅ NEW: optional subsample for sanity / speed
    if getattr(args, "num_train_samples", 0) and int(args.num_train_samples) > 0:
        n = len(dataset_train)
        k = int(args.num_train_samples)
        if k >= n:
            if misc.is_main_process():
                print(f"[Dataset] num_train_samples={k} >= len(dataset)={n}, use full dataset.")
        else:
            rng = np.random.RandomState(int(getattr(args, "subset_seed", 0)))
            idx = rng.choice(n, size=k, replace=False)
            idx = np.sort(idx).tolist()
            dataset_train = Subset(dataset_train, idx)
            if misc.is_main_process():
                print(f"[Dataset] Subsample: {k}/{n} items (seed={getattr(args, 'subset_seed', 0)})")

    if distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=misc.get_world_size(),
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_mem),
        drop_last=True,
        collate_fn=coco_collate_fn,
    )

    # -------------------------
    # VAE
    # -------------------------
    if misc.is_main_process():
        print("Loading VAE...")
    vae = AutoencoderKL(
        embed_dim=args.vae_embed_dim,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=args.vae_path,
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    # -------------------------
    # CLIP Text Encoder
    # -------------------------
    if misc.is_main_process():
        print("Loading CLIP Text Encoder...")

    te_kwargs = {"model_dir": args.clip_model_dir, "device": device}
    try:
        te_sig = inspect.signature(HFCLIPTextEncoder.__init__)
        if "return_tokens_default" in te_sig.parameters:
            te_kwargs["return_tokens_default"] = (not args.clip_return_pooled)
        if "l2_normalize_default" in te_sig.parameters:
            te_kwargs["l2_normalize_default"] = bool(args.clip_l2_normalize)
    except Exception:
        pass

    text_encoder = HFCLIPTextEncoder(**te_kwargs)

    # -------------------------
    # MAR Model
    # -------------------------
    if not args.resume:
        raise ValueError("RL training MUST resume from SFT checkpoint!")

    if misc.is_main_process():
        print("Loading MAR Model...")
        print(f"Resume checkpoint: {args.resume}")

    model = mar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        grad_checkpointing=args.grad_checkpointing,

        label_drop_prob=float(args.label_drop_prob),

        cond_dim=getattr(text_encoder, "text_dim", None),
        layout_class_num=80,

        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,

        buffer_size=args.buffer_size,

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

    checkpoint = torch.load(args.resume, map_location="cpu")
    sd = checkpoint["model"] if isinstance(checkpoint, dict) and ("model" in checkpoint) else checkpoint
    msg = model.load_state_dict(sd, strict=False)
    print('msg.unexpected_keys[:50]:')
    print(msg.unexpected_keys[:50])

    if misc.is_main_process():
        print(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")

    model.to(device)
    model.train()

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    # -------------------------
    # Reference Model (frozen) for KL
    # -------------------------
    ref_model = None
    if float(getattr(args, "kl_coef", 0.0)) > 0.0:
        if misc.is_main_process():
            print("[GRPO] Building frozen reference model for KL...")
        ref_model = copy.deepcopy(model_without_ddp).to(device).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # -------------------------
    # Reward Model
    # -------------------------
    if misc.is_main_process():
        print(f"Loading Reward Model from local: {args.detector_path}")
    reward_model = LayoutScorer(detector_path=args.detector_path, device=device)

    # -------------------------
    # Optimizer / EMA
    # -------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model_without_ddp.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    loss_scaler = NativeScaler(enabled=args.amp)


    model_params = list(model_without_ddp.parameters())
    ema_params = [p.clone().detach() for p in model_params]

    if misc.is_main_process():
        print(f"Start RL training for {args.epochs} epochs (distributed={distributed})")
    start_time = time.time()

    for epoch in range(int(args.epochs)):
        if distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        _train_stats = train_one_epoch_grpo(
            model=model,
            ref_model=ref_model,
            vae=vae,
            reward_model=reward_model,
            model_params=model_params,
            ema_params=ema_params,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            text_encoder=text_encoder,
            log_writer=log_writer,
            args=args,
        )

        if misc.is_main_process():
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                ema_params=ema_params,
                epoch_name=f"rl-epoch{epoch}",
            )
            print(f"Saved checkpoint: rl-epoch{epoch}")

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        args.log_dir = args.output_dir
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
