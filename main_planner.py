import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from coco_mini_dataset import build_coco_mini_dataset
from text_encoder_clip import HFCLIPTextEncoder

from models.planner.planner import LayoutPlanner
from engine_planner import (
    CriterionConfig,
    LayoutSetCriterion,
    train_one_epoch_planner,
)


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
    parser = argparse.ArgumentParser("Planner training (text -> layout)", add_help=False)

    parser.add_argument(
        "--use_preset",
        action="store_true",
        help="Use hard-coded preset args (ignore most CLI args).",
    )

    # dataset
    parser.add_argument("--coco_root", default="data/coco", type=str)
    parser.add_argument("--coco_split", default="val2017", type=str, choices=["val2017", "train2017"])
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--max_objects", default=16, type=int, help="num_queries / K")

    # text encoder
    parser.add_argument(
        "--clip_model_dir",
        default="pretrained_models/clip-vit-base-patch32",
        type=str,
    )

    # planner model
    parser.add_argument("--cond_dim", default=512, type=int, help="text embedding dim")
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--num_classes", default=80, type=int)

    # criterion / matching
    parser.add_argument("--cost_class", default=1.0, type=float)
    parser.add_argument("--cost_bbox", default=5.0, type=float)
    parser.add_argument("--bbox_loss_coef", default=5.0, type=float)
    parser.add_argument("--no_object_coef", default=0.1, type=float)

    # optimization
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.02, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # resume / save
    parser.add_argument("--resume", default="", type=str, help="path to planner checkpoint .pth")
    parser.add_argument("--output_dir", default="./output_planner", type=str)
    parser.add_argument("--log_dir", default="./output_planner", type=str)
    parser.add_argument("--save_freq", default=1, type=int)

    # misc
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1, type=int)

    # distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser


def _apply_preset(args):
    """
    Apply generic preset arguments.
    """
    args.coco_root = "data/coco"
    args.coco_split = "val2017"
    args.max_objects = 16
    args.batch_size = 64
    args.epochs = 5
    args.lr = 1e-4
    args.output_dir = "./output_planner"
    args.log_dir = args.output_dir
    return args


def try_resume(model, optimizer, args):
    if not args.resume:
        return 0
    if not os.path.isfile(args.resume):
        print(f"[WARN] resume not found: {args.resume}")
        return 0
    ckpt = torch.load(args.resume, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt and optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Resumed planner from {args.resume}, start_epoch={start_epoch}")
        return start_epoch
    else:
        model.load_state_dict(ckpt, strict=True)
        print(f"Resumed planner weights from {args.resume}")
        return 0


def save_ckpt(model, optimizer, epoch, args):
    if not misc.is_main_process():
        return
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"planner-epoch{epoch:04d}.pth")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "planner-last.pth"),
    )
    print(f"Saved: {path}")


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    dataset_train = build_coco_mini_dataset(
        root=args.coco_root,
        split=args.coco_split,
        image_size=args.img_size,
        max_objects=args.max_objects,
    )

    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=coco_collate_fn,
    )

    text_encoder = HFCLIPTextEncoder(
        model_dir=args.clip_model_dir,
        device=args.device,
    )

    model = LayoutPlanner(
        cond_dim=args.cond_dim,
        hidden_dim=args.hidden_dim,
        num_queries=args.max_objects,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.0,
    ).to(device)

    crit_cfg = CriterionConfig(
        num_classes=args.num_classes,
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        bbox_loss_coef=args.bbox_loss_coef,
        no_object_coef=args.no_object_coef,
    )
    criterion = LayoutSetCriterion(crit_cfg).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    start_epoch = try_resume(model_without_ddp, optimizer, args)

    print(f"Start planner training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch_planner(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            text_encoder=text_encoder,
            loss_scaler=loss_scaler,
            max_norm=args.grad_clip,
            log_writer=log_writer,
            print_freq=20,
        )

        if (epoch % args.save_freq) == 0 or (epoch + 1) == args.epochs:
            save_ckpt(model_without_ddp, optimizer, epoch, args)

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Planner training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Planner training", parents=[get_args_parser()])
    args = parser.parse_args()

    if getattr(args, "use_preset", False):
        args = _apply_preset(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)