# coco_mini_dataset.py
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _center_square_crop_params(h: int, w: int) -> Tuple[int, int, int]:
    size = min(h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return top, left, size


def _img_to_tensor_minus1_1(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32)  # [H, W, 3], 0..255
    arr = arr / 127.5 - 1.0
    arr = np.transpose(arr, (2, 0, 1))  # [3, H, W]
    return torch.from_numpy(arr).contiguous()


@dataclass
class CocoMiniConfig:
    root: str
    split: str = "val2017"          # "val2017" or "train2017"
    image_size: int = 256
    max_objects: int = 16

    random_flip: bool = False
    keep_crowd: bool = False
    min_box_area: float = 1.0       # pixels^2, after intersection with crop
    sort_by_area: bool = True

    # === cache support ===
    use_cached: bool = False
    cached_path: Optional[str] = None
    cached_use_flip: bool = False   # if True, read moments_flip when flipped


class CocoMiniCaptionLayoutDataset(Dataset):
    """
    Returns:
      samples:
        - if use_cached=False: img Tensor [3,S,S] in [-1,1]
        - if use_cached=True : moments Tensor [C,H,W] (as saved in .npz)  (caller wraps DiagonalGaussianDistribution)
      caption: str
      layout: Tensor [max_objects, 5] float32 => [cls, x0, y0, x1, y1] normalized xyxy in crop-square space
      layout_mask: Tensor [max_objects] bool (True=valid)
      meta: dict
    """

    def __init__(self, cfg: CocoMiniConfig):
        super().__init__()
        self.cfg = cfg

        self.img_dir = os.path.join(cfg.root, cfg.split)
        self.ann_dir = os.path.join(cfg.root, "annotations")
        self.cap_path = os.path.join(self.ann_dir, f"captions_{cfg.split}.json")
        self.ins_path = os.path.join(self.ann_dir, f"instances_{cfg.split}.json")

        if not cfg.use_cached:
            if not os.path.isdir(self.img_dir):
                raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        else:
            if not cfg.cached_path:
                raise ValueError("use_cached=True requires cached_path")
            if not os.path.isdir(cfg.cached_path):
                raise FileNotFoundError(f"cached_path not found: {cfg.cached_path}")

        if not os.path.isfile(self.cap_path):
            raise FileNotFoundError(f"Captions json not found: {self.cap_path}")
        if not os.path.isfile(self.ins_path):
            raise FileNotFoundError(f"Instances json not found: {self.ins_path}")

        cap = _load_json(self.cap_path)
        ins = _load_json(self.ins_path)

        self.image_id_to_info: Dict[int, dict] = {int(img["id"]): img for img in cap.get("images", [])}
        if len(self.image_id_to_info) == 0:
            raise RuntimeError(f"No images found in captions json: {self.cap_path}")

        raw_caption_anns = cap.get("annotations", [])
        self.caption_anns: List[dict] = [a for a in raw_caption_anns if int(a["image_id"]) in self.image_id_to_info]
        if len(self.caption_anns) == 0:
            raise RuntimeError(f"No valid caption annotations found in {self.cap_path}")

        self.image_id_to_instances: Dict[int, List[dict]] = {}
        for ann in ins.get("annotations", []):
            img_id = int(ann["image_id"])
            self.image_id_to_instances.setdefault(img_id, []).append(ann)

        cats = ins.get("categories", [])
        if len(cats) == 0:
            raise RuntimeError(f"No categories found in {self.ins_path}")
        cats_sorted = sorted(cats, key=lambda x: x["id"])
        self.cat_id_to_contig: Dict[int, int] = {int(c["id"]): i for i, c in enumerate(cats_sorted)}
        self.contig_to_cat_name: List[str] = [c.get("name", str(c["id"])) for c in cats_sorted]

    def __len__(self) -> int:
        return len(self.caption_anns)

    def _load_image(self, file_name: str) -> Image.Image:
        path = os.path.join(self.img_dir, file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path).convert("RGB")

    def _load_cached_moments(self, split: str, file_name: str, flipped: bool) -> torch.Tensor:
        # save layout: cached_path/{split}/{file_name}.npz
        npz_path = os.path.join(self.cfg.cached_path, split, file_name + ".npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Cached moments not found: {npz_path}")

        d = np.load(npz_path)
        key = "moments_flip" if (flipped and self.cfg.cached_use_flip) else "moments"
        if key not in d:
            raise KeyError(f"{npz_path} missing key '{key}'. keys={list(d.keys())}")

        moments = d[key]  # usually [C,H,W] or [2C,H,W] depending on VAE impl
        moments_t = torch.from_numpy(moments).float().contiguous()
        return moments_t

    def _encode_layout_for_image(
        self,
        image_id: int,
        crop_tls: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        top, left, size = crop_tls
        max_objects = self.cfg.max_objects

        insts = self.image_id_to_instances.get(image_id, [])
        items = []
        num_total_after_filter = 0

        for ann in insts:
            if (not self.cfg.keep_crowd) and (ann.get("iscrowd", 0) == 1):
                continue

            bbox = ann.get("bbox", None)  # [x, y, w, h]
            if bbox is None:
                continue

            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            if w <= 0 or h <= 0:
                continue

            num_total_after_filter += 1

            x0, y0 = x, y
            x1, y1 = x + w, y + h

            ix0 = max(x0, float(left))
            iy0 = max(y0, float(top))
            ix1 = min(x1, float(left + size))
            iy1 = min(y1, float(top + size))

            if ix1 <= ix0 or iy1 <= iy0:
                continue

            inter_area = (ix1 - ix0) * (iy1 - iy0)
            if inter_area < self.cfg.min_box_area:
                continue

            cx0 = (ix0 - float(left)) / float(size)
            cy0 = (iy0 - float(top)) / float(size)
            cx1 = (ix1 - float(left)) / float(size)
            cy1 = (iy1 - float(top)) / float(size)

            cx0 = min(max(cx0, 0.0), 1.0)
            cy0 = min(max(cy0, 0.0), 1.0)
            cx1 = min(max(cx1, 0.0), 1.0)
            cy1 = min(max(cy1, 0.0), 1.0)

            # guard against degenerate after clamp
            if cx1 <= cx0 or cy1 <= cy0:
                continue

            cat_id = int(ann["category_id"])
            if cat_id not in self.cat_id_to_contig:
                continue
            cls = float(self.cat_id_to_contig[cat_id])

            items.append((inter_area, cls, cx0, cy0, cx1, cy1))

        if self.cfg.sort_by_area:
            items.sort(key=lambda t: t[0], reverse=True)

        items = items[:max_objects]

        layout = torch.zeros((max_objects, 5), dtype=torch.float32)
        layout_mask = torch.zeros((max_objects,), dtype=torch.bool)

        for i, it in enumerate(items):
            _area, cls, x0, y0, x1, y1 = it
            layout[i, 0] = cls
            layout[i, 1] = x0
            layout[i, 2] = y0
            layout[i, 3] = x1
            layout[i, 4] = y1
            layout_mask[i] = True

        stats = {
            "num_instances_total": int(num_total_after_filter),
            "num_instances_kept": int(len(items)),
        }
        return layout, layout_mask, stats

    def __getitem__(self, idx: int):
        ann = self.caption_anns[idx]
        image_id = int(ann["image_id"])
        caption = ann.get("caption", "")

        info = self.image_id_to_info[image_id]
        file_name = info["file_name"]

        # center crop params from original size in annotations
        # (we still need orig size to compute crop, even if cached)
        orig_w = int(info.get("width", 0))
        orig_h = int(info.get("height", 0))
        if orig_w <= 0 or orig_h <= 0:
            # fallback: if width/height missing, read image once
            img_tmp = self._load_image(file_name)
            orig_w, orig_h = img_tmp.size

        top, left, size = _center_square_crop_params(orig_h, orig_w)

        do_flip = bool(self.cfg.random_flip and (np.random.rand() < 0.5))

        # samples
        if self.cfg.use_cached:
            samples = self._load_cached_moments(split=self.cfg.split, file_name=file_name, flipped=do_flip)
        else:
            img = self._load_image(file_name)
            img = img.crop((left, top, left + size, top + size))
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.resize((self.cfg.image_size, self.cfg.image_size), resample=Image.BICUBIC)
            samples = _img_to_tensor_minus1_1(img)

        crop_tls = (top, left, size)

        layout, layout_mask, stats = self._encode_layout_for_image(image_id=image_id, crop_tls=crop_tls)

        if do_flip and layout_mask.any():
            m = layout_mask
            x0 = layout[m, 1].clone()
            x1 = layout[m, 3].clone()
            layout[m, 1] = 1.0 - x1
            layout[m, 3] = 1.0 - x0

        meta = {
            "image_id": image_id,
            "file_name": file_name,
            "cache_key": file_name,
            "orig_size": (int(orig_h), int(orig_w)),
            "crop_left_top_size": (int(top), int(left), int(size)),
            "flipped": do_flip,
            "num_instances_total": stats["num_instances_total"],
            "num_instances_kept": stats["num_instances_kept"],
            "caption_ann_id": int(ann.get("id", -1)),
        }

        return samples, caption, layout, layout_mask, meta


def coco_collate_fn(batch):
    samples, caps, layouts, masks, metas = zip(*batch)
    return (
        torch.stack(samples, 0),
        list(caps),
        torch.stack(layouts, 0),
        torch.stack(masks, 0),
        list(metas),
    )


def build_coco_mini_dataset(
    root: str,
    split: str,
    image_size: int = 256,
    max_objects: int = 16,
    random_flip: bool = False,
    keep_crowd: bool = False,
    min_box_area: float = 1.0,
    use_cached: bool = False,
    cached_path: Optional[str] = None,
    cached_use_flip: bool = False,
) -> CocoMiniCaptionLayoutDataset:
    cfg = CocoMiniConfig(
        root=root,
        split=split,
        image_size=image_size,
        max_objects=max_objects,
        random_flip=random_flip,
        keep_crowd=keep_crowd,
        min_box_area=min_box_area,
        use_cached=use_cached,
        cached_path=cached_path,
        cached_use_flip=cached_use_flip,
    )
    return CocoMiniCaptionLayoutDataset(cfg)