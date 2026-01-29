import os
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


def calculate_iou_xyxy(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    a2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = a1 + a2 - inter
    return inter / (union + 1e-6)


def _center_xy(box_xyxy):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


# COCO 80 category ids sorted by id (standard)
COCO80_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]


class LayoutScorer(nn.Module):
    """
    Dense Layout Reward with Detector (offline):
      reward = w_iou * mean_iou + w_conf * mean_conf + w_center * mean_center
               - w_miss * miss_ratio - w_extra * extra_ratio

    Detector backend:
      - .pth -> torchvision Faster R-CNN
      - .pt  -> ultralytics YOLOv8
    """
    def __init__(
        self,
        detector_path=None,
        device="cuda",
        conf_thres=0.5,

        # reward weights
        w_iou=1.0,
        w_conf=0.2,
        w_center=0.2,
        w_miss=0.1,
        w_extra=0.05,

        # behavior
        class_aware=False,
        center_norm="diag",  # "diag" or "max"
        iou_missing_penalty=0.0,

        # speed
        use_amp=True,

        # YOLO options (only used when detector_path is .pt)
        yolo_iou_thres=0.7,
        yolo_half=True,
    ):
        super().__init__()
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.conf_thres = float(conf_thres)

        self.w_iou = float(w_iou)
        self.w_conf = float(w_conf)
        self.w_center = float(w_center)
        self.w_miss = float(w_miss)
        self.w_extra = float(w_extra)

        self.class_aware = bool(class_aware)
        self.center_norm = str(center_norm)
        self.iou_missing_penalty = float(iou_missing_penalty)

        self.use_amp = bool(use_amp)

        self.backend = "torchvision"  # or "yolo"
        self.yolo_iou_thres = float(yolo_iou_thres)
        self.yolo_half = bool(yolo_half)

        det_path = detector_path or ""
        det_lower = det_path.lower()

        if det_lower.endswith(".pt"):
            self.backend = "yolo"
            print("[LayoutScorer] Initializing YOLOv8 backend...")
            try:
                from ultralytics import YOLO
            except Exception as e:
                raise ImportError(
                    "ultralytics is required for YOLO backend."
                ) from e

            if not (det_path and os.path.isfile(det_path)):
                raise FileNotFoundError(
                    f"[LayoutScorer] YOLO detector_path not found: {det_path}"
                )

            self.model = YOLO(det_path)
            print(f"[LayoutScorer] Loaded YOLO weights: {det_path}")

        else:
            self.backend = "torchvision"
            print("[LayoutScorer] Initializing Faster R-CNN (ResNet50)...")
            self.model = fasterrcnn_resnet50_fpn_v2(weights=None, box_score_thresh=self.conf_thres)

            if detector_path and os.path.isfile(detector_path):
                print(f"[LayoutScorer] Loading weights from local file: {detector_path}")
                sd = torch.load(detector_path, map_location="cpu")
                if isinstance(sd, dict):
                    for k in ["model", "state_dict", "net"]:
                        if k in sd and isinstance(sd[k], dict):
                            sd = sd[k]
                            break
                try:
                    self.model.load_state_dict(sd, strict=True)
                except Exception as e:
                    print(f"[LayoutScorer] strict=True load failed, trying strict=False. Error: {e}")
                    self.model.load_state_dict(sd, strict=False)
            else:
                print(f"[LayoutScorer] WARNING: detector_path missing/not found: {detector_path}")
                print("[LayoutScorer] Model will run with RANDOM weights!")

            self.model.to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    def _autocast_ctx(self):
        if self.device.type == "cuda":
            return torch.amp.autocast("cuda", enabled=self.use_amp)
        return nullcontext()

    def _to_01(self, images: torch.Tensor) -> torch.Tensor:
        """Accept [0,1] or [-1,1]. Auto-detect by range."""
        if images.numel() == 0:
            return images
        mn = float(images.min().item())
        if mn < -0.1:
            x = (images + 1.0) * 0.5
        else:
            x = images
        return x.clamp(0, 1)

    def _contig_to_coco_catid(self, cls_contig: int):
        if cls_contig < 0 or cls_contig >= len(COCO80_CAT_IDS):
            return None
        return int(COCO80_CAT_IDS[cls_contig])

    def _infer_torchvision(self, imgs01: torch.Tensor):
        imgs01 = imgs01.to(self.device, non_blocking=True)
        img_list = list(imgs01)

        with torch.inference_mode():
            with self._autocast_ctx():
                preds = self.model(img_list)
        return preds

    def _infer_yolo(self, imgs01: torch.Tensor):
        """
        imgs01: [B,3,H,W] float in [0,1]
        Return: list of dict with keys boxes/scores/labels (numpy)
        """
        B, _, H, W = imgs01.shape
        imgs_np = (imgs01.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)

        dev = 0
        if self.device.type == "cuda":
            dev = int(self.device.index) if self.device.index is not None else 0
        else:
            dev = "cpu"

        results = self.model.predict(
            imgs_np,
            verbose=False,
            conf=self.conf_thres,
            iou=self.yolo_iou_thres,
            device=dev,
            imgsz=max(H, W),
            half=(self.yolo_half and self.device.type == "cuda"),
        )

        outs = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                outs.append({"boxes": np.zeros((0, 4), np.float32),
                             "scores": np.zeros((0,), np.float32),
                             "labels": np.zeros((0,), np.int64)})
                continue
            b = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            s = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
            c = r.boxes.cls.detach().cpu().numpy().astype(np.int64)  # YOLO: 0..79
            outs.append({"boxes": b, "scores": s, "labels": c})
        return outs

    @torch.no_grad()
    def score(self, images, target_layout, target_mask, captions=None):
        return self.forward(images, target_layout, target_mask, captions=captions)

    @torch.no_grad()
    def forward(self, images, target_layout, target_mask, captions=None):
        """
        return dict:
          {"reward": [B], "iou":[B], "conf":[B], "centroid":[B], "miss":[B], "extra":[B]}
        """
        imgs01 = self._to_01(images)
        B, _, H, W = imgs01.shape

        diag = float(np.sqrt(H * H + W * W) + 1e-6)
        norm_denom = diag if self.center_norm == "diag" else float(max(H, W) + 1e-6)

        if self.backend == "yolo":
            preds = self._infer_yolo(imgs01)
        else:
            self.model.eval()
            preds = self._infer_torchvision(imgs01)

        rewards = np.zeros((B,), dtype=np.float32)
        iou_m   = np.zeros((B,), dtype=np.float32)
        conf_m  = np.zeros((B,), dtype=np.float32)
        cen_m   = np.zeros((B,), dtype=np.float32)
        miss_r  = np.zeros((B,), dtype=np.float32)
        extra_r = np.zeros((B,), dtype=np.float32)

        for i in range(B):
            pred = preds[i]

            if self.backend == "yolo":
                pboxes = pred["boxes"]
                pscores = pred["scores"]
                plabels = pred["labels"]
            else:
                pboxes = pred.get("boxes", torch.empty(0, 4)).detach().cpu().numpy()
                pscores = pred.get("scores", torch.empty(0)).detach().cpu().numpy()
                plabels = pred.get("labels", torch.empty(0, dtype=torch.long)).detach().cpu().numpy()

            keep = pscores >= self.conf_thres
            pboxes = pboxes[keep]
            pscores = pscores[keep]
            plabels = plabels[keep]

            layout = target_layout[i]
            mask = target_mask[i].bool()
            tgts = layout[mask]  # [M,5]
            M = int(tgts.shape[0])

            if M == 0:
                rewards[i] = 0.0
                continue

            t_abs, t_cls, t_area = [], [], []
            for t in tgts:
                cls_contig = int(float(t[0].item()))
                xy = t[1:].detach().cpu().numpy().astype(np.float32)
                xy_abs = xy * np.array([W, H, W, H], dtype=np.float32)
                xy_abs[0] = np.clip(xy_abs[0], 0, W - 1)
                xy_abs[2] = np.clip(xy_abs[2], 0, W - 1)
                xy_abs[1] = np.clip(xy_abs[1], 0, H - 1)
                xy_abs[3] = np.clip(xy_abs[3], 0, H - 1)
                if xy_abs[2] <= xy_abs[0] or xy_abs[3] <= xy_abs[1]:
                    continue

                if self.class_aware:
                    tgt_label = cls_contig if (self.backend == "yolo") else self._contig_to_coco_catid(cls_contig)
                else:
                    tgt_label = None

                area = float((xy_abs[2] - xy_abs[0]) * (xy_abs[3] - xy_abs[1]))
                t_abs.append(xy_abs)
                t_cls.append(tgt_label)
                t_area.append(area)

            if len(t_abs) == 0:
                rewards[i] = 0.0
                continue

            order = np.argsort(-np.array(t_area, dtype=np.float32))
            t_abs = [t_abs[k] for k in order]
            t_cls = [t_cls[k] for k in order]
            M = len(t_abs)

            used = np.zeros((pboxes.shape[0],), dtype=np.bool_)
            sum_iou, sum_conf, sum_center = 0.0, 0.0, 0.0
            miss = 0

            for j in range(M):
                tgt_box = t_abs[j]
                tgt_cls = t_cls[j]

                best_idx = -1
                best_iou = -1.0

                for k in range(pboxes.shape[0]):
                    if used[k]:
                        continue
                    if self.class_aware and (tgt_cls is not None):
                        if int(plabels[k]) != int(tgt_cls):
                            continue
                    iou = calculate_iou_xyxy(tgt_box, pboxes[k])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = k

                if best_idx < 0:
                    miss += 1
                    if self.iou_missing_penalty > 0:
                        sum_iou -= self.iou_missing_penalty
                    continue

                used[best_idx] = True
                sum_iou += float(max(0.0, best_iou))
                sum_conf += float(pscores[best_idx])

                c_t = _center_xy(tgt_box)
                c_p = _center_xy(pboxes[best_idx])
                d = float(np.linalg.norm(c_t - c_p))
                c_reward = float(np.clip(1.0 - d / norm_denom, 0.0, 1.0))
                sum_center += c_reward

            mean_iou = sum_iou / max(1, M)
            mean_conf = sum_conf / max(1, M)
            mean_center = sum_center / max(1, M)

            miss_ratio = float(miss / max(1, M))
            extra = int((~used).sum())
            extra_ratio = float(extra / max(1, M))

            r = (
                self.w_iou * mean_iou +
                self.w_conf * mean_conf +
                self.w_center * mean_center -
                self.w_miss * miss_ratio -
                self.w_extra * extra_ratio
            )

            rewards[i] = float(r)
            iou_m[i] = float(mean_iou)
            conf_m[i] = float(mean_conf)
            cen_m[i] = float(mean_center)
            miss_r[i] = float(miss_ratio)
            extra_r[i] = float(extra_ratio)

        out = {
            "reward": torch.from_numpy(rewards).to(self.device, dtype=torch.float32),
            "iou": torch.from_numpy(iou_m).to(self.device, dtype=torch.float32),
            "conf": torch.from_numpy(conf_m).to(self.device, dtype=torch.float32),

            "centroid": torch.from_numpy(cen_m).to(self.device, dtype=torch.float32),
            "center": torch.from_numpy(cen_m).to(self.device, dtype=torch.float32),

            "miss": torch.from_numpy(miss_r).to(self.device, dtype=torch.float32),
            "extra": torch.from_numpy(extra_r).to(self.device, dtype=torch.float32),
        }
        return out