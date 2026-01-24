# R2/engine_planner.py
# Planner training loop + Hungarian matching criterion

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hungarian (preferred)
_HUNGARIAN_OK = True
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:
    _HUNGARIAN_OK = False
    linear_sum_assignment = None


def _safe_text_encode(text_encoder, captions: List[str], device: torch.device) -> torch.Tensor:
    """
    Robustly obtain text embedding [B, D] from your HFCLIPTextEncoder wrapper.
    This matches your debug where text_emb.shape == [B,512].
    """
    # common patterns
    if hasattr(text_encoder, "encode_text"):
        emb = text_encoder.encode_text(captions)
    elif hasattr(text_encoder, "encode"):
        emb = text_encoder.encode(captions)
    elif callable(text_encoder):
        emb = text_encoder(captions)
    elif hasattr(text_encoder, "get_text_features"):
        emb = text_encoder.get_text_features(captions)
    else:
        raise AttributeError("Cannot find a usable text-encoding method on text_encoder.")

    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)

    return emb.to(device=device)


def _build_targets_from_layout(
    layout: torch.Tensor, layout_mask: torch.Tensor
) -> List[Dict[str, torch.Tensor]]:
    """
    layout: [B, K, 5] => (cls, x0,y0,x1,y1) float
    layout_mask: [B, K] bool
    returns list of targets with variable length:
      targets[i]["labels"]: [Mi] long
      targets[i]["boxes"] : [Mi,4] float
    """
    B = layout.size(0)
    targets: List[Dict[str, torch.Tensor]] = []
    for i in range(B):
        m = layout_mask[i].bool()
        if m.sum().item() == 0:
            targets.append({"labels": layout.new_zeros((0,), dtype=torch.long),
                            "boxes": layout.new_zeros((0, 4), dtype=torch.float32)})
            continue

        li = layout[i, m]  # [Mi,5]
        labels = li[:, 0].to(dtype=torch.long)
        boxes = li[:, 1:5].to(dtype=torch.float32)
        # safety clamp
        boxes = boxes.clamp(0.0, 1.0)
        targets.append({"labels": labels, "boxes": boxes})
    return targets


def _hungarian_match_one(
    pred_logits: torch.Tensor,  # [Q, C+1]
    pred_boxes: torch.Tensor,   # [Q, 4]
    tgt_labels: torch.Tensor,   # [M]
    tgt_boxes: torch.Tensor,    # [M,4]
    cost_class: float,
    cost_bbox: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (idx_pred, idx_tgt) for one sample.
    If M==0, return empty indices.
    """
    Q = pred_logits.size(0)
    M = tgt_labels.size(0)
    if M == 0:
        return pred_logits.new_zeros((0,), dtype=torch.long), pred_logits.new_zeros((0,), dtype=torch.long)

    # probabilities excluding no-object are still fine for cost
    out_prob = pred_logits.softmax(dim=-1)  # [Q, C+1]
    # classification cost: -P(class)
    # gather: [Q, M]
    cost_cls = -out_prob[:, tgt_labels]  # tgt_labels are indices into C+1; ok as long as labels in [0,C-1]
    # bbox L1 cost
    cost_l1 = torch.cdist(pred_boxes, tgt_boxes, p=1)  # [Q, M]
    C = cost_class * cost_cls + cost_bbox * cost_l1

    if _HUNGARIAN_OK:
        row_ind, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
        idx_pred = torch.as_tensor(row_ind, dtype=torch.long, device=pred_logits.device)
        idx_tgt = torch.as_tensor(col_ind, dtype=torch.long, device=pred_logits.device)
        return idx_pred, idx_tgt

    # fallback: greedy match (not optimal, but avoids crash)
    idx_pred = []
    idx_tgt = []
    C_work = C.detach().clone()
    used_q = torch.zeros((Q,), dtype=torch.bool, device=C_work.device)
    used_m = torch.zeros((M,), dtype=torch.bool, device=C_work.device)
    for _ in range(min(Q, M)):
        C_masked = C_work.clone()
        C_masked[used_q] = 1e9
        C_masked[:, used_m] = 1e9
        flat = torch.argmin(C_masked)
        q = (flat // M).item()
        m = (flat % M).item()
        if used_q[q] or used_m[m]:
            break
        used_q[q] = True
        used_m[m] = True
        idx_pred.append(q)
        idx_tgt.append(m)
    if len(idx_pred) == 0:
        return pred_logits.new_zeros((0,), dtype=torch.long), pred_logits.new_zeros((0,), dtype=torch.long)
    return (torch.tensor(idx_pred, dtype=torch.long, device=pred_logits.device),
            torch.tensor(idx_tgt, dtype=torch.long, device=pred_logits.device))


@dataclass
class CriterionConfig:
    num_classes: int = 80
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    bbox_loss_coef: float = 5.0
    no_object_coef: float = 0.1


class LayoutSetCriterion(nn.Module):
    """
    DETR-like criterion for set prediction:
      - Hungarian match predictions to GT objects
      - CE loss over all queries (unmatched -> no-object)
      - L1 bbox loss over matched pairs only
    """

    def __init__(self, cfg: CriterionConfig):
        super().__init__()
        self.cfg = cfg

        # weight for no-object class in CE
        empty_weight = torch.ones(cfg.num_classes + 1)
        empty_weight[cfg.num_classes] = cfg.no_object_coef
        self.register_buffer("empty_weight", empty_weight)

    @torch.no_grad()
    def match(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        """
        returns list of (idx_pred, idx_tgt) for each batch element
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        B = pred_logits.size(0)
        indices = []
        for i in range(B):
            idx_pred, idx_tgt = _hungarian_match_one(
                pred_logits[i], pred_boxes[i],
                targets[i]["labels"], targets[i]["boxes"],
                cost_class=self.cfg.cost_class,
                cost_bbox=self.cfg.cost_bbox,
            )
            indices.append((idx_pred, idx_tgt))
        return indices

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pred_logits = outputs["pred_logits"]  # [B,Q,C+1]
        pred_boxes = outputs["pred_boxes"]    # [B,Q,4]
        B, Q, _ = pred_logits.shape
        device = pred_logits.device

        indices = self.match(outputs, targets)

        # build target classes for all queries: default to no-object
        target_classes = torch.full((B, Q), fill_value=self.cfg.num_classes, dtype=torch.long, device=device)
        num_boxes = 0

        for i, (idx_pred, idx_tgt) in enumerate(indices):
            if idx_pred.numel() == 0:
                continue
            tgt_labels = targets[i]["labels"][idx_tgt]
            target_classes[i, idx_pred] = tgt_labels
            num_boxes += idx_pred.numel()

        # CE over all queries
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),  # [B, C+1, Q]
            target_classes,
            weight=self.empty_weight.to(device=device),
        )

        # bbox loss over matched only
        if num_boxes == 0:
            loss_bbox = pred_boxes.sum() * 0.0
        else:
            src = []
            tgt = []
            for i, (idx_pred, idx_tgt) in enumerate(indices):
                if idx_pred.numel() == 0:
                    continue
                src.append(pred_boxes[i, idx_pred])
                tgt.append(targets[i]["boxes"][idx_tgt])
            src_boxes = torch.cat(src, dim=0)
            tgt_boxes = torch.cat(tgt, dim=0)
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / max(1, num_boxes)

        loss = loss_ce + self.cfg.bbox_loss_coef * loss_bbox
        return {
            "loss": loss,
            "loss_ce": loss_ce.detach(),
            "loss_bbox": loss_bbox.detach(),
            "num_matched": torch.tensor(float(num_boxes), device=device),
        }


def train_one_epoch_planner(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    text_encoder,
    loss_scaler=None,
    max_norm: float = 0.0,
    log_writer=None,
    print_freq: int = 20,
):
    model.train()
    criterion.train()

    it = 0
    for batch in data_loader:
        # expected: imgs, captions, layout, layout_mask, meta
        imgs, captions, layout, layout_mask, metas = batch

        # planner does not need imgs, but we keep batch format consistent
        text_emb = _safe_text_encode(text_encoder, captions, device=device)
        layout = layout.to(device=device)
        layout_mask = layout_mask.to(device=device)

        targets = _build_targets_from_layout(layout, layout_mask)

        optimizer.zero_grad(set_to_none=True)

        if loss_scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(text_emb)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss is not finite: {loss.item()}")

            # NativeScalerWithGradNormCount-style API (same as your MAR training)
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=False,
                update_grad=True,
            )
        else:
            outputs = model(text_emb)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss is not finite: {loss.item()}")

            loss.backward()
            if max_norm and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if (it % print_freq) == 0:
            msg = (
                f"[Planner][Epoch {epoch}] iter {it} "
                f"loss={loss.item():.4f} "
                f"ce={float(loss_dict['loss_ce']):.4f} "
                f"bbox={float(loss_dict['loss_bbox']):.4f} "
                f"matched={float(loss_dict['num_matched']):.1f}"
            )
            print(msg)

        if log_writer is not None:
            step = epoch * 1000000 + it
            log_writer.add_scalar("planner/loss", loss.item(), step)
            log_writer.add_scalar("planner/loss_ce", float(loss_dict["loss_ce"]), step)
            log_writer.add_scalar("planner/loss_bbox", float(loss_dict["loss_bbox"]), step)
            log_writer.add_scalar("planner/num_matched", float(loss_dict["num_matched"]), step)

        it += 1
