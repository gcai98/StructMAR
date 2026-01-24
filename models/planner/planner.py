# R2/models/planner/planner.py
# A minimal text->layout set prediction planner (DETR-style queries, but text-only context)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    """
    cxcywh: [..., 4] in [0,1]
    return  xyxy : [..., 4] in [0,1] with x0<=x1,y0<=y1
    """
    cx, cy, w, h = cxcywh.unbind(dim=-1)
    x0 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y0 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x1 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy + 0.5 * h).clamp(0.0, 1.0)

    # ensure ordering after clamp
    x0_, x1_ = torch.minimum(x0, x1), torch.maximum(x0, x1)
    y0_, y1_ = torch.minimum(y0, y1), torch.maximum(y0, y1)
    return torch.stack([x0_, y0_, x1_, y1_], dim=-1)


@dataclass
class PlannerOutput:
    pred_logits: torch.Tensor  # [B, Q, C+1]
    pred_boxes: torch.Tensor   # [B, Q, 4]  (x0,y0,x1,y1) in [0,1]


class LayoutPlanner(nn.Module):
    """
    Text -> (K queries) -> set of (class, box)

    - Input: text_emb [B, cond_dim] (e.g., CLIP pooled embedding 512)
    - Output:
        pred_logits [B, Q, num_classes+1]  (last idx = "no-object")
        pred_boxes  [B, Q, 4] in xyxy normalized [0,1]
    """

    def __init__(
        self,
        cond_dim: int = 512,
        hidden_dim: int = 512,
        num_queries: int = 16,
        num_classes: int = 80,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)

        if self.hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim({self.hidden_dim}) must be divisible by num_heads({num_heads})")

        self.text_proj = nn.Linear(self.cond_dim, self.hidden_dim)

        # learned queries
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.hidden_dim) * 0.02)

        # a small transformer over queries (self-attn only), text injected as bias
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # heads
        self.class_head = nn.Linear(self.hidden_dim, self.num_classes + 1)  # +1 for no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 4),  # predict cx,cy,w,h (sigmoid)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.constant_(self.text_proj.bias, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, text_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        text_emb: [B, cond_dim]
        return dict with keys: pred_logits, pred_boxes
        """
        if text_emb.dim() != 2:
            raise ValueError(f"text_emb must be [B, D], got {tuple(text_emb.shape)}")

        B = text_emb.size(0)
        device = text_emb.device

        ctx = self.text_proj(text_emb)  # [B, H]
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=ctx.dtype)  # [B,Q,H]
        q = q + ctx.unsqueeze(1)  # inject text context

        h = self.query_encoder(q)  # [B,Q,H]

        pred_logits = self.class_head(h)  # [B,Q,C+1]

        # predict cxcywh in [0,1]
        cxcywh = torch.sigmoid(self.bbox_head(h))
        pred_boxes = _cxcywh_to_xyxy(cxcywh)  # [B,Q,4]

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    @torch.no_grad()
    def predict_layout(self, text_emb: torch.Tensor, score_thresh: float = 0.0) -> torch.Tensor:
        """
        Convenience: produce layout tensor [B, Q, 5] => (cls, x0,y0,x1,y1)
        where cls is argmax (excluding no-object).
        """
        out = self.forward(text_emb)
        logits = out["pred_logits"]
        boxes = out["pred_boxes"]
        probs = logits.softmax(dim=-1)
        cls = probs[..., :-1].argmax(dim=-1)  # exclude no-object
        scores = probs[..., :-1].max(dim=-1).values

        layout = torch.zeros(text_emb.size(0), self.num_queries, 5, device=text_emb.device, dtype=boxes.dtype)
        layout[..., 0] = cls.to(layout.dtype)
        layout[..., 1:5] = boxes

        if score_thresh > 0:
            keep = scores >= score_thresh
            # set low-score slots to "null-like": cls=0, bbox=0
            layout = layout * keep.unsqueeze(-1).to(layout.dtype)

        return layout
