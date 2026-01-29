from functools import partial
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.diffloss import DiffLoss


# ==========================================================
# Utils
# ==========================================================

def mask_by_order(mask_len, order, bsz, seq_len):
    """
    mask_len: scalar tensor
    order: [B, seq_len]
    return: [B, seq_len] bool mask
    """
    device = order.device
    masking = torch.zeros(bsz, seq_len, device=device)
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, :mask_len.long()],
        src=torch.ones(bsz, seq_len, device=device),
    ).bool()
    return masking


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ==========================================================
# 2D RoPE
# ==========================================================

class RotaryEmbedding2D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
        self.head_dim = int(head_dim)
        self.half_dim = self.head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(self, pos: torch.Tensor, dtype: torch.dtype):
        # pos: [B, L]
        pos = pos.to(dtype=torch.float32)
        freqs = torch.einsum("bl,d->bld", pos, self.inv_freq)  # [B, L, half_dim/2]
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        out = torch.stack([out1, out2], dim=-1).flatten(-2)
        return out

    def apply_2d(self, q: torch.Tensor, k: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor):
        dtype = q.dtype
        cos_x, sin_x = self._get_cos_sin(pos_x, dtype=dtype)
        cos_y, sin_y = self._get_cos_sin(pos_y, dtype=dtype)

        qx, qy = q[..., : self.half_dim], q[..., self.half_dim :]
        kx, ky = k[..., : self.half_dim], k[..., self.half_dim :]

        qx = self._apply_rotary(qx, cos_x, sin_x)
        kx = self._apply_rotary(kx, cos_x, sin_x)
        qy = self._apply_rotary(qy, cos_y, sin_y)
        ky = self._apply_rotary(ky, cos_y, sin_y)

        q = torch.cat([qx, qy], dim=-1)
        k = torch.cat([kx, ky], dim=-1)
        return q, k


# ==========================================================
# Attention blocks
# ==========================================================

class AttentionRoPE2D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_2d_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        head_dim = self.dim // self.num_heads
        assert head_dim * self.num_heads == self.dim, "dim must be divisible by num_heads"
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_2d_rope = bool(use_2d_rope)
        self.rope = RotaryEmbedding2D(head_dim=head_dim, base=rope_base) if self.use_2d_rope else None

    def forward(
        self,
        x: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        attn_bias: torch.Tensor,
    ):
        """
        x: [B, L, C]
        attn_bias: [B, 1, L, L]
        """
        B, L, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_2d_rope:
            q, k = self.rope.apply_2d(q, k, pos_x=pos_x, pos_y=pos_y)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ctx_dim: Optional[int] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        head_dim = self.dim // self.num_heads
        assert head_dim * self.num_heads == self.dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        ctx_dim = int(ctx_dim) if ctx_dim is not None else self.dim

        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(ctx_dim, self.dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        """
        x: [B, L, D]
        ctx: [B, T, C]
        ctx_mask: [B, T]
        """
        B, L, D = x.shape
        if ctx is None or ctx.shape[1] == 0:
            return torch.zeros_like(x)

        T = ctx.shape[1]

        q = self.q(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(ctx).reshape(B, T, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if ctx_mask is not None:
            m = ctx_mask.to(device=attn.device).bool().view(B, 1, 1, T)
            if attn.dtype in (torch.float16, torch.bfloat16):
                neg = -1e4
            else:
                neg = -1e9
            attn = attn.masked_fill(~m, neg)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MARBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_prob: float = 0.0,
        use_2d_rope: bool = False,
        rope_base: float = 10000.0,
        use_text_cross_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionRoPE2D(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_2d_rope=use_2d_rope,
            rope_base=rope_base,
        )
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

        self.use_text_cross_attn = bool(use_text_cross_attn)
        if self.use_text_cross_attn:
            self.norm_text = norm_layer(dim)
            self.text_attn = CrossAttention(
                dim=dim,
                num_heads=num_heads,
                ctx_dim=dim,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            self.text_gate = nn.Parameter(torch.zeros(1))
        else:
            self.norm_text = None
            self.text_attn = None
            self.text_gate = None

        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden, drop=proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        attn_bias: torch.Tensor,
        text_ctx: torch.Tensor,
        text_ctx_mask: Optional[torch.Tensor],
    ):
        x = x + self.drop_path(self.attn(self.norm1(x), pos_x=pos_x, pos_y=pos_y, attn_bias=attn_bias))
        if self.use_text_cross_attn:
            gate = torch.tanh(self.text_gate)
            x = x + self.drop_path(gate * self.text_attn(self.norm_text(x), text_ctx, ctx_mask=text_ctx_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ==========================================================
# MAR
# ==========================================================

class MAR(nn.Module):
    def __init__(
        self,
        img_size=256,
        vae_stride=16,
        patch_size=1,
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.1,
        proj_dropout=0.1,
        buffer_size=64,
        diffloss_d=3,
        diffloss_w=1024,
        num_sampling_steps="100",
        diffusion_batch_mul=4,
        grad_checkpointing=False,
        cond_dim=None,
        layout_class_num=80,
        layout_bbox_hidden=None,

        use_2d_rope: bool = False,
        rope_base: float = 10000.0,
        disable_learned_pos_emb_when_rope: bool = True,

        use_text_cross_attn: bool = False,
        text_cross_every_n_layers: int = 1,

        use_layout_bias: bool = False,
        layout_bias_value: float = 10000.0,
        layout_bias_on_encoder: bool = True,
        layout_bias_on_decoder: bool = True,
    ):
        super().__init__()

        self.vae_embed_dim = vae_embed_dim
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = float(label_drop_prob)
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = int(buffer_size)

        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim)
        )

        self.use_2d_rope = bool(use_2d_rope)
        self.rope_base = float(rope_base)
        self.disable_learned_pos_emb_when_rope = bool(disable_learned_pos_emb_when_rope)

        self.use_text_cross_attn = bool(use_text_cross_attn)
        self.text_cross_every_n_layers = max(1, int(text_cross_every_n_layers))

        self.use_layout_bias = bool(use_layout_bias)
        self.layout_bias_value = float(layout_bias_value)
        self.layout_bias_on_encoder = bool(layout_bias_on_encoder)
        self.layout_bias_on_decoder = bool(layout_bias_on_decoder)

        self.encoder_blocks = nn.ModuleList(
            [
                MARBlock(
                    dim=encoder_embed_dim,
                    num_heads=encoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    attn_drop=attn_dropout,
                    proj_drop=proj_dropout,
                    drop_path_prob=0.0,
                    use_2d_rope=self.use_2d_rope,
                    rope_base=self.rope_base,
                    use_text_cross_attn=(self.use_text_cross_attn and (i % self.text_cross_every_n_layers == 0)),
                )
                for i in range(encoder_depth)
            ]
        )
        self.encoder_norm = norm_layer(encoder_embed_dim)

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim)
        )

        self.decoder_blocks = nn.ModuleList(
            [
                MARBlock(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    attn_drop=attn_dropout,
                    proj_drop=proj_dropout,
                    drop_path_prob=0.0,
                    use_2d_rope=self.use_2d_rope,
                    rope_base=self.rope_base,
                    use_text_cross_attn=(self.use_text_cross_attn and (i % self.text_cross_every_n_layers == 0)),
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )

        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
        )
        self.diffusion_batch_mul = int(diffusion_batch_mul)

        self.text_proj = nn.Linear(cond_dim, encoder_embed_dim) if cond_dim is not None else None

        if self.use_text_cross_attn:
            if cond_dim is None:
                raise ValueError("cond_dim must be set if use_text_cross_attn=True")
            self.text_ctx_proj_enc = nn.Linear(cond_dim, encoder_embed_dim)
            self.text_ctx_proj_dec = nn.Linear(cond_dim, decoder_embed_dim)
            self.text_null_enc = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
            self.text_null_dec = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        else:
            self.text_ctx_proj_enc = None
            self.text_ctx_proj_dec = None
            self.text_null_enc = None
            self.text_null_dec = None

        self.layout_class_num = int(layout_class_num)
        self.layout_cls_emb = nn.Embedding(self.layout_class_num, encoder_embed_dim)

        if layout_bbox_hidden is None:
            self.layout_bbox_mlp = nn.Linear(4, encoder_embed_dim)
        else:
            h = int(layout_bbox_hidden)
            self.layout_bbox_mlp = nn.Sequential(
                nn.Linear(4, h),
                nn.GELU(),
                nn.Linear(h, encoder_embed_dim),
            )

        self.layout_null = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))

        total_len = self.buffer_size + self.seq_len
        pos_x = torch.zeros(total_len, dtype=torch.long)
        pos_y = torch.zeros(total_len, dtype=torch.long)
        img_ids = torch.arange(self.seq_len, dtype=torch.long)
        pos_x[self.buffer_size:] = img_ids % self.seq_w
        pos_y[self.buffer_size:] = img_ids // self.seq_w
        self.register_buffer("rope_pos_x_full", pos_x, persistent=False)
        self.register_buffer("rope_pos_y_full", pos_y, persistent=False)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.class_emb.weight, std=0.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=0.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)

        torch.nn.init.normal_(self.layout_cls_emb.weight, std=0.02)
        torch.nn.init.normal_(self.layout_null, std=0.02)

        if self.use_text_cross_attn:
            torch.nn.init.normal_(self.text_null_enc, std=0.02)
            torch.nn.init.normal_(self.text_null_dec, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # ----------------------------------------------------------
    # Patchify helpers
    # ----------------------------------------------------------
    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p**2)
        return x  # [B, L, D]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [B, C, H, W]

    # ----------------------------------------------------------
    # Masking helpers
    # ----------------------------------------------------------
    def sample_orders(self, bsz, device):
        return torch.stack([torch.randperm(self.seq_len, device=device) for _ in range(bsz)], dim=0)

    def random_masking(self, x, orders):
        bsz, seq_len, _ = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(
            mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        return mask

    # ----------------------------------------------------------
    # Conditioning
    # ----------------------------------------------------------
    def _make_base_token(self, bsz, device, labels=None, text_emb=None, text_mask: Optional[torch.Tensor] = None):
        """
        text_emb: [B, D] or [B, T, D]
        """
        if (labels is not None) and (text_emb is not None):
            raise ValueError("labels and text_emb are mutually exclusive.")

        if text_emb is not None:
            if self.text_proj is None:
                raise AssertionError("cond_dim must be set when using text_emb")

            if text_emb.dim() == 3:
                tok = text_emb
                if text_mask is not None:
                    m = text_mask.to(device=tok.device).to(dtype=tok.dtype)
                else:
                    m = torch.ones(tok.shape[0], tok.shape[1], device=tok.device, dtype=tok.dtype)
                denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
                base_in = (tok * m.unsqueeze(-1)).sum(dim=1) / denom
            elif text_emb.dim() == 2:
                base_in = text_emb
            else:
                raise ValueError(f"text_emb must be [B,D] or [B,T,D], got {tuple(text_emb.shape)}")

            base = self.text_proj(base_in.to(device).detach().clone())

        elif labels is not None:
            base = self.class_emb(labels.to(device))
        else:
            base = self.fake_latent.expand(bsz, -1).to(device)

        return base

    def build_cond_tokens(self, bsz, device, labels=None, text_emb=None, text_mask=None, layout=None, layout_mask=None):
        base = self._make_base_token(
            bsz=bsz, device=device, labels=labels, text_emb=text_emb, text_mask=text_mask
        )
        D = base.size(-1)
        cond_tokens = base.unsqueeze(1).repeat(1, self.buffer_size, 1)

        if layout is None:
            return cond_tokens

        # Check/Fix layout shape
        if layout.shape[0] != bsz:
            if layout.ndim == 3 and layout.shape[1] == bsz:
                layout = layout.permute(1, 0, 2)
            elif layout.shape[0] > bsz:
                layout = layout[:bsz]

        # Check/Fix mask shape
        if layout_mask is not None:
            if layout_mask.shape[0] != bsz:
                if layout_mask.ndim == 2 and layout_mask.shape[1] == bsz:
                    layout_mask = layout_mask.permute(1, 0)
                elif layout_mask.shape[0] > bsz:
                    layout_mask = layout_mask[:bsz]

        layout = layout.to(device=device)
        N = min(layout.size(1), self.buffer_size)

        cls = layout[:, :N, 0].long()
        cls = torch.clamp(cls, 0, self.layout_class_num - 1)
        bbox = layout[:, :N, 1:5].float().clamp(0.0, 1.0)

        cls_tok = self.layout_cls_emb(cls)
        bbox_tok = self.layout_bbox_mlp(bbox)
        obj_tok = cls_tok + bbox_tok

        if layout_mask is not None:
            m = layout_mask[:, :N].to(device=device).float().unsqueeze(-1)
            if obj_tok.shape[0] != bsz:
                raise RuntimeError(f"layout batch mismatch: {obj_tok.shape[0]} vs {bsz}.")
            if m.shape[0] != bsz:
                raise RuntimeError(f"layout_mask batch mismatch: {m.shape[0]} vs {bsz}.")

            null_tok = self.layout_null.to(device=device, dtype=obj_tok.dtype).expand(bsz, N, D)
            obj_tok = obj_tok * m + null_tok * (1.0 - m)

        if cond_tokens.shape[0] != obj_tok.shape[0]:
             cond_tokens = base.unsqueeze(1).repeat(1, self.buffer_size, 1)

        obj_tok = obj_tok + base.unsqueeze(1)
        cond_tokens[:, :N, :] = obj_tok.to(dtype=cond_tokens.dtype)
        return cond_tokens

    def _build_text_ctx(
        self,
        text_emb,
        text_mask: Optional[torch.Tensor],
        bsz,
        device,
        dtype,
        proj: nn.Module,
        null_token: nn.Parameter,
        drop_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return: ctx: [B, T, dim], ctx_mask: [B, T]
        """
        dim = proj.out_features
        if text_emb is None:
            ctx = torch.zeros(bsz, 0, dim, device=device, dtype=dtype)
            ctx_mask_out = torch.zeros(bsz, 0, device=device, dtype=torch.bool)
            return ctx, ctx_mask_out

        if text_emb.dim() == 2:
            text_tok = text_emb.unsqueeze(1)
            ctx_mask_out = torch.ones(bsz, 1, device=device, dtype=torch.bool)
        elif text_emb.dim() == 3:
            text_tok = text_emb
            T = text_tok.shape[1]
            if text_mask is not None:
                if text_mask.shape[0] != bsz or text_mask.shape[1] != T:
                    raise ValueError(f"text_mask shape {tuple(text_mask.shape)} must match text_emb")
                ctx_mask_out = text_mask.to(device=device).bool()
            else:
                ctx_mask_out = torch.ones(bsz, T, device=device, dtype=torch.bool)
        else:
            raise ValueError(f"text_emb must be [B,D] or [B,T,D], got {tuple(text_emb.shape)}")
        ctx = proj(text_tok.to(device=device).detach().clone())
        ctx = ctx.to(dtype=dtype)

        if drop_mask is not None:
            T = ctx.shape[1]
            null = null_token.to(device=device, dtype=dtype).expand(bsz, T, dim)
            ctx = drop_mask * null + (1.0 - drop_mask) * ctx
        return ctx, ctx_mask_out

    # ----------------------------------------------------------
    # Layout-guided attention bias
    # ----------------------------------------------------------
    def _build_layout_attn_bias(
        self,
        layout,
        layout_mask,
        q_pos_x,
        q_pos_y,
        total_len: int,
        dtype,
        device,
        cond_drop_mask: Optional[torch.Tensor] = None,
    ):
        B = q_pos_x.shape[0]
        attn_bias = torch.zeros(B, 1, total_len, total_len, device=device, dtype=dtype)

        if (not self.use_layout_bias) or (layout is None):
            return attn_bias

        if layout.dim() != 3 or layout.size(-1) != 5:
            raise ValueError(f"layout must be [B,N,5], got {tuple(layout.shape)}")

        layout = layout.to(device=device)
        N = min(layout.size(1), self.buffer_size)
        if N <= 0:
            return attn_bias

        boxes = layout[:, :N, 1:5].float().clamp(0.0, 1.0)

        if layout_mask is not None:
            valid = layout_mask[:, :N].to(device=device).float()
        else:
            valid = torch.ones(B, N, device=device, dtype=torch.float32)

        if cond_drop_mask is not None:
            drop = cond_drop_mask.view(B).to(device=device, dtype=torch.float32)
            valid = valid * (1.0 - drop).unsqueeze(-1)

        qx = (q_pos_x.float() + 0.5) / float(self.seq_w)
        qy = (q_pos_y.float() + 0.5) / float(self.seq_h)

        x1, y1, x2, y2 = boxes.unbind(-1)
        inside = (
            (qx.unsqueeze(-1) >= x1.unsqueeze(1)) &
            (qx.unsqueeze(-1) <= x2.unsqueeze(1)) &
            (qy.unsqueeze(-1) >= y1.unsqueeze(1)) &
            (qy.unsqueeze(-1) <= y2.unsqueeze(1))
        )

        is_legal_attn = inside & (valid.unsqueeze(1) > 0.5)

        neg_val = -float(getattr(self, "layout_bias_value", 10000.0))
        if dtype == torch.float16:
            neg_val = max(neg_val, float(torch.finfo(torch.float16).min))

        neg = torch.full((), neg_val, device=device, dtype=dtype)
        bias_patch = torch.where(
            is_legal_attn,
            torch.zeros((), device=device, dtype=dtype),
            neg,
        )

        K = q_pos_x.shape[1]
        attn_bias[:, :, self.buffer_size:self.buffer_size + K, :N] = bias_patch.unsqueeze(1)

        return attn_bias

    # ----------------------------------------------------------
    # Encoder / Decoder
    # ----------------------------------------------------------
    def forward_mae_encoder(
        self,
        x,
        mask,
        cond_tokens,
        text_ctx_enc,
        text_ctx_mask_enc,
        layout=None,
        layout_mask=None,
        cond_drop_mask=None,
    ):
        x = self.z_proj(x)
        bsz, _, embed_dim = x.shape

        x = torch.cat(
            [torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device, dtype=x.dtype), x],
            dim=1,
        )
        mask_with_buffer = torch.cat(
            [torch.zeros(bsz, self.buffer_size, device=x.device, dtype=mask.dtype), mask],
            dim=1,
        )

        if cond_drop_mask is not None:
            uncond_tokens = (
                self.fake_latent.expand(bsz, embed_dim)
                .to(device=x.device, dtype=cond_tokens.dtype)
                .unsqueeze(1)
                .repeat(1, self.buffer_size, 1)
            )
            cond_tokens = cond_drop_mask * uncond_tokens + (1.0 - cond_drop_mask) * cond_tokens

        x[:, :self.buffer_size] = cond_tokens.to(device=x.device, dtype=x.dtype)

        if not (self.use_2d_rope and self.disable_learned_pos_emb_when_rope):
            x = x + self.encoder_pos_embed_learned.to(dtype=x.dtype)

        x = self.z_proj_ln(x)

        keep = (mask_with_buffer == 0)
        keep_len = int(keep.sum(dim=1).max().item())
        ids_keep = keep.nonzero(as_tuple=False)[:, 1].view(bsz, keep_len)
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim))

        pos_x = self.rope_pos_x_full.to(device=x.device)[ids_keep]
        pos_y = self.rope_pos_y_full.to(device=x.device)[ids_keep]

        if self.layout_bias_on_encoder:
            q_pos_x = pos_x[:, self.buffer_size:]
            q_pos_y = pos_y[:, self.buffer_size:]
            attn_bias = self._build_layout_attn_bias(
                layout=layout, layout_mask=layout_mask,
                q_pos_x=q_pos_x, q_pos_y=q_pos_y,
                total_len=keep_len, dtype=x.dtype, device=x.device,
                cond_drop_mask=cond_drop_mask,
            )
        else:
            attn_bias = torch.zeros(bsz, 1, keep_len, keep_len, device=x.device, dtype=x.dtype)

        text_ctx_enc = text_ctx_enc.to(device=x.device, dtype=x.dtype)
        text_ctx_mask_enc = text_ctx_mask_enc.to(device=x.device) if text_ctx_mask_enc is not None else None

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x, pos_x, pos_y, attn_bias, text_ctx_enc, text_ctx_mask_enc)
        else:
            for block in self.encoder_blocks:
                x = block(x, pos_x, pos_y, attn_bias, text_ctx_enc, text_ctx_mask_enc)

        x = self.encoder_norm(x)
        return x

    def forward_mae_decoder(
        self,
        x,
        mask,
        text_ctx_dec,
        text_ctx_mask_dec,
        layout=None,
        layout_mask=None,
        cond_drop_mask=None,
    ):
        x = self.decoder_embed(x)

        mask_with_buffer = torch.cat(
            [torch.zeros(x.size(0), self.buffer_size, device=x.device, dtype=mask.dtype), mask],
            dim=1,
        )

        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        x = x_after_pad

        if not (self.use_2d_rope and self.disable_learned_pos_emb_when_rope):
            x = x + self.decoder_pos_embed_learned.to(dtype=x.dtype)

        bsz, total_len, dim = x.shape

        pos_x = self.rope_pos_x_full.to(device=x.device).unsqueeze(0).expand(bsz, -1)
        pos_y = self.rope_pos_y_full.to(device=x.device).unsqueeze(0).expand(bsz, -1)

        if self.layout_bias_on_decoder:
            q_pos_x = pos_x[:, self.buffer_size:]
            q_pos_y = pos_y[:, self.buffer_size:]
            attn_bias = self._build_layout_attn_bias(
                layout=layout, layout_mask=layout_mask,
                q_pos_x=q_pos_x, q_pos_y=q_pos_y,
                total_len=total_len, dtype=x.dtype, device=x.device,
                cond_drop_mask=cond_drop_mask,
            )
        else:
            attn_bias = torch.zeros(bsz, 1, total_len, total_len, device=x.device, dtype=x.dtype)

        text_ctx_dec = text_ctx_dec.to(device=x.device, dtype=x.dtype)
        text_ctx_mask_dec = text_ctx_mask_dec.to(device=x.device) if text_ctx_mask_dec is not None else None

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x, pos_x, pos_y, attn_bias, text_ctx_dec, text_ctx_mask_dec)
        else:
            for block in self.decoder_blocks:
                x = block(x, pos_x, pos_y, attn_bias, text_ctx_dec, text_ctx_mask_dec)

        x = self.decoder_norm(x)
        x = x[:, self.buffer_size:]

        if not (self.use_2d_rope and self.disable_learned_pos_emb_when_rope):
            x = x + self.diffusion_pos_embed_learned.to(dtype=x.dtype)

        return x

    # ----------------------------------------------------------
    # Loss
    # ----------------------------------------------------------
    def forward_loss(self, z, target, mask, reduction="mean", timesteps=None):
        bsz, seq_len, _ = target.shape

        target_flat = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z_flat = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask_flat = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)

        loss = self.diffloss(
            z=z_flat,
            target=target_flat,
            mask=mask_flat,
            reduction=("mean" if reduction == "mean" else "none"),
            timesteps=timesteps,
            batch_size=bsz,
            tokens_per_sample=seq_len,
        )

        if reduction == "mean":
            return loss

        if loss.ndim == 1 and loss.shape[0] == bsz:
            return loss

        if loss.ndim != 1:
            loss = loss.view(-1)

        N = loss.shape[0]
        expected = bsz * seq_len * self.diffusion_batch_mul
        if N != expected:
            raise RuntimeError(f"Unexpected token-loss length: got {N}, expected {expected}.")

        mul = self.diffusion_batch_mul
        loss_3d = loss.view(mul, bsz, seq_len)
        mask_3d = mask_flat.view(mul, bsz, seq_len).to(dtype=loss_3d.dtype)

        denom = mask_3d.sum(dim=-1).clamp_min(1e-6)
        per_sample = (loss_3d.sum(dim=-1) / denom).mean(dim=0)
        return per_sample

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(
        self,
        imgs=None,
        latents=None,
        labels=None,
        text_emb=None,
        text_mask: Optional[torch.Tensor] = None,
        layout=None,
        layout_mask=None,
        reduction="mean",
        timesteps=None,
        external_mask=None,
    ):
        if imgs is not None:
            x_in = self.patchify(imgs)
        elif latents is not None:
            x_in = self.patchify(latents)
        else:
            raise ValueError("forward() requires either 'imgs' or 'latents'.")

        if (labels is None) and (text_emb is None) and (layout is None):
            raise ValueError("forward() requires labels or text_emb or layout.")

        device = x_in.device
        bsz = x_in.size(0)

        cond_drop_mask = None
        if self.training and self.label_drop_prob > 0:
            drop = (torch.rand(bsz, device=device) < self.label_drop_prob).float().view(bsz, 1, 1)
            cond_drop_mask = drop

        cond_tokens = self.build_cond_tokens(
            bsz=bsz,
            device=device,
            labels=labels,
            text_emb=text_emb,
            text_mask=text_mask,
            layout=layout,
            layout_mask=layout_mask,
        )

        if self.use_text_cross_attn:
            text_ctx_enc, text_ctx_mask_enc = self._build_text_ctx(
                text_emb=text_emb, text_mask=text_mask, bsz=bsz, device=device, dtype=x_in.dtype,
                proj=self.text_ctx_proj_enc, null_token=self.text_null_enc, drop_mask=cond_drop_mask
            )
            text_ctx_dec, text_ctx_mask_dec = self._build_text_ctx(
                text_emb=text_emb, text_mask=text_mask, bsz=bsz, device=device, dtype=x_in.dtype,
                proj=self.text_ctx_proj_dec, null_token=self.text_null_dec, drop_mask=cond_drop_mask
            )
        else:
            text_ctx_enc = torch.zeros(bsz, 0, self.z_proj.out_features, device=device, dtype=x_in.dtype)
            text_ctx_dec = torch.zeros(bsz, 0, self.decoder_embed.out_features, device=device, dtype=x_in.dtype)
            text_ctx_mask_enc = torch.zeros(bsz, 0, device=device, dtype=torch.bool)
            text_ctx_mask_dec = torch.zeros(bsz, 0, device=device, dtype=torch.bool)

        gt_latents = x_in.clone().detach()

        if external_mask is not None:
            mask = external_mask.to(device=device, dtype=x_in.dtype)
        else:
            orders = self.sample_orders(bsz=bsz, device=device)
            mask = self.random_masking(x_in, orders)

        x = self.forward_mae_encoder(
            x_in, mask, cond_tokens, text_ctx_enc, text_ctx_mask_enc,
            layout=layout, layout_mask=layout_mask,
            cond_drop_mask=cond_drop_mask,
        )
        z = self.forward_mae_decoder(
            x, mask, text_ctx_dec, text_ctx_mask_dec,
            layout=layout, layout_mask=layout_mask,
            cond_drop_mask=cond_drop_mask,
        )

        loss = self.forward_loss(z=z, target=gt_latents, mask=mask, reduction=reduction, timesteps=timesteps)

        if reduction == "none":
            if loss.ndim == 0:
                raise RuntimeError("reduction='none' but got scalar loss; per-sample [B] is required.")
            if loss.shape[0] != bsz:
                if loss.shape[0] == bsz * self.diffusion_batch_mul:
                     loss = loss.view(self.diffusion_batch_mul, bsz).mean(dim=0)
                else:
                     raise RuntimeError(f"reduction='none' expected [B]={bsz}, got {tuple(loss.shape)}")

        return loss

    # ----------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------
    @torch.no_grad()
    def sample_tokens(
        self,
        bsz,
        num_iter=64,
        cfg=1.0,
        cfg_schedule="linear",
        labels=None,
        temperature=1.0,
        progress=False,
        text_emb=None,
        text_mask: Optional[torch.Tensor] = None,
        layout=None,
        layout_mask=None,
    ):
        if (labels is not None) and (text_emb is not None):
            raise ValueError("labels and text_emb are mutually exclusive.")

        device = (
            text_emb.device if text_emb is not None else
            labels.device if labels is not None else
            layout.device if layout is not None else
            self.fake_latent.device
        )

        mask = torch.ones(bsz, self.seq_len, device=device, dtype=torch.float32)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=device, dtype=torch.float32)
        orders = self.sample_orders(bsz, device=device)

        indices = range(num_iter)
        if progress:
            indices = tqdm(indices)

        cond_tokens = self.build_cond_tokens(
            bsz=bsz,
            device=device,
            labels=labels,
            text_emb=text_emb,
            text_mask=text_mask,
            layout=layout,
            layout_mask=layout_mask,
        )

        D = cond_tokens.size(-1)
        uncond_tokens = (
            self.fake_latent.expand(bsz, D)
            .to(device=device, dtype=cond_tokens.dtype)
            .unsqueeze(1)
            .repeat(1, self.buffer_size, 1)
        )

        if self.use_text_cross_attn:
            text_ctx_enc_c, text_ctx_mask_enc_c = self._build_text_ctx(
                text_emb, text_mask, bsz, device, dtype=torch.float32,
                proj=self.text_ctx_proj_enc, null_token=self.text_null_enc, drop_mask=None
            )
            text_ctx_dec_c, text_ctx_mask_dec_c = self._build_text_ctx(
                text_emb, text_mask, bsz, device, dtype=torch.float32,
                proj=self.text_ctx_proj_dec, null_token=self.text_null_dec, drop_mask=None
            )

            if text_ctx_enc_c.shape[1] > 0:
                T_enc = text_ctx_enc_c.shape[1]
                text_ctx_enc_u = self.text_null_enc.to(device=device, dtype=torch.float32).expand(bsz, T_enc, -1)
                text_ctx_mask_enc_u = text_ctx_mask_enc_c
            else:
                text_ctx_enc_u = text_ctx_enc_c
                text_ctx_mask_enc_u = text_ctx_mask_enc_c

            if text_ctx_dec_c.shape[1] > 0:
                T_dec = text_ctx_dec_c.shape[1]
                text_ctx_dec_u = self.text_null_dec.to(device=device, dtype=torch.float32).expand(bsz, T_dec, -1)
                text_ctx_mask_dec_u = text_ctx_mask_dec_c
            else:
                text_ctx_dec_u = text_ctx_dec_c
                text_ctx_mask_dec_u = text_ctx_mask_dec_c
        else:
            text_ctx_enc_c = torch.zeros(bsz, 0, self.z_proj.out_features, device=device, dtype=torch.float32)
            text_ctx_dec_c = torch.zeros(bsz, 0, self.decoder_embed.out_features, device=device, dtype=torch.float32)
            text_ctx_mask_enc_c = torch.zeros(bsz, 0, device=device, dtype=torch.bool)
            text_ctx_mask_dec_c = torch.zeros(bsz, 0, device=device, dtype=torch.bool)

            text_ctx_enc_u = text_ctx_enc_c
            text_ctx_dec_u = text_ctx_dec_c
            text_ctx_mask_enc_u = text_ctx_mask_enc_c
            text_ctx_mask_dec_u = text_ctx_mask_dec_c

        if layout is not None:
            layout_c = layout
            if layout_mask is None:
                layout_mask_c = torch.ones(layout.shape[0], layout.shape[1], device=device, dtype=torch.bool)
            else:
                layout_mask_c = layout_mask.to(device=device).bool()

            layout_u = torch.zeros_like(layout_c)
            layout_mask_u = torch.zeros_like(layout_mask_c)
        else:
            layout_c = None
            layout_mask_c = None
            layout_u = None
            layout_mask_u = None

        for step in indices:
            cur_tokens = tokens.clone()

            if cfg != 1.0:
                tokens_in = torch.cat([tokens, tokens], dim=0)
                mask_in = torch.cat([mask, mask], dim=0)
                cond_tokens_in = torch.cat([cond_tokens, uncond_tokens], dim=0)

                text_ctx_enc_in = torch.cat([text_ctx_enc_c, text_ctx_enc_u], dim=0)
                text_ctx_dec_in = torch.cat([text_ctx_dec_c, text_ctx_dec_u], dim=0)
                text_ctx_mask_enc_in = torch.cat([text_ctx_mask_enc_c, text_ctx_mask_enc_u], dim=0)
                text_ctx_mask_dec_in = torch.cat([text_ctx_mask_dec_c, text_ctx_mask_dec_u], dim=0)

                layout_in = torch.cat([layout_c, layout_u], dim=0) if layout is not None else None
                layout_mask_in = torch.cat([layout_mask_c, layout_mask_u], dim=0) if (layout_mask_c is not None) else None
            else:
                tokens_in = tokens
                mask_in = mask
                cond_tokens_in = cond_tokens

                text_ctx_enc_in = text_ctx_enc_c
                text_ctx_dec_in = text_ctx_dec_c
                text_ctx_mask_enc_in = text_ctx_mask_enc_c
                text_ctx_mask_dec_in = text_ctx_mask_dec_c

                layout_in = layout_c
                layout_mask_in = layout_mask_c

            x = self.forward_mae_encoder(
                tokens_in, mask_in, cond_tokens_in,
                text_ctx_enc_in, text_ctx_mask_enc_in,
                layout=layout_in, layout_mask=layout_mask_in, cond_drop_mask=None
            )
            z = self.forward_mae_decoder(
                x, mask_in,
                text_ctx_dec_in, text_ctx_mask_dec_in,
                layout=layout_in, layout_mask=layout_mask_in, cond_drop_mask=None
            )

            mask_ratio = float(np.cos(math.pi / 2.0 * (step + 1) / num_iter))
            mask_len_int = int(np.floor(self.seq_len * mask_ratio))

            with torch.no_grad():
                remaining = (mask.sum(dim=-1) - 1.0)
                max_allow = int(torch.clamp(remaining.min(), min=1.0).item())

            if mask_len_int < 1:
                mask_len_int = 1
            if mask_len_int > max_allow:
                mask_len_int = max_allow

            mask_len_t = torch.tensor(mask_len_int, device=device, dtype=torch.long)
            mask_next = mask_by_order(mask_len_t, orders, bsz, self.seq_len)

            if step >= num_iter - 1:
                mask_to_pred = mask.bool()
            else:
                mask_to_pred = torch.logical_xor(mask.bool(), mask_next.bool())

            mask = mask_next.float()

            if cfg != 1.0:
                mask_to_pred_in = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            else:
                mask_to_pred_in = mask_to_pred

            z_sel = z[mask_to_pred_in.nonzero(as_tuple=True)]

            if cfg_schedule == "linear":
                cfg_iter = float(1.0 + (cfg - 1.0) * (self.seq_len - mask_len_int) / self.seq_len)
            elif cfg_schedule == "constant":
                cfg_iter = float(cfg)
            else:
                raise NotImplementedError

            sampled_token_latent = self.diffloss.sample(z_sel, temperature, cfg_iter)

            if cfg != 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred_in.chunk(2, dim=0)
            else:
                mask_to_pred = mask_to_pred_in

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        tokens = self.unpatchify(tokens)
        return tokens


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=768,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280,
        encoder_depth=20,
        encoder_num_heads=16,
        decoder_embed_dim=1280,
        decoder_depth=20,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model