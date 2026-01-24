# text_encoder_clip.py

from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class HFCLIPTextEncoder(nn.Module):
    """
    从本地目录加载 CLIP 文本模型（完全离线）。
    默认 encode() 返回 token-level embedding: [B, T, D]，用于 Cross-Attention / Deep Fusion。

    用法:
        encoder = HFCLIPTextEncoder(model_dir, device="cuda")
        text_tokens = encoder.encode(["a cat", "a dog"])          # [B, T, D]
        text_pooled = encoder.encode_pooled(["a cat", "a dog"])   # [B, D]

    进阶:
        tokens, attn_mask = encoder.encode(["a cat"], return_attention_mask=True)
        # tokens: [B,T,D], attn_mask: [B,T] (1=valid,0=pad)
    """

    def __init__(
        self,
        model_dir: str,
        device: Union[str, torch.device] = "cuda",
        return_tokens_default: bool = True,   # True: encode() 默认返回 [B,T,D]
        l2_normalize_default: bool = False,   # 默认不做 L2 normalize（cross-attn 更自然）
        pad_to_pooled_default: bool = True,   # token-level 时：pad token 替换为 pooled embedding（更稳）
        max_length: Optional[int] = None,     # 可选：强制最大长度（不传则用 tokenizer 默认）
    ):
        super().__init__()
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.return_tokens_default = bool(return_tokens_default)
        self.l2_normalize_default = bool(l2_normalize_default)
        self.pad_to_pooled_default = bool(pad_to_pooled_default)
        self.max_length = int(max_length) if max_length is not None else None

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model = CLIPTextModel.from_pretrained(
            model_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.text_dim = int(self.model.config.hidden_size)
        self.eos_token_id = int(self.tokenizer.eos_token_id)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @torch.inference_mode()
    def encode(
        self,
        captions: List[str],
        return_tokens: Optional[bool] = None,
        l2_normalize: Optional[bool] = None,
        return_attention_mask: bool = False,
        pad_to_pooled: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        captions: list[str]
        return:
          - return_tokens=True  -> tokens: [B, T, D]  (token-level)
          - return_tokens=False -> pooled: [B, D]     (pooled)

        如果 return_attention_mask=True：
          - token-level: (tokens, attn_mask)
          - pooled: (pooled, attn_mask)   # 仍返回 [B,T] 的 mask，方便你 debug/对齐
        """
        if return_tokens is None:
            return_tokens = self.return_tokens_default
        if l2_normalize is None:
            l2_normalize = self.l2_normalize_default
        if pad_to_pooled is None:
            pad_to_pooled = self.pad_to_pooled_default

        # 允许空 batch（有时调试会用到）
        if captions is None:
            captions = []
        if len(captions) == 0:
            if return_tokens:
                # 返回一个空的 tokens：shape [0,0,D]
                tokens = torch.empty(0, 0, self.text_dim, device=self.device)
                attn_mask = torch.empty(0, 0, dtype=torch.long, device=self.device)
                return (tokens, attn_mask) if return_attention_mask else tokens
            pooled = torch.empty(0, self.text_dim, device=self.device)
            attn_mask = torch.empty(0, 0, dtype=torch.long, device=self.device)
            return (pooled, attn_mask) if return_attention_mask else pooled

        tok_kwargs = dict(
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if self.max_length is not None:
            tok_kwargs["max_length"] = self.max_length

        inputs = self.tokenizer(captions, **tok_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        attn_mask = inputs.get("attention_mask", None)  # [B,T] long
        if attn_mask is None:
            # fallback（理论上不会发生）
            attn_mask = torch.ones(outputs.last_hidden_state.shape[:2], device=self.device, dtype=torch.long)

        if return_tokens:
            # token-level: [B,T,D]
            tokens = outputs.last_hidden_state  # [B,T,D]

            # 可选：把 padding 位置替换成 pooled(EOS) embedding，让 padding 更“无害”
            # （因为 cross-attn 如果没有 ctx_mask，pad token 会带来噪声）
            if pad_to_pooled and attn_mask.ndim == 2:
                pooled = outputs.pooler_output  # 通常就是 EOS embedding（已过 LN）
                if pooled is None:
                    # fallback：取最后一个有效 token
                    last_idx = (attn_mask.sum(dim=1) - 1).clamp_min(0)  # [B]
                    pooled = tokens[torch.arange(tokens.shape[0], device=self.device), last_idx]  # [B,D]

                pad = (attn_mask == 0).unsqueeze(-1)  # [B,T,1]
                tokens = torch.where(pad, pooled.unsqueeze(1), tokens)

            if l2_normalize:
                tokens = self._normalize(tokens)

            return (tokens, attn_mask) if return_attention_mask else tokens

        # pooled: [B,D]
        pooled = outputs.pooler_output
        if pooled is None:
            # fallback：mean pool（不太 CLIP 正统，但能跑）
            pooled = outputs.last_hidden_state.mean(dim=1)

        if l2_normalize:
            pooled = self._normalize(pooled)

        return (pooled, attn_mask) if return_attention_mask else pooled

    @torch.inference_mode()
    def encode_pooled(self, captions: List[str], l2_normalize: bool = True) -> torch.Tensor:
        """显式拿 pooled embedding [B,D]（旧 baseline / reward / debug 用）"""
        return self.encode(captions, return_tokens=False, l2_normalize=l2_normalize, return_attention_mask=False)
