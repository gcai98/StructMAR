from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class HFCLIPTextEncoder(nn.Module):
    """
    Loads a CLIP text encoder from a local directory (offline).
    Default encode() returns token-level embedding: [B, T, D].
    """

    def __init__(
        self,
        model_dir: str,
        device: Union[str, torch.device] = "cuda",
        return_tokens_default: bool = True,   # True: encode() returns [B,T,D] by default
        l2_normalize_default: bool = False,   # Default False for cross-attn
        pad_to_pooled_default: bool = True,   # If True: replace pad tokens with pooled embedding
        max_length: Optional[int] = None,     # Optional: force max length
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
          - return_tokens=True  -> tokens: [B, T, D]
          - return_tokens=False -> pooled: [B, D]

        If return_attention_mask=True:
          - returns (embedding, attn_mask)
        """
        if return_tokens is None:
            return_tokens = self.return_tokens_default
        if l2_normalize is None:
            l2_normalize = self.l2_normalize_default
        if pad_to_pooled is None:
            pad_to_pooled = self.pad_to_pooled_default

        # Allow empty batch
        if captions is None:
            captions = []
        if len(captions) == 0:
            if return_tokens:
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
            # fallback
            attn_mask = torch.ones(outputs.last_hidden_state.shape[:2], device=self.device, dtype=torch.long)

        if return_tokens:
            tokens = outputs.last_hidden_state  # [B,T,D]

            # Optional: replace padding with pooled(EOS) embedding to reduce noise in cross-attn
            if pad_to_pooled and attn_mask.ndim == 2:
                pooled = outputs.pooler_output
                if pooled is None:
                    # fallback: use last valid token
                    last_idx = (attn_mask.sum(dim=1) - 1).clamp_min(0)
                    pooled = tokens[torch.arange(tokens.shape[0], device=self.device), last_idx]

                pad = (attn_mask == 0).unsqueeze(-1)  # [B,T,1]
                tokens = torch.where(pad, pooled.unsqueeze(1), tokens)

            if l2_normalize:
                tokens = self._normalize(tokens)

            return (tokens, attn_mask) if return_attention_mask else tokens

        # pooled: [B,D]
        pooled = outputs.pooler_output
        if pooled is None:
            # fallback: mean pool
            pooled = outputs.last_hidden_state.mean(dim=1)

        if l2_normalize:
            pooled = self._normalize(pooled)

        return (pooled, attn_mask) if return_attention_mask else pooled

    @torch.inference_mode()
    def encode_pooled(self, captions: List[str], l2_normalize: bool = True) -> torch.Tensor:
        """Explicitly get pooled embedding [B,D]."""
        return self.encode(captions, return_tokens=False, l2_normalize=l2_normalize, return_attention_mask=False)