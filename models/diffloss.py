import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from diffusion import create_diffusion


class DiffLoss(nn.Module):
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")

        if isinstance(num_sampling_steps, int):
            timestep_respacing = str(num_sampling_steps)
        else:
            timestep_respacing = num_sampling_steps
        self.gen_diffusion = create_diffusion(timestep_respacing=timestep_respacing, noise_schedule="cosine")

    def forward(
        self,
        target,
        z,
        mask=None,
        reduction="mean",
        timesteps=None,
        batch_size=None,
        tokens_per_sample=None,
    ):
        """
        Args:
            target: [N, C]
            z: [N, D]
            mask: [N]
            reduction: 'mean' or 'none'
            timesteps: None, int, or Tensor
        """
        N = target.shape[0]
        device = target.device

        if timesteps is None:
            t = torch.randint(0, self.train_diffusion.num_timesteps, (N,), device=device)
        elif isinstance(timesteps, int):
            t = torch.full((N,), int(timesteps), device=device, dtype=torch.long)
        elif torch.is_tensor(timesteps):
            if timesteps.ndim != 1:
                raise ValueError(f"timesteps must be 1-D, got shape {tuple(timesteps.shape)}")
            if timesteps.shape[0] == N:
                t = timesteps.to(device=device, dtype=torch.long)
            else:
                if batch_size is None or tokens_per_sample is None:
                    raise ValueError(
                        f"timesteps has shape {tuple(timesteps.shape)} but cannot expand to per-token "
                        f"without batch_size and tokens_per_sample."
                    )
                B = int(batch_size)
                L = int(tokens_per_sample)
                if timesteps.shape[0] != B:
                    raise ValueError(
                        f"timesteps length {timesteps.shape[0]} doesn't match batch_size {B}"
                    )
                if (B * L) <= 0 or (N % (B * L)) != 0:
                    raise ValueError(
                        f"Cannot infer diffusion_batch_mul: N={N}, B={B}, L={L}. Need N divisible by (B*L)."
                    )
                mul = N // (B * L)
                t = timesteps.to(device=device, dtype=torch.long).repeat_interleave(L * mul)
        else:
            raise TypeError(f"Unsupported timesteps type: {type(timesteps)}")

        model_kwargs = dict(c=z)

        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]

        if loss.dim() > 1:
            loss = loss.mean(dim=1)

        if mask is not None:
            if mask.dtype == torch.bool:
                mask_f = mask.to(dtype=loss.dtype)
            else:
                mask_f = mask.to(dtype=loss.dtype)
            loss = loss * mask_f

        if reduction == "mean":
            if mask is not None:
                denom = mask_f.sum().clamp_min(1e-6)
                return loss.sum() / denom
            return loss.mean()

        elif reduction == "none":
            if batch_size is None or tokens_per_sample is None:
                return loss

            B = int(batch_size)
            L = int(tokens_per_sample)
            if (B * L) <= 0 or (N % (B * L)) != 0:
                raise ValueError(
                    f"Cannot reshape loss to per-sample: N={N}, B={B}, L={L}. Need N divisible by (B*L)."
                )
            mul = N // (B * L)

            loss_3d = loss.view(mul, B, L)

            if mask is not None:
                mask_3d = mask_f.view(mul, B, L)
                denom = mask_3d.sum(dim=-1).clamp_min(1e-6)
                per_sample = (loss_3d.sum(dim=-1) / denom).mean(dim=0)
            else:
                per_sample = loss_3d.mean(dim=-1).mean(dim=0)

            return per_sample

        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def sample(self, z, temperature=1.0, cfg=1.0):
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).to(z.device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).to(z.device)
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )
        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)