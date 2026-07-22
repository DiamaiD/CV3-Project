import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(ch, max_groups=32):
    g = max_groups
    while g > 1 and ch % g != 0:
        g //= 2
    return nn.GroupNorm(g, ch, eps=1e-6)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: depthwise 7x7 for spatial mixing, then a pointwise
    4x MLP with GELU for channel mixing. Same (in_ch, out_ch) interface and
    zero-init residual convention as ResBlock."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 7, padding=3, groups=in_ch)
        self.norm = _group_norm(in_ch, max_groups=1)
        self.pw1 = nn.Conv2d(in_ch, 4 * out_ch, 1)
        self.pw2 = nn.Conv2d(4 * out_ch, out_ch, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x):
        h = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))
        return self.skip(x) + h


class MobileBlock(nn.Module):
    """MobileNetV2-style inverted residual: 1x1 expand 4x, depthwise 3x3, 1x1
    project -- most of the compute lives in cheap pointwise/depthwise convs, so
    it trains noticeably faster than ResBlock at similar parameter count.
    Pre-activation norms and zero-init projection, matching ResBlock's style."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = 4 * in_ch
        self.norm1 = _group_norm(in_ch)
        self.expand = nn.Conv2d(in_ch, mid, 1)
        self.norm2 = _group_norm(mid)
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.norm3 = _group_norm(mid)
        self.project = nn.Conv2d(mid, out_ch, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        h = self.expand(F.silu(self.norm1(x)))
        h = self.dw(F.silu(self.norm2(h)))
        h = self.project(F.silu(self.norm3(h)))
        return self.skip(x) + h


VAE_BLOCKS = {"res": ResBlock, "convnext": ConvNeXtBlock, "mobile": MobileBlock}


class AttnBlock(nn.Module):
    def __init__(self, ch, n_heads=4):
        super().__init__()
        assert ch % n_heads == 0, "channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.norm = _group_norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        to_heads = lambda t: t.reshape(B, self.n_heads, C // self.n_heads, H * W).transpose(2, 3)
        out = F.scaled_dot_product_attention(to_heads(q), to_heads(k), to_heads(v))
        out = out.transpose(2, 3).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class CNNVAE(nn.Module):
    def __init__(self, latent_ch=32, latent_grid=8, img_size=64,
                 base_ch=64, ch_mult=(1, 2, 4), enc_res_blocks=1, attn=True, dec_res_blocks=None,
                 block="res", mid_blocks=1):
        super().__init__()
        Block = VAE_BLOCKS[block]
        n_stages = int(round(math.log2(img_size / latent_grid)))
        assert n_stages >= 1 and img_size == latent_grid * (2 ** n_stages), \
            f"img_size {img_size} must be latent_grid {latent_grid} x a power of two (>=2)"
        self.latent_ch = latent_ch
        self.latent_grid = latent_grid
        # Bound the sampled log-variance during training (None = off). Pure numeric
        # safety with KL weight 0: the log-variance head is unregularized and can
        # drift to huge values (a 1e26 KL diagnostic, exp() overflow risk); this
        # caps it. Inactive on healthy runs (logvar stays well inside the range), so
        # it never changes a well-behaved model. Not a Parameter/buffer -> checkpoints
        # are unaffected, and encode() uses only mu so inference is untouched.
        self.logvar_clamp = None
        if dec_res_blocks is None:
            dec_res_blocks = enc_res_blocks
        self.enc_res_blocks = enc_res_blocks
        self.dec_res_blocks = dec_res_blocks

        widths = [base_ch * ch_mult[min(i, len(ch_mult) - 1)] for i in range(n_stages)]

        self.conv_in = nn.Conv2d(3, widths[0], 3, padding=1)
        enc, cur = [], widths[0]
        for i in range(n_stages):
            enc.append(Downsample(cur))
            if enc_res_blocks > 0:
                for _ in range(enc_res_blocks):
                    enc.append(Block(cur, widths[i])); cur = widths[i]
            else:
                enc.append(nn.Conv2d(cur, widths[i], 3, padding=1)); cur = widths[i]
                enc.append(nn.SiLU())
        self.enc = nn.Sequential(*enc)

        if enc_res_blocks > 0:
            mid_enc = ([Block(cur, cur)] * 0 + [Block(cur, cur) for _ in range(mid_blocks)]
                       + ([AttnBlock(cur)] if attn else [])
                       + [Block(cur, cur) for _ in range(mid_blocks)])
            self.enc_mid = nn.Sequential(*mid_enc)
        else:
            self.enc_mid = nn.Identity()
        self.enc_norm = _group_norm(cur)
        self.to_mu = nn.Conv2d(cur, latent_ch, 3, padding=1)
        self.to_logvar = nn.Conv2d(cur, latent_ch, 3, padding=1)
        nn.init.zeros_(self.to_logvar.weight); nn.init.zeros_(self.to_logvar.bias)

        self.dec_in = nn.Conv2d(latent_ch, cur, 3, padding=1)
        if dec_res_blocks > 0:
            mid_dec = ([Block(cur, cur) for _ in range(mid_blocks)]
                       + ([AttnBlock(cur)] if attn else [])
                       + [Block(cur, cur) for _ in range(mid_blocks)])
            self.dec_mid = nn.Sequential(*mid_dec)
        else:
            self.dec_mid = nn.Identity()
        dec = []
        for i in reversed(range(n_stages)):
            if dec_res_blocks > 0:
                for _ in range(dec_res_blocks):
                    dec.append(Block(cur, widths[i])); cur = widths[i]
            else:
                dec.append(nn.Conv2d(cur, widths[i], 3, padding=1)); cur = widths[i]
                dec.append(nn.SiLU())
            dec.append(Upsample(cur))
        self.dec = nn.Sequential(*dec)
        self.dec_norm = _group_norm(cur)
        self.conv_out = nn.Conv2d(cur, 3, 3, padding=1)

    def _encode_trunk(self, x):
        h = self.enc_mid(self.enc(self.conv_in(x)))
        return F.silu(self.enc_norm(h))

    def encode_dist(self, x):
        h = self._encode_trunk(x)
        return self.to_mu(h), self.to_logvar(h)

    def encode(self, x):
        return self.to_mu(self._encode_trunk(x))

    def decode(self, z):
        h = self.dec(self.dec_mid(self.dec_in(z)))
        return torch.sigmoid(self.conv_out(F.silu(self.dec_norm(h))))

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        if self.logvar_clamp is not None:
            logvar = logvar.clamp(-self.logvar_clamp, self.logvar_clamp)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, scale=1000.0, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.max_period = max_period

    def forward(self, t):
        t = t.float().reshape(-1) * self.scale
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device).float() / max(1, half)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, d_model),
        )

        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def _attn(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self._attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, latent_ch=32, context_len=5, grid=8, chunk_len=5,
                 d_model=256, n_layers=6, n_heads=8, dropout=0.0, latent_scale=1.0):
        super().__init__()
        self.latent_ch = latent_ch
        self.context_len = context_len
        self.grid = grid
        self.chunk_len = chunk_len
        self.register_buffer("latent_scale", torch.as_tensor(latent_scale, dtype=torch.float32))

        self.seq_len = (context_len + chunk_len) * grid * grid
        self.patch = nn.Linear(latent_ch, d_model)
        self.time_pos = nn.Parameter(torch.zeros(1, context_len + chunk_len, 1, d_model))
        self.space_pos = nn.Parameter(torch.zeros(1, 1, grid * grid, d_model))
        self.segment = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.time_pos, std=0.02)
        nn.init.normal_(self.space_pos, std=0.02)
        nn.init.normal_(self.segment, std=0.02)

        self.t_embed = SinusoidalPositionEmbedding(d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.head = nn.Linear(d_model, latent_ch)

        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_noisy, time_t, context_latents):
        B, K = x_noisy.shape[0], x_noisy.shape[1]
        gh = gw = self.grid

        T = self.context_len
        g2 = gh * gw
        z = torch.cat([context_latents, x_noisy], dim=1)
        z = z.permute(0, 1, 3, 4, 2).reshape(B, T + K, g2, self.latent_ch)
        x = self.patch(z) + self.time_pos + self.space_pos
        seg = torch.cat([self.segment[0].expand(T, g2, -1),
                         self.segment[1].expand(K, g2, -1)], dim=0)
        x = (x + seg).reshape(B, (T + K) * g2, -1)

        cond = self.t_mlp(self.t_embed(time_t))
        for block in self.blocks:
            x = block(x, cond)

        shift, scale = self.ada_out(cond).chunk(2, dim=1)
        x = _modulate(self.norm_out(x), shift, scale)
        x = x[:, -K * gh * gw:]
        x = self.head(x)

        return x.reshape(B, K, gh, gw, self.latent_ch).permute(0, 1, 4, 2, 3)
