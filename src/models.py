import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNVAE(nn.Module):
    """ Part 1: a convolutional VAE that compresses a 64x64 frame into a SPATIAL latent grid
    (latent_ch x latent_grid x latent_grid). Its KL term makes that latent normalized AND smooth
    -- i.e. good for the dynamics model, not merely good for reconstruction.

    `latent_grid` sets how far the frame is spatially compressed (the number of stride-2 stages is
    log2(img_size / latent_grid), and the encoder + decoder are built to match):
      * 8  -> 8x8   : more compression, 64 DiT tokens/frame, ~2 GB latent cache.
      * 16 -> 16x16 : less compression -- motion is more spatially LOCAL in the latent (a moving
                      ball shifts a few cells rather than reshaping one cell's vector), which is
                      often easier for the dynamics model to predict -- at 4x the token count
                      (256/frame, ~16x attention) and 4x the latent-cache memory (~8 GB).

    The encoder produces a per-cell Gaussian posterior (mu, logvar). Training samples
    z ~ N(mu, sigma^2) (reparameterized) and minimizes reconstruction + beta * KL to N(0, 1).
    The dynamics pipeline uses the deterministic posterior mean mu as 'the latent' (see
    `encode`); the KL keeps mu ~unit-Gaussian per channel, so no external normalization is
    needed downstream. """
    def __init__(self, latent_ch=32, latent_grid=8, img_size=64):
        super().__init__()
        n_stages = int(round(math.log2(img_size / latent_grid)))   # stride-2 down/up steps
        assert n_stages >= 1 and img_size == latent_grid * (2 ** n_stages), \
            f"img_size {img_size} must be latent_grid {latent_grid} x a power of two (>=2)"
        self.latent_ch = latent_ch
        self.latent_grid = latent_grid

        # Encoder trunk: (n_stages - 1) stride-2 convs (channels 3 -> 32 -> 64 -> ...); the final
        # stride-2 stage is the (mu, logvar) heads. At latent_grid=8 this is 64->32->16 then ->8.
        trunk_widths = [32 * (2 ** i) for i in range(n_stages - 1)]
        enc_layers, prev = [], 3
        for w in trunk_widths:
            enc_layers += [nn.Conv2d(prev, w, 3, stride=2, padding=1), nn.ReLU()]
            prev = w
        self.enc = nn.Sequential(*enc_layers)              # -> (B, prev, 2*latent_grid, 2*latent_grid)

        # Two heads map the trunk down to the latent grid: mean and log-variance.
        self.to_mu = nn.Conv2d(prev, latent_ch, 3, stride=2, padding=1)
        self.to_logvar = nn.Conv2d(prev, latent_ch, 3, stride=2, padding=1)

        # Decoder mirrors the encoder: n_stages stride-2 transposed convs back up to img_size.
        dec_layers, prev = [], latent_ch
        for w in reversed(trunk_widths):
            dec_layers += [nn.ConvTranspose2d(prev, w, 3, stride=2, padding=1, output_padding=1),
                           nn.ReLU()]
            prev = w
        dec_layers += [nn.ConvTranspose2d(prev, 3, 3, stride=2, padding=1, output_padding=1),
                       nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def encode_dist(self, x):
        """Posterior over the latent grid: (mu, logvar), each (B, latent_ch, latent_grid, latent_grid)."""
        h = self.enc(x)
        return self.to_mu(h), self.to_logvar(h)

    def encode(self, x):
        """Deterministic latent for the dynamics pipeline: the posterior mean mu."""
        return self.to_mu(self.enc(x))  # (B, latent_ch, latent_grid, latent_grid)

    def decode(self, z):
        return self.decoder(z)  # (B, 3, 64, 64)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        """Training forward: returns (reconstruction, mu, logvar)."""
        mu, logvar = self.encode_dist(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class SinusoidalPositionEmbedding(nn.Module):
    """ Encodes a continuous scalar -- here the rectified-flow time t in [0, 1] -- as a
    `dim`-dimensional sinusoidal vector, the same construction as Transformer positional
    encodings / diffusion timestep embeddings. t is scaled up first so the geometric band of
    frequencies is well separated over the unit interval, letting the network tell nearby t
    apart. Returns (B, dim). """
    def __init__(self, dim, scale=1000.0, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.max_period = max_period

    def forward(self, t):
        # t: (B,) float in [0, 1].
        t = t.float().reshape(-1) * self.scale
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device).float() / max(1, half)
        )
        args = t[:, None] * freqs[None]                  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:                            # pad if dim is odd
            emb = F.pad(emb, (0, 1))
        return emb                                       # (B, dim)


def _modulate(x, shift, scale):
    """adaLN modulation: x (B, L, D) scaled/shifted by per-sample (B, D) vectors."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """ A Diffusion Transformer block (Peebles & Xie, 2023).

    Bidirectional (UNMASKED) multi-head self-attention + an MLP, each wrapped in Adaptive
    LayerNorm (adaLN): the per-block scale (gamma), shift (beta) and residual gate (alpha) are
    produced from the timestep/conditioning embedding by a small MLP, so the block's behaviour is
    conditioned on the diffusion time t. The modulation projection is zero-initialised
    (adaLN-Zero), so each block starts as the identity and training is stable. Attention uses
    F.scaled_dot_product_attention with NO causal mask -- every cell attends to every cell. """
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, d_model),
        )

        # adaLN-Zero: condition -> (shift, scale, gate) for both the attention and the MLP.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def _attn(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # each (B, n_heads, L, head_dim)
        out = F.scaled_dot_product_attention(              # bidirectional: no causal mask
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

    def forward(self, x, cond):
        # cond: (B, d_model) timestep embedding.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self._attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiffusionTransformer(nn.Module):
    """ Stage 2: a DiT that predicts the rectified-flow VELOCITY field of a CHUNK of the next
    `chunk_len` (K) frame latents jointly -- chunk prediction, which lets the model lay down a
    whole short rollout at once.

    Inputs:
      * x_noisy         (B, K, latent_ch, gh, gw)      -- the interpolated state z_t for K frames
      * time_t          (B,)                           -- continuous flow time in [0, 1]
      * context_latents (B, T, latent_ch, gh, gw)      -- the T context-frame latents

    The context latents are merged into the channel axis (T*latent_ch) and BROADCAST across the K
    chunk frames, then concatenated with each noisy frame ((T+1)*latent_ch channels, e.g. 5 context
    + 1 noisy = 6*32 = 192). The (K, gh, gw) grid is flattened into a length K*gh*gw token sequence
    (one token per (frame, cell)); bidirectional adaLN DiT blocks process it, conditioned on
    time_t; a final adaLN + linear head unpatchifies back to (B, K, latent_ch, gh, gw). The output
    is the predicted velocity v = z_1 - z_0 for all K frames, integrated by an Euler ODE solver at
    inference. """
    def __init__(self, latent_ch=32, context_len=5, grid=8, chunk_len=5,
                 d_model=256, n_layers=6, n_heads=8, dropout=0.0, latent_scale=1.0):
        super().__init__()
        self.latent_ch = latent_ch
        self.context_len = context_len
        self.grid = grid
        self.chunk_len = chunk_len
        # LDM scale factor: latents are normalized to ~unit variance for the flow model. The DiT
        # operates entirely in this normalized space; callers apply encode -> /latent_scale and
        # decode -> *latent_scale at the VAE boundary. Stored as a buffer so it rides in the
        # checkpoint. See train.build_latent_cache.
        self.register_buffer("latent_scale", torch.tensor(float(latent_scale)))
        self.seq_len = chunk_len * grid * grid           # K * gh * gw tokens
        self.in_ch = (context_len + 1) * latent_ch       # context frames (broadcast) + this noisy frame

        self.patch = nn.Linear(self.in_ch, d_model)      # patch size 1: one token per (frame, cell)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))

        # Timestep embedding -> conditioning vector shared by every block's adaLN.
        self.t_embed = SinusoidalPositionEmbedding(d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])

        # Final adaLN + zero-init head (predicts the velocity, so starts at 0).
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.head = nn.Linear(d_model, latent_ch)

        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_noisy, time_t, context_latents):
        # x_noisy: (B, K, latent_ch, gh, gw); context_latents: (B, T, latent_ch, gh, gw).
        B, K = x_noisy.shape[0], x_noisy.shape[1]
        gh = gw = self.grid

        # Context frames -> channels, then broadcast across the K chunk frames.
        ctx = context_latents.reshape(B, self.context_len * self.latent_ch, gh, gw)   # (B, T*Cl, gh, gw)
        ctx = ctx.unsqueeze(1).expand(-1, K, -1, -1, -1)                              # (B, K, T*Cl, gh, gw)
        x = torch.cat([ctx, x_noisy], dim=2)                                          # (B, K, (T+1)*Cl, gh, gw)

        # Flatten (K, gh, gw) into a sequence of K*gh*gw tokens, channels last.
        x = x.permute(0, 1, 3, 4, 2).reshape(B, K * gh * gw, self.in_ch)              # (B, L, in_ch)
        x = self.patch(x) + self.pos_emb[:, :x.shape[1]]                              # (B, L, d_model)

        cond = self.t_mlp(self.t_embed(time_t))          # (B, d_model)
        for block in self.blocks:
            x = block(x, cond)

        shift, scale = self.ada_out(cond).chunk(2, dim=1)
        x = _modulate(self.norm_out(x), shift, scale)
        x = self.head(x)                                 # (B, L, latent_ch)

        # Unpatchify: (B, K*gh*gw, latent_ch) -> (B, K, latent_ch, gh, gw).
        return x.reshape(B, K, gh, gw, self.latent_ch).permute(0, 1, 4, 2, 3)
