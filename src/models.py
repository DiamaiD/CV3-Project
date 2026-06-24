import torch
import torch.nn as nn

class CNNVAE(nn.Module):
    """ Part 1: a convolutional VAE that compresses a frame into a SPATIAL latent grid
    (latent_ch x 8 x 8). Its KL term makes that latent normalized AND smooth -- i.e. good
    for the dynamics model, not merely good for reconstruction.

    The encoder produces a per-cell Gaussian posterior (mu, logvar). Training samples
    z ~ N(mu, sigma^2) (reparameterized) and minimizes reconstruction + beta * KL to N(0, 1).
    The dynamics pipeline uses the deterministic posterior mean mu as 'the latent' (see
    `encode`); the KL keeps mu ~unit-Gaussian per channel, so no external normalization is
    needed downstream. """
    def __init__(self, latent_ch=32):
        super().__init__()
        # Encoder trunk: 64x64 -> 32x32 -> 16x16 (channels 3 -> 32 -> 64).
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Two heads map the 16x16 trunk down to the 8x8 latent grid: mean and log-variance.
        self.to_mu = nn.Conv2d(64, latent_ch, 3, stride=2, padding=1)
        self.to_logvar = nn.Conv2d(64, latent_ch, 3, stride=2, padding=1)

        # Decoder (8x8 -> 16x16 -> 32x32 -> 64x64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode_dist(self, x):
        """Posterior over the latent grid: (mu, logvar), each (B, latent_ch, 8, 8)."""
        h = self.enc(x)
        return self.to_mu(h), self.to_logvar(h)

    def encode(self, x):
        """Deterministic latent for the dynamics pipeline: the posterior mean mu."""
        return self.to_mu(self.enc(x))  # (B, latent_ch, 8, 8)

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


def _gn_groups(ch):
    """Largest GroupNorm group count (<=8) that divides `ch`."""
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class _ResBlock(nn.Module):
    """ Residual block: (conv -> GroupNorm -> ReLU) x2 with a skip connection.
    GroupNorm (not BatchNorm) because the model is rolled out autoregressively on
    its own predictions, where BatchNorm's running stats would drift off-distribution. """
    def __init__(self, ch, ksize=3):
        super().__init__()
        pad = ksize // 2
        g = _gn_groups(ch)
        self.conv1 = nn.Conv2d(ch, ch, ksize, padding=pad)
        self.norm1 = nn.GroupNorm(g, ch)
        self.conv2 = nn.Conv2d(ch, ch, ksize, padding=pad)
        self.norm2 = nn.GroupNorm(g, ch)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class LatentResidualCNN(nn.Module):
    """ Part 2: a deep residual CNN that predicts the *change* in the latent grid.

    Two design choices:

      * Predicts a RESIDUAL delta-z, not the next latent outright. Frame-to-frame the
        latent is ~95% identical (static background, small ball motion), so the model
        only has to learn what moved. The head is zero-initialised, so the model starts
        out as exact persistence (delta-z = 0) and learns corrections from there.

      * Stacks the context frames along channels (no recurrence). On an 8x8 grid a few
        3x3 residual blocks already cover the whole grid, so depth -- not kernel size
        or recurrence -- is what supplies the compute to model bounces and collisions.

    The input latents are the VAE's posterior mean, which the KL keeps ~unit-Gaussian, so
    the dynamics model needs no separate normalization.
    """
    def __init__(self, latent_ch=32, context_len=5, hidden_ch=128, n_blocks=6):
        super().__init__()
        self.latent_ch = latent_ch
        self.context_len = context_len

        self.stem = nn.Conv2d(context_len * latent_ch, hidden_ch, 3, padding=1)
        self.blocks = nn.Sequential(*[_ResBlock(hidden_ch) for _ in range(n_blocks)])
        self.head = nn.Conv2d(hidden_ch, latent_ch, 3, padding=1)
        # Start as the identity (persistence): delta-z = 0 until training moves it.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, latent_seq):
        # latent_seq: (B, T, latent_ch, H, W), with T == context_len.
        B, T, C, H, W = latent_seq.shape
        assert T == self.context_len, f"expected {self.context_len} context frames, got {T}"
        x = latent_seq.reshape(B, T * C, H, W)
        h = self.blocks(self.stem(x))
        return latent_seq[:, -1] + self.head(h)  # residual: z_next = z_last + delta-z


class PixelDynamicsCNN(nn.Module):
    """ A baseline model that attempts to learn physics directly on pixels. """
    def __init__(self, context_len=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(context_len * 3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is (B, Context, C, H, W)
        B, T, C, H, W = x.shape
        # Flatten Time and Channels together: (B, T*C, H, W) -> (B, 5, 64, 64)
        x = x.view(B, T * C, H, W)
        return self.net(x)