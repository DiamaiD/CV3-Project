import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    """ Part 1: Compresses frames into a SPATIAL latent grid (latent_ch x 8 x 8)."""
    def __init__(self, latent_ch=32):
        super().__init__()
        # Encoder (64x64 -> 32x32 -> 16x16 -> 8x8), output is a (latent_ch, 8, 8) grid.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_ch, 3, stride=2, padding=1)
        )

        # Decoder (8x8 -> 16x16 -> 32x32 -> 64x64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)  # (B, latent_ch, 8, 8)

    def decode(self, z):
        return self.decoder(z)  # (B, 3, 64, 64)

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvGRUCell(nn.Module):
    """ A single ConvGRU step: a GRU where the linear gates are replaced by
    convolutions, so the hidden state stays a spatial feature map. """
    def __init__(self, in_ch, hidden_ch, ksize=3):
        super().__init__()
        pad = ksize // 2
        self.hidden_ch = hidden_ch
        # Update (z) and reset (r) gates computed together for efficiency.
        self.conv_zr = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, ksize, padding=pad)
        # Candidate state (n).
        self.conv_n = nn.Conv2d(in_ch + hidden_ch, hidden_ch, ksize, padding=pad)

    def forward(self, x, h):
        zr = self.conv_zr(torch.cat([x, h], dim=1))
        z, r = torch.chunk(zr, 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        return (1 - z) * n + z * h


class LatentDynamicsConvGRU(nn.Module):
    """ Part 2: Learns physics directly on the spatial latent grid."""
    def __init__(self, latent_ch=32, hidden_ch=64):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.cell = ConvGRUCell(latent_ch, hidden_ch)
        self.predictor = nn.Conv2d(hidden_ch, latent_ch, 3, padding=1)

    def forward(self, latent_seq):
        # latent_seq shape: (Batch, Time, latent_ch, H, W)
        B, T, _, H, W = latent_seq.shape
        h = torch.zeros(B, self.hidden_ch, H, W, device=latent_seq.device)
        for t in range(T):
            h = self.cell(latent_seq[:, t], h)

        return self.predictor(h)  # (B, latent_ch, H, W)


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