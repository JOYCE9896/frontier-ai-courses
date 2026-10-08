"""A small U-Net that looks at a noisy image, the time step and (optionally) a digit label,
and predicts the noise that was added."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(t, dim):
    """Sinusoidal embedding of the time step, the same idea as positional encoding in Transformers."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection; the time/label embedding is added in between."""
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]       # tell every pixel which time step / digit this is
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """32x32 -> 16x16 -> 8x8 and back up, with skip connections between matching resolutions."""
    def __init__(self, ch=(64, 128, 256), emb_dim=256, n_classes=10):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.label_emb = nn.Embedding(n_classes + 1, emb_dim)   # the extra label (index 10) means "no label"
        self.inp = nn.Conv2d(1, ch[0], 3, padding=1)
        self.down1 = ResBlock(ch[0], ch[0], emb_dim)
        self.down2 = ResBlock(ch[0], ch[1], emb_dim)
        self.down3 = ResBlock(ch[1], ch[2], emb_dim)
        self.mid = ResBlock(ch[2], ch[2], emb_dim)
        self.up3 = ResBlock(ch[2] + ch[2], ch[1], emb_dim)
        self.up2 = ResBlock(ch[1] + ch[1], ch[0], emb_dim)
        self.up1 = ResBlock(ch[0] + ch[0], ch[0], emb_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv2d(ch[0], 1, 3, padding=1))

    def forward(self, x, t, y):
        emb = self.time_mlp(time_embedding(t, self.emb_dim)) + self.label_emb(y)
        h0 = self.inp(x)
        h1 = self.down1(h0, emb)                          # 32x32
        h2 = self.down2(F.avg_pool2d(h1, 2), emb)         # 16x16
        h3 = self.down3(F.avg_pool2d(h2, 2), emb)         # 8x8
        m = self.mid(h3, emb)
        u3 = self.up3(torch.cat([m, h3], 1), emb)                                  # 8x8
        u2 = self.up2(torch.cat([F.interpolate(u3, scale_factor=2), h2], 1), emb)  # 16x16
        u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2), h1], 1), emb)  # 32x32
        return self.out(u1)
