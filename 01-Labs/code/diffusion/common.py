"""Small helpers shared by the scripts: device, image grids, loading the trained model."""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unet import UNet

dev = "mps"
IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "img")
RUN = os.path.expanduser("~/.labs_data/diffusion_run")
os.makedirs(RUN, exist_ok=True)

def save_grid(rows, path, row_titles=None, col_titles=None, scale=0.9):
    """rows: list of tensors (n, 1, 32, 32) in [-1, 1]; each tensor becomes one row of the picture."""
    nr, nc = len(rows), max(len(r) for r in rows)
    lw = 1.6 if row_titles else 0
    fig, axes = plt.subplots(nr, nc, figsize=(nc * scale + lw, nr * scale + (0.3 if col_titles else 0)), dpi=120, squeeze=False)
    for i, r in enumerate(rows):
        for j in range(nc):
            ax = axes[i][j]; ax.axis("off")
            if j < len(r):
                ax.imshow(r[j, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
            if col_titles and i == 0:
                ax.set_title(col_titles[j], fontsize=8)
        if row_titles:
            axes[i][0].text(-4, 16, row_titles[i], ha="right", va="center", fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(IMG, path)); plt.close(fig)

def load_model(name="ema.pt"):
    m = UNet().to(dev)
    m.load_state_dict(torch.load(os.path.join(RUN, name), map_location=dev))
    return m.eval()
