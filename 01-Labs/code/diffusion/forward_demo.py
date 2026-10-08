"""The forward process: what a digit looks like after t steps of adding noise."""
import torch
from data import load
from diffusion import add_noise, alpha_bar
from common import save_grid

x, y = load("train")
x0 = x[:6]                                        # six training digits
steps = [0, 50, 100, 200, 400, 600, 800, 999]
torch.manual_seed(0)
noise = torch.randn_like(x0)
cols = [add_noise(x0, torch.full((6,), t), noise) for t in steps]
rows = [torch.stack([c[i] for c in cols]) for i in range(6)]
save_grid(rows, "diffusion_forward.png", col_titles=[f"t={t}" for t in steps])
for t in steps:
    ab = alpha_bar[t].item()
    print(f"t={t:4d}: x_t = {ab**0.5:.3f} * image + {(1-ab)**0.5:.3f} * noise")
