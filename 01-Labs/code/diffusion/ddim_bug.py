"""A bug found while writing this lab: in DDIM, clipping the predicted clean image to [-1, 1]
without recomputing the noise makes long runs drift. Compare three versions of one DDIM step."""
import torch
from diffusion import alpha_bar, T, predict_noise
from classifier import load_classifier
from common import dev, load_model

model, judge = load_model(), load_classifier()
labels = torch.arange(10).repeat_interleave(20).to(dev)      # 20 images of each digit

@torch.no_grad()
def ddim(steps, w, mode):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(len(labels), 1, 32, 32, generator=g).to(dev)
    ts = torch.linspace(T - 1, 0, steps).long().tolist()
    for i, t in enumerate(ts):
        eps = predict_noise(model, x, t, labels, w)
        ab = alpha_bar[t].item()
        x0 = (x - (1 - ab) ** 0.5 * eps) / ab ** 0.5
        if mode != "no clipping":
            x0 = x0.clamp(-1, 1)
        if mode == "clip, then recompute noise":
            eps = (x - ab ** 0.5 * x0) / (1 - ab) ** 0.5     # make the noise consistent with the clipped image
        ab_next = alpha_bar[ts[i + 1]].item() if i + 1 < len(ts) else 1.0
        x = ab_next ** 0.5 * x0 + (1 - ab_next) ** 0.5 * eps
    return (judge(x.clamp(-1, 1)).argmax(1) == labels).float().mean().item()

print(f"{'version':<30}" + "".join(f"{f'{s} steps, w={w:g}':>18}" for s in [10, 50] for w in [1.0, 2.0]))
for mode in ["clip only (buggy)", "no clipping", "clip, then recompute noise"]:
    print(f"{mode:<30}" + "".join(f"{ddim(s, w, mode):>18.1%}" for s in [10, 50] for w in [1.0, 2.0]))
