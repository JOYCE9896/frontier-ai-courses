"""Use the trained model: watch the reverse process, compare DDPM with DDIM, and try classifier-free guidance."""
import time
import torch
from diffusion import ddpm_sample, ddim_sample
from classifier import load_classifier
from common import dev, save_grid, load_model

model, judge = load_model(), load_classifier()

def accuracy(imgs, labels):
    """Fraction of generated images that the classifier reads as the requested digit."""
    with torch.no_grad():
        return (judge(imgs.to(dev)).argmax(1) == labels.to(dev)).float().mean().item()

def diversity(imgs, labels):
    """Average pixel distance between two images of the same requested digit (higher = more varied)."""
    d = []
    for k in range(10):
        g = imgs[labels.cpu() == k].flatten(1)
        d.append(torch.cdist(g, g).sum() / (len(g) * (len(g) - 1)))
    return torch.stack(d).mean().item()

# 1. The reverse process, step by step (no label: guidance 0)
print("=== 1. reverse process (DDPM, 1000 steps, no label) ===")
y = torch.full((8,), 0, device=dev)
t0 = time.time()
_, snaps = ddpm_sample(model, y, guidance=0.0, gen=torch.Generator().manual_seed(3), keep_every=100)
print(f"8 images, 1000 steps: {time.time()-t0:.1f}s")
rows = [torch.stack([s[1][i] for s in snaps]) for i in range(8)]
save_grid(rows, "diffusion_reverse.png", col_titles=[f"t={s[0]}" for s in snaps])

# 2. 100 unconditional samples: does the model produce all ten digits?
print("\n=== 2. 100 samples without a label ===")
t0 = time.time()
imgs, _ = ddpm_sample(model, torch.zeros(100, dtype=torch.long, device=dev), guidance=0.0, gen=torch.Generator().manual_seed(4))
print(f"100 images with DDPM (1000 steps): {time.time()-t0:.1f}s")
save_grid([imgs[i * 10:(i + 1) * 10].cpu() for i in range(10)], "diffusion_uncond.png", scale=0.7)
with torch.no_grad():
    pred = judge(imgs).argmax(1).cpu()
print("how the classifier reads them:", {d: int((pred == d).sum()) for d in range(10)})

# 3. DDIM: fewer steps, same model
print("\n=== 3. DDPM vs DDIM with fewer steps (guidance 2, 50 images per digit) ===")
labels = torch.arange(10).repeat_interleave(50).to(dev)
show = torch.arange(10, device=dev)
rows, names = [], []
for name, fn in [("DDIM 5 steps", lambda l, g: ddim_sample(model, l, 5, 2.0, g)),
                 ("DDIM 10 steps", lambda l, g: ddim_sample(model, l, 10, 2.0, g)),
                 ("DDIM 25 steps", lambda l, g: ddim_sample(model, l, 25, 2.0, g)),
                 ("DDIM 50 steps", lambda l, g: ddim_sample(model, l, 50, 2.0, g)),
                 ("DDPM 1000 steps", lambda l, g: ddpm_sample(model, l, 2.0, g)[0])]:
    t0 = time.time()
    imgs = fn(labels, torch.Generator().manual_seed(5))
    dt = time.time() - t0
    print(f"{name:<16} {dt:6.1f}s for 500 images   classifier agrees with the label: {accuracy(imgs, labels):.1%}")
    rows.append(fn(show, torch.Generator().manual_seed(6)).cpu()); names.append(name)
save_grid(rows, "diffusion_steps.png", row_titles=names, col_titles=[str(d) for d in range(10)])

# 4. Classifier-free guidance: how strongly to follow the label
print("\n=== 4. classifier-free guidance (DDIM 50 steps, 50 images per digit) ===")
rows, names = [], []
for w in [0.0, 1.0, 2.0, 4.0, 8.0]:
    imgs = ddim_sample(model, labels, 50, w, torch.Generator().manual_seed(8))
    print(f"guidance w={w:<4} classifier agrees with the label: {accuracy(imgs, labels):6.1%}   "
          f"diversity (avg distance between same-digit images): {diversity(imgs.cpu(), labels):.2f}")
    three = labels == 3
    rows.append(imgs[three][:10].cpu()); names.append(f"w = {w:g}")
save_grid(rows, "diffusion_guidance.png", row_titles=names)
