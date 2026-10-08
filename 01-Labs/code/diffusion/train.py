"""Train the U-Net to predict the noise in noisy digits. 10% of the time the label is hidden,
so the same model also learns to generate without a label (needed for classifier-free guidance)."""
import copy, json, os, time
import torch
import torch.nn.functional as F
from data import load
from diffusion import T, NULL, add_noise, ddim_sample
from unet import UNet
from common import dev, RUN, save_grid

STEPS, BATCH, LR, P_DROP, EMA = 6000, 128, 2e-4, 0.1, 0.999
SNAP_AT = [250, 1000, 2500, 6000]
torch.manual_seed(0)
x_all, y_all = load("train")
model = UNet().to(dev)
ema = copy.deepcopy(model).eval()               # a slowly updated average of the weights, used for sampling
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")

log, snaps, t0 = [], [], time.time()
gen = torch.Generator().manual_seed(1)
for step in range(1, STEPS + 1):
    idx = torch.randint(len(x_all), (BATCH,), generator=gen)
    x0, y = x_all[idx].to(dev), y_all[idx].to(dev)
    y = torch.where(torch.rand(BATCH, device=dev) < P_DROP, torch.full_like(y, NULL), y)   # hide some labels
    t = torch.randint(0, T, (BATCH,), device=dev)        # a random time step for each image
    noise = torch.randn_like(x0)
    xt = add_noise(x0, t, noise)
    loss = F.mse_loss(model(xt, t, y), noise)           # how far the predicted noise is from the real noise
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(EMA).add_(pm, alpha=1 - EMA)
    if step % 100 == 0:
        log.append([step, loss.item()])
    if step % 500 == 0:
        print(f"step {step:5d}  loss {sum(l for _, l in log[-5:]) / 5:.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    if step in SNAP_AT:
        torch.save(ema.state_dict(), os.path.join(RUN, f"ema_step{step}.pt"))
        labels = torch.arange(10, device=dev)
        snaps.append(ddim_sample(ema, labels, steps=50, guidance=2.0, gen=torch.Generator().manual_seed(7)).cpu())
        save_grid(snaps, "diffusion_progress.png", row_titles=[f"step {s}" for s in SNAP_AT[:len(snaps)]],
                  col_titles=[str(d) for d in range(10)])
torch.save(model.state_dict(), os.path.join(RUN, "model.pt"))
torch.save(ema.state_dict(), os.path.join(RUN, "ema.pt"))
json.dump(log, open(os.path.join(RUN, "loss.json"), "w"))
print(f"done: {STEPS} steps in {(time.time()-t0)/60:.1f} min")
