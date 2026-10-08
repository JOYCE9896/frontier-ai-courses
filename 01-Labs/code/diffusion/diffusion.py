"""The diffusion process: a noise schedule, adding noise (forward), and two ways to remove it (DDPM and DDIM)."""
import torch

T = 1000
betas = torch.linspace(1e-4, 0.02, T)              # how much noise is added at each step
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, 0)               # fraction of the original image's "signal" left after t steps
NULL = 10                                          # label index meaning "no label" (used for guidance)


def add_noise(x0, t, noise):
    """Jump straight to step t: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise."""
    ab = alpha_bar.to(x0.device)[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def predict_noise(model, x, t, y, guidance):
    """Classifier-free guidance: eps = eps_uncond + w * (eps_cond - eps_uncond). w = 0 ignores the label."""
    tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    if guidance == 0:
        return model(x, tt, torch.full_like(y, NULL))
    eps_c = model(x, tt, y)
    eps_u = model(x, tt, torch.full_like(y, NULL))
    return eps_u + guidance * (eps_c - eps_u)


@torch.no_grad()
def ddpm_sample(model, y, guidance=0.0, gen=None, keep_every=None):
    """DDPM: start from pure noise and take all 1000 small denoising steps. Optionally keep snapshots."""
    dev = y.device
    x = torch.randn(len(y), 1, 32, 32, generator=gen).to(dev)
    snaps = []
    for t in reversed(range(T)):
        eps = predict_noise(model, x, t, y, guidance)
        a, ab, b = alphas[t].item(), alpha_bar[t].item(), betas[t].item()
        mean = (x - b / (1 - ab) ** 0.5 * eps) / a ** 0.5      # remove the predicted noise for this step
        if t > 0:
            x = mean + b ** 0.5 * torch.randn(x.shape, generator=gen).to(dev)   # plus a little fresh noise
        else:
            x = mean
        if keep_every and (t % keep_every == 0 or t == T - 1):
            snaps.append((t, x.clamp(-1, 1).cpu()))
    return x.clamp(-1, 1), snaps


@torch.no_grad()
def ddim_sample(model, y, steps=50, guidance=0.0, gen=None):
    """DDIM: same trained model, but jump through only `steps` time steps and add no fresh noise."""
    dev = y.device
    x = torch.randn(len(y), 1, 32, 32, generator=gen).to(dev)
    ts = torch.linspace(T - 1, 0, steps).long().tolist()
    for i, t in enumerate(ts):
        eps = predict_noise(model, x, t, y, guidance)
        ab = alpha_bar[t].item()
        x0 = (x - (1 - ab) ** 0.5 * eps) / ab ** 0.5            # the model's current guess of the clean image
        x0 = x0.clamp(-1, 1)                                     # pixel values must stay in [-1, 1]
        eps = (x - ab ** 0.5 * x0) / (1 - ab) ** 0.5             # recompute the noise so it matches the clipped image
        ab_next = alpha_bar[ts[i + 1]].item() if i + 1 < len(ts) else 1.0
        x = ab_next ** 0.5 * x0 + (1 - ab_next) ** 0.5 * eps     # move to the next (less noisy) time step
    return x.clamp(-1, 1)
