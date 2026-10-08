"""Control: instead of GRPO or DPO, just keep doing SFT on the same 2000 labeled problems."""
import json, os, time, torch
from arith import *

STEPS, BATCH, LR = 600, 64, 3e-4
train = make_problems(2000, seed=1)                      # the same labeled problems as sft.py
model = load_model("~/.labs_data/sft_math.pt")
opt = torch.optim.AdamW(model.parameters(), lr=LR)
g = torch.Generator().manual_seed(7); t0 = time.time(); log = []
print(f"start (after SFT): test accuracy {accuracy(model, TEST):.1%}")
for step in range(1, STEPS + 1):
    batch = [train[i] for i in torch.randint(2000, (BATCH,), generator=g).tolist()]
    seq = torch.tensor([prompt_ids(a, b) + answer_ids(a, b) for a, b in batch], device=dev)
    logits = model(seq[:, :-1])[0]
    loss = F.cross_entropy(logits[:, 7:].reshape(-1, logits.size(-1)), seq[:, 8:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        acc = accuracy(model, TEST); log.append([step, loss.item(), acc])
        print(f"step {step:3d}  train loss {loss.item():.3f}  test accuracy (greedy) {acc:.1%}  ({time.time()-t0:.0f}s)", flush=True)
torch.save(model.state_dict(), os.path.expanduser("~/.labs_data/sftmore_math.pt"))
json.dump(log, open(os.path.expanduser("~/.labs_data/sftmore_log.json"), "w"))
