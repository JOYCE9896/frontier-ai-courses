"""Step 1: supervised fine-tuning (SFT). Show the model problems with correct answers."""
import json, os, sys, time, torch
from arith import *

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 450
BATCH, LR = 64, 3e-4
train = make_problems(N_TRAIN, seed=1)
model = load_model("~/.labs_data/tinyllama_run/model.pt")
print(f"Story-only model, accuracy on 500 test problems: {accuracy(model, TEST):.1%}")
for (a, b), o in zip(TEST[:3], generate(model, TEST[:3]).tolist()):
    print(f"  {a}+{b}=  model wrote {tok.decode(o)!r}, correct answer {a+b}")
print(f"SFT on {N_TRAIN} labeled problems for {STEPS} steps (batch {BATCH})")

opt = torch.optim.AdamW(model.parameters(), lr=LR)
g = torch.Generator().manual_seed(0)
log, t0 = [], time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(N_TRAIN, (BATCH,), generator=g).tolist()
    batch = [train[i] for i in idx]
    seq = torch.tensor([prompt_ids(a, b) + answer_ids(a, b) for a, b in batch], device=dev)
    logits = model(seq[:, :-1])[0]
    # loss only on the answer tokens: the question is given, the model does not need to learn to write it
    loss = F.cross_entropy(logits[:, 7:].reshape(-1, logits.size(-1)), seq[:, 8:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % (STEPS // 6) == 0:
        acc = accuracy(model, TEST)
        log.append([step, loss.item(), acc])
        print(f"step {step:4d}  loss {loss.item():.3f}  test accuracy {acc:.1%}  ({time.time()-t0:.0f}s)", flush=True)
torch.save(model.state_dict(), os.path.expanduser("~/.labs_data/sft_math.pt"))
json.dump(log, open(os.path.expanduser("~/.labs_data/sft_log.json"), "w"))
