"""Step 2b: DPO. First collect preference pairs from the SFT model (one right answer, one wrong answer
to the same problem), then train the model to prefer the right one, relative to a frozen copy of itself."""
import json, os, sys, time, torch
from arith import *

N_PROB, G, STEPS, BATCH, BETA = 3000, 8, 600, 64, 0.1
LR = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-5
NLL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0      # weight of an extra SFT loss on the chosen answer
USE_DPO = (sys.argv[3] != "nodpo") if len(sys.argv) > 3 else True   # "nodpo": keep only the SFT loss (control)
TAG = f"lr{LR:g}_nll{NLL:g}" + ("" if USE_DPO else "_nodpo")
print(f"learning rate {LR:g}, DPO loss {'on' if USE_DPO else 'off'}, extra SFT loss weight {NLL:g}")
policy = load_model("~/.labs_data/sft_math.pt")
ref = load_model("~/.labs_data/sft_math.pt"); ref.eval()          # frozen reference model
for p in ref.parameters(): p.requires_grad = False

# 1. build preference pairs: sample G answers per problem, keep one correct (chosen) and one wrong (rejected)
gen = torch.Generator().manual_seed(0)
probs, chosen, rejected = [], [], []
problems = make_problems(N_PROB, seed=50_000)
for i in range(0, N_PROB, 100):
    chunk = problems[i:i + 100]
    rep = [p for p in chunk for _ in range(G)]
    outs = generate(policy, rep, 1.0, gen).view(len(chunk), G, -1)
    for (a, b), cand in zip(chunk, outs):
        right = [c for c in cand if read_answer(c.tolist()) == a + b]
        wrong = [c for c in cand if read_answer(c.tolist()) != a + b]
        if right and wrong:
            probs.append((a, b)); chosen.append(right[0]); rejected.append(wrong[0])
chosen, rejected = torch.stack(chosen), torch.stack(rejected)
print(f"{N_PROB} problems x {G} samples -> {len(probs)} usable pairs "
      f"(the other problems had all-right or all-wrong samples)")
a, b = probs[0]
print(f"example pair: {a}+{b}=  chosen {tok.decode(chosen[0].tolist())!r}  rejected {tok.decode(rejected[0].tolist())!r}")

def seq_logp(model, idx, answers):
    lp, mask = answer_logprobs(model, [probs[i] for i in idx], answers)
    return (lp * mask).sum(1)

# 2. DPO training
opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0)
g = torch.Generator().manual_seed(1); log, t0 = [], time.time()
with torch.no_grad():
    i0 = list(range(256))
    print(f"start (after SFT): test accuracy {accuracy(policy, TEST):.1%}  "
          f"logp(chosen) {seq_logp(policy, i0, chosen[:256]).mean().item():.2f}  logp(rejected) {seq_logp(policy, i0, rejected[:256]).mean().item():.2f}")
for step in range(1, STEPS + 1):
    idx = torch.randint(len(probs), (BATCH,), generator=g).tolist()
    cw, cl = chosen[idx], rejected[idx]
    with torch.no_grad():
        ref_w, ref_l = seq_logp(ref, idx, cw), seq_logp(ref, idx, cl)
    pol_w, pol_l = seq_logp(policy, idx, cw), seq_logp(policy, idx, cl)
    # how much more the policy likes chosen over rejected, compared with the reference model
    margin = BETA * ((pol_w - ref_w) - (pol_l - ref_l))
    loss = -F.logsigmoid(margin).mean() if USE_DPO else torch.zeros((), device=dev)
    if NLL > 0:
        # also keep the chosen answer likely in absolute terms, not only relative to the rejected one
        loss = loss - NLL * pol_w.mean() / ANS_LEN
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        acc = accuracy(policy, TEST)
        log.append([step, loss.item(), acc, pol_w.mean().item(), pol_l.mean().item()])
        print(f"step {step:3d}  loss {loss.item():.3f}  logp(chosen) {pol_w.mean().item():7.2f}  "
              f"logp(rejected) {pol_l.mean().item():7.2f}  test accuracy (greedy) {acc:.1%}  ({time.time()-t0:.0f}s)", flush=True)
torch.save(policy.state_dict(), os.path.expanduser(f"~/.labs_data/dpo_math_{TAG}.pt"))
json.dump(log, open(os.path.expanduser(f"~/.labs_data/dpo_log_{TAG}.json"), "w"))
