"""Step 2a: GRPO. No answers are shown to the model. It writes several answers per problem,
a checker gives reward 1 (correct) or 0 (wrong), and answers that beat their group's average are reinforced."""
import json, os, time, torch
from arith import *

STEPS, Q, G, LR, TEMP = 600, 32, 8, 2e-5, 1.0
model = load_model("~/.labs_data/sft_math.pt")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
gen = torch.Generator().manual_seed(0)
log, t0 = [], time.time()
print(f"start (after SFT): test accuracy {accuracy(model, TEST):.1%}")
for step in range(1, STEPS + 1):
    # 1. fresh problems every step; each problem is repeated G times
    problems = [p for p in make_problems(Q, seed=10_000 + step) for _ in range(G)]
    # 2. the current model writes one answer per copy (sampling, so the G answers differ)
    model.eval()
    answers = generate(model, problems, TEMP, gen)
    model.train()
    # 3. reward from the checker: 1 if the answer is exactly right, else 0
    reward = torch.tensor([float(read_answer(o) == a + b) for o, (a, b) in zip(answers.tolist(), problems)], device=dev)
    # 4. group-relative advantage: compare each answer with the other answers to the same problem
    r = reward.view(Q, G)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
    # 5. policy gradient: raise the log-probability of answers with positive advantage, lower the others
    lp, mask = answer_logprobs(model, problems, answers)
    seq_lp = (lp * mask).sum(1)
    loss = -(adv * seq_lp).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        acc = accuracy(model, TEST)
        log.append([step, reward.mean().item(), acc])
        print(f"step {step:3d}  mean reward of samples {reward.mean().item():.2f}  test accuracy (greedy) {acc:.1%}  ({time.time()-t0:.0f}s)", flush=True)
torch.save(model.state_dict(), os.path.expanduser("~/.labs_data/grpo_math.pt"))
json.dump(log, open(os.path.expanduser("~/.labs_data/grpo_log.json"), "w"))
