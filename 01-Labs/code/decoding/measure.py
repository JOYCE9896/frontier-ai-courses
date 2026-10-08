"""Continue 40 openings by 100 tokens with each method and measure three numbers."""
import os, time
import torch
import pyarrow.parquet as pq
from decoding import tok, greedy, sample, beam_search, logprob_of, next_logprobs, EOT

DATA = os.path.expanduser("~/.labs_data/tinystories")
f = [x for x in os.listdir(DATA) if x.startswith("validation")][0]
stories = pq.read_table(os.path.join(DATA, f)).column("text").to_pylist()
prompts = [tok.encode(s).ids[:12] for s in stories[:40]]   # openings: first 12 tokens of 40 unseen stories
N = 100

def ngrams(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]

def repeat_rate(cont):
    """Fraction of 4-grams (4 consecutive tokens) in the continuation that already appeared earlier in it."""
    grams = ngrams(cont, 4)
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen; seen.add(g)
    return rep / max(len(grams), 1)

methods = {
    "greedy": lambda p, s: greedy(p, N),
    "beam search (width 5)": lambda p, s: beam_search(p, N, 5),
    "pure sampling": lambda p, s: sample(p, N, gen=torch.Generator().manual_seed(s)),
    "temperature 0.7": lambda p, s: sample(p, N, temperature=0.7, gen=torch.Generator().manual_seed(s)),
    "temperature 1.8": lambda p, s: sample(p, N, temperature=1.8, gen=torch.Generator().manual_seed(s)),
    "top-k (k = 40)": lambda p, s: sample(p, N, top_k=40, gen=torch.Generator().manual_seed(s)),
    "top-p (p = 0.9)": lambda p, s: sample(p, N, top_p=0.9, gen=torch.Generator().manual_seed(s)),
}
print(f"{'method':<24}{'repeat rate':>12}{'distinct bigrams':>18}{'avg log-prob':>14}{'time':>7}")
for name, fn in methods.items():
    t0 = time.time()
    reps, lps, bigrams, total = [], [], set(), 0
    for i, p in enumerate(prompts):
        out = fn(p, i)
        cont = [t for t in out[len(p):] if t != EOT]
        reps.append(repeat_rate(cont))
        lps.append(logprob_of(out, len(p)))
        bg = ngrams(cont, 2); bigrams.update(bg); total += len(bg)
    print(f"{name:<24}{sum(reps)/len(reps):>12.1%}{len(bigrams)/total:>18.1%}{sum(lps)/len(lps):>14.2f}{time.time()-t0:>6.0f}s", flush=True)

# How often does pure sampling draw a token from the tail?
print("\npure sampling: rank of the drawn token among all 4096 tokens")
ranks = []
for i, p in enumerate(prompts):
    ids = list(p); g = torch.Generator().manual_seed(i)
    for _ in range(N):
        lp = next_logprobs(ids)
        nxt = torch.multinomial(lp.exp().cpu(), 1, generator=g).item()
        ranks.append((lp > lp[nxt]).sum().item() + 1)   # rank = number of more likely tokens + 1
        ids.append(nxt)
        if nxt == EOT: break
n = len(ranks)
print(f"  {n} steps in total")
for k in [1, 5, 40, 100, 1000]:
    print(f"  within the top {k:>4}: {sum(r <= k for r in ranks)/n:6.1%}")
print(f"  ranked below 40: {sum(r > 40 for r in ranks)} steps, {100*sum(r > 40 for r in ranks)/n:.1f} per 100 tokens")
