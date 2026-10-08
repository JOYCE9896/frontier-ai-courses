"""在 40 个开头上，每种方法各续写 100 个 token，量三个指标。"""
import os, time
import torch
import pyarrow.parquet as pq
from decoding import tok, greedy, sample, beam_search, logprob_of, next_logprobs, EOT

DATA = os.path.expanduser("~/.labs_data/tinystories")
f = [x for x in os.listdir(DATA) if x.startswith("validation")][0]
stories = pq.read_table(os.path.join(DATA, f)).column("text").to_pylist()
prompts = [tok.encode(s).ids[:12] for s in stories[:40]]   # 取 40 篇没见过的故事的前 12 个 token 当开头
N = 100

def ngrams(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]

def repeat_rate(cont):
    """续写里的 4 元组（连续 4 个 token），有多少是前面已经出现过的。"""
    grams = ngrams(cont, 4)
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen; seen.add(g)
    return rep / max(len(grams), 1)

methods = {
    "贪心": lambda p, s: greedy(p, N),
    "束搜索（宽度 5）": lambda p, s: beam_search(p, N, 5),
    "纯采样": lambda p, s: sample(p, N, gen=torch.Generator().manual_seed(s)),
    "温度 0.7": lambda p, s: sample(p, N, temperature=0.7, gen=torch.Generator().manual_seed(s)),
    "温度 1.8": lambda p, s: sample(p, N, temperature=1.8, gen=torch.Generator().manual_seed(s)),
    "top-k（k = 40）": lambda p, s: sample(p, N, top_k=40, gen=torch.Generator().manual_seed(s)),
    "top-p（p = 0.9）": lambda p, s: sample(p, N, top_p=0.9, gen=torch.Generator().manual_seed(s)),
}
print(f"{'方法':<16}{'重复率':>8}{'不同二元组比例':>14}{'平均对数概率':>12}{'用时':>8}")
for name, fn in methods.items():
    t0 = time.time()
    reps, lps, bigrams, total = [], [], set(), 0
    for i, p in enumerate(prompts):
        out = fn(p, i)
        cont = [t for t in out[len(p):] if t != EOT]
        reps.append(repeat_rate(cont))
        lps.append(logprob_of(out, len(p)))
        bg = ngrams(cont, 2); bigrams.update(bg); total += len(bg)
    print(f"{name:<16}{sum(reps)/len(reps):>9.1%}{len(bigrams)/total:>14.1%}{sum(lps)/len(lps):>14.2f}{time.time()-t0:>7.0f}秒", flush=True)

# 纯采样到底多常抽到"尾巴"里的 token？
print("\n纯采样时，被抽中的 token 在模型的排名：")
ranks = []
for i, p in enumerate(prompts):
    ids = list(p); g = torch.Generator().manual_seed(i)
    for _ in range(N):
        lp = next_logprobs(ids)
        nxt = torch.multinomial(lp.exp().cpu(), 1, generator=g).item()
        ranks.append((lp > lp[nxt]).sum().item() + 1)   # 比它概率高的有几个，加 1 就是它的名次
        ids.append(nxt)
        if nxt == EOT: break
n = len(ranks)
print(f"  一共 {n} 步")
for k in [1, 5, 40, 100, 1000]:
    print(f"  排在前 {k:>4} 名以内：{sum(r <= k for r in ranks)/n:6.1%}")
print(f"  排在第 40 名以后：{sum(r > 40 for r in ranks)} 步，平均每 100 个 token 有 {100*sum(r > 40 for r in ranks)/n:.1f} 个")
