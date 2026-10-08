"""几种解码方法：贪心、束搜索、纯采样、温度、top-k、top-p。都用练习 01 训练好的小 LLaMA。"""
import os, sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinyllama"))
from model import Config, TinyLlama

DATA = os.path.expanduser("~/.labs_data/tinystories/prepared")
dev = "mps"
tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
EOT = tok.token_to_id("<|endoftext|>")
model = TinyLlama(Config()).to(dev)
model.load_state_dict(torch.load(os.path.expanduser("~/.labs_data/tinyllama_run/model.pt"), map_location=dev))
model.eval()


@torch.no_grad()
def next_logprobs(ids):
    """给一串 token，返回模型对下一个 token 的对数概率（4096 个数）。"""
    x = torch.tensor([ids[-Config.max_seq_len:]], device=dev)
    return F.log_softmax(model(x)[0][0, -1].float(), dim=-1)


def greedy(ids, n):
    """贪心：每一步都选概率最高的 token。"""
    ids = list(ids)
    for _ in range(n):
        nxt = next_logprobs(ids).argmax().item()
        ids.append(nxt)
        if nxt == EOT: break
    return ids


def sample(ids, n, temperature=1.0, top_k=None, top_p=None, gen=None):
    """按概率抽签。temperature 调整分布的平坦程度，top_k / top_p 先砍掉概率低的尾巴再抽。"""
    ids = list(ids)
    for _ in range(n):
        logits = next_logprobs(ids) / temperature
        probs = F.softmax(logits, dim=-1)
        if top_k is not None:
            # 只留概率最高的 k 个，其余设为 0
            kth = torch.topk(probs, top_k).values[-1]
            probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
        if top_p is not None:
            # 从高到低排序，留下累计概率刚好达到 p 的那些 token
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, 0)
            keep = cum - sp < top_p          # 加上它之前累计还不到 p，就保留它
            probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep])
        probs = probs / probs.sum()           # 砍掉尾巴后重新归一化，让概率加起来等于 1
        nxt = torch.multinomial(probs.cpu(), 1, generator=gen).item()
        ids.append(nxt)
        if nxt == EOT: break
    return ids


def beam_search(ids, n, width=5):
    """束搜索：同时保留得分最高的 width 条候选，每步把每条都往后延伸一个 token，再留下最好的 width 条。
    得分是整条续写的对数概率之和。"""
    beams = [(0.0, list(ids))]               # (累计对数概率, token 序列)
    finished = []
    for _ in range(n):
        cands = []
        for score, seq in beams:
            lp = next_logprobs(seq)
            top = torch.topk(lp, width)
            for v, i in zip(top.values.tolist(), top.indices.tolist()):
                cands.append((score + v, seq + [i]))
        cands.sort(key=lambda c: c[0], reverse=True)
        beams = []
        for score, seq in cands:
            if seq[-1] == EOT:
                finished.append((score, seq))
            else:
                beams.append((score, seq))
            if len(beams) == width: break
        if len(finished) >= width: break
    pool = finished + beams
    return max(pool, key=lambda c: c[0])[1]


def logprob_of(ids, start):
    """模型给 ids[start:] 这段续写的平均每 token 对数概率（越接近 0，模型越觉得它'正常'）。"""
    x = torch.tensor([ids[-Config.max_seq_len:]], device=dev)
    with torch.no_grad():
        lp = F.log_softmax(model(x)[0][0].float(), dim=-1)
    off = len(ids) - x.shape[1]
    vals = [lp[t - 1 - off, ids[t]].item() for t in range(max(start, off + 1), len(ids))]
    return sum(vals) / len(vals)


def text(ids):
    return tok.decode([t for t in ids if t != EOT])
