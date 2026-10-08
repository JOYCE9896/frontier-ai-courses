"""Decoding methods: greedy, beam search, pure sampling, temperature, top-k, top-p. All use the small LLaMA trained in lab 01."""
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
    """Given a list of tokens, return the model's log-probabilities for the next token (4096 numbers)."""
    x = torch.tensor([ids[-Config.max_seq_len:]], device=dev)
    return F.log_softmax(model(x)[0][0, -1].float(), dim=-1)


def greedy(ids, n):
    """Greedy: always pick the most likely token."""
    ids = list(ids)
    for _ in range(n):
        nxt = next_logprobs(ids).argmax().item()
        ids.append(nxt)
        if nxt == EOT: break
    return ids


def sample(ids, n, temperature=1.0, top_k=None, top_p=None, gen=None):
    """Draw a token according to the probabilities. temperature flattens or sharpens the distribution;
    top_k / top_p cut off the low-probability tail before drawing."""
    ids = list(ids)
    for _ in range(n):
        logits = next_logprobs(ids) / temperature
        probs = F.softmax(logits, dim=-1)
        if top_k is not None:
            # keep only the k most likely tokens, set the rest to 0
            kth = torch.topk(probs, top_k).values[-1]
            probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
        if top_p is not None:
            # sort from most to least likely and keep tokens until the running total reaches p
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, 0)
            keep = cum - sp < top_p          # keep a token if the total before it is still below p
            probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep])
        probs = probs / probs.sum()           # renormalize after cutting the tail so the probabilities sum to 1
        nxt = torch.multinomial(probs.cpu(), 1, generator=gen).item()
        ids.append(nxt)
        if nxt == EOT: break
    return ids


def beam_search(ids, n, width=5):
    """Beam search: keep the width best candidates; at each step extend every candidate by one token and keep
    the width best again. A candidate's score is the sum of the log-probabilities of its tokens."""
    beams = [(0.0, list(ids))]               # (total log-probability, token list)
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
    """Average per-token log-probability the model gives to ids[start:] (closer to 0 = more "normal" to the model)."""
    x = torch.tensor([ids[-Config.max_seq_len:]], device=dev)
    with torch.no_grad():
        lp = F.log_softmax(model(x)[0][0].float(), dim=-1)
    off = len(ids) - x.shape[1]
    vals = [lp[t - 1 - off, ids[t]].item() for t in range(max(start, off + 1), len(ids))]
    return sum(vals) / len(vals)


def text(ids):
    return tok.decode([t for t in ids if t != EOT])
