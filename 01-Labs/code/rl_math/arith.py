"""Shared helpers: make problems, tokenize them, generate answers in batches, check answers, measure accuracy."""
import os, sys, random
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinyllama"))
from model import Config, TinyLlama

dev = "mps"
tok = Tokenizer.from_file(os.path.expanduser("~/.labs_data/tinystories/prepared/tokenizer.json"))
EOT = tok.token_to_id("<|endoftext|>")
DIGIT = [tok.token_to_id(str(d)) for d in range(10)]
PLUS, EQ = tok.token_to_id("+"), tok.token_to_id("=")
ANS_LEN = 5          # at most 4 answer digits plus one end-of-text token

def make_problems(n, seed):
    """n three-digit addition problems (a, b), with a and b between 100 and 999."""
    rng = random.Random(seed)
    return [(rng.randint(100, 999), rng.randint(100, 999)) for _ in range(n)]

def prompt_ids(a, b):
    """"347+285=" becomes 8 tokens: one token per character."""
    return [DIGIT[int(c)] for c in str(a)] + [PLUS] + [DIGIT[int(c)] for c in str(b)] + [EQ]

def answer_ids(a, b):
    """Tokens of the correct answer, followed by end-of-text, padded to 5 tokens."""
    ids = [DIGIT[int(c)] for c in str(a + b)] + [EOT]
    return ids + [EOT] * (ANS_LEN - len(ids))

def read_answer(ids):
    """Read generated tokens as an integer, stopping at end-of-text. Anything that is not a digit -> None."""
    s = ""
    for t in ids:
        if t == EOT: break
        if t not in DIGIT: return None
        s += str(DIGIT.index(t))
    return int(s) if s else None

def load_model(path):
    m = TinyLlama(Config()).to(dev)
    m.load_state_dict(torch.load(os.path.expanduser(path), map_location=dev))
    return m

@torch.no_grad()
def generate(model, problems, temperature=0.0, gen=None):
    """Generate answers for a batch of problems. temperature 0 = greedy, otherwise sample. Returns (N, 5) tokens."""
    x = torch.tensor([prompt_ids(a, b) for a, b in problems], device=dev)
    for _ in range(ANS_LEN):
        logits = model(x)[0][:, -1, :].float()
        if temperature == 0:
            nxt = logits.argmax(-1)
        else:
            probs = F.softmax(logits / temperature, -1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=gen).squeeze(1).to(dev)
        x = torch.cat([x, nxt[:, None]], 1)
    return x[:, 8:]

def answer_logprobs(model, problems, answers):
    """Log-probability the model assigns to each answer token, shape (N, 5), plus a mask
    that is 1 up to and including the first end-of-text token and 0 after it."""
    x = torch.tensor([prompt_ids(a, b) for a, b in problems], device=dev)
    seq = torch.cat([x, answers], 1)
    logits = model(seq[:, :-1])[0].float()
    lp = F.log_softmax(logits, -1).gather(-1, seq[:, 1:, None]).squeeze(-1)[:, 7:]   # the 5 answer positions
    is_eot = (answers == EOT).int()
    after_first_eot = (torch.cumsum(is_eot, 1) - is_eot) > 0
    return lp, (~after_first_eot).float()

def accuracy(model, problems, temperature=0.0, k=1, seed=0):
    """Generate k answers per problem; a problem counts as solved if any of them is right (pass@k)."""
    model.eval(); gen = torch.Generator().manual_seed(seed)
    ok = torch.zeros(len(problems), dtype=torch.bool)
    for _ in range(k):
        out = generate(model, problems, temperature, gen).tolist()
        ok |= torch.tensor([read_answer(o) == a + b for o, (a, b) in zip(out, problems)])
    model.train()
    return ok.float().mean().item()

TEST = make_problems(500, seed=999)    # fixed 500 test problems used by every method
