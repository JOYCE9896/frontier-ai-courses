"""Seven decoding settings, each continuing the same opening."""
import torch
from decoding import tok, greedy, sample, beam_search, text

prompt = "Once upon a time, there was a little girl named Mia. She"
ids = tok.encode(prompt).ids
N = 120
methods = {
    "greedy": lambda: greedy(ids, N),
    "beam search (width 5)": lambda: beam_search(ids, N, width=5),
    "pure sampling (temperature 1, no truncation)": lambda: sample(ids, N, gen=torch.Generator().manual_seed(3)),
    "temperature 0.7": lambda: sample(ids, N, temperature=0.7, gen=torch.Generator().manual_seed(3)),
    "temperature 1.8": lambda: sample(ids, N, temperature=1.8, gen=torch.Generator().manual_seed(3)),
    "top-k (k = 40)": lambda: sample(ids, N, top_k=40, gen=torch.Generator().manual_seed(3)),
    "top-p (p = 0.9)": lambda: sample(ids, N, top_p=0.9, gen=torch.Generator().manual_seed(3)),
}
print("opening:", prompt)
for name, fn in methods.items():
    print(f"\n=== {name} ===")
    print(text(fn()))
