"""同一个开头，七种解码方法各写一段。"""
import torch
from decoding import tok, greedy, sample, beam_search, text

prompt = "Once upon a time, there was a little girl named Mia. She"
ids = tok.encode(prompt).ids
N = 120
methods = {
    "贪心": lambda: greedy(ids, N),
    "束搜索（宽度 5）": lambda: beam_search(ids, N, width=5),
    "纯采样（温度 1，不砍尾巴）": lambda: sample(ids, N, gen=torch.Generator().manual_seed(3)),
    "温度 0.7": lambda: sample(ids, N, temperature=0.7, gen=torch.Generator().manual_seed(3)),
    "温度 1.8": lambda: sample(ids, N, temperature=1.8, gen=torch.Generator().manual_seed(3)),
    "top-k（k = 40）": lambda: sample(ids, N, top_k=40, gen=torch.Generator().manual_seed(3)),
    "top-p（p = 0.9）": lambda: sample(ids, N, top_p=0.9, gen=torch.Generator().manual_seed(3)),
}
print("开头：", prompt)
for name, fn in methods.items():
    print(f"\n=== {name} ===")
    print(text(fn()))
