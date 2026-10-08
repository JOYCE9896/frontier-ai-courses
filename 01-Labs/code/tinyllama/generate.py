"""用训练好的模型：看它对下一个 token 的概率分布，再用不同的开头写故事。"""
import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import Config, TinyLlama

DATA = os.path.expanduser("~/.labs_data/tinystories/prepared")
RUN = os.path.expanduser("~/.labs_data/tinyllama_run")
dev = "mps"
tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
eot = tok.token_to_id("<|endoftext|>")
model = TinyLlama(Config()).to(dev)
model.load_state_dict(torch.load(os.path.join(RUN, "model.pt"), map_location=dev))
model.eval()

# 1. 模型对下一个 token 的猜测：取概率最高的 5 个
print("=== 下一个 token 概率最高的 5 个 ===")
for text in ["Once upon a", "Lily went to the park with her", "The cat was very", "Tom was sad because he lost his"]:
    ids = torch.tensor([tok.encode(text).ids], device=dev)
    with torch.no_grad():
        probs = F.softmax(model(ids)[0][0, -1], dim=-1)
    top = torch.topk(probs, 5)
    guesses = "，".join(f"{tok.decode([i])!r} {p:.1%}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
    print(f"{text} ___  ->  {guesses}")

# 2. 用不同的开头写故事
def write(prompt, temperature, seed=0):
    torch.manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt).ids], device=dev)
    out = model.generate(idx, 200, temperature=temperature, top_k=50, eot_id=eot)[0].tolist()
    return tok.decode([t for t in out if t != eot])

for prompt in ["Once upon a time, there was a little robot", "Sara and her dog went to the beach.", "The old tree in the forest"]:
    print(f"\n=== 开头：{prompt}（温度 0.8）===")
    print(write(prompt, 0.8))

print("\n=== 同一个开头，不同温度 ===")
for t in [0, 0.5, 1.0, 1.5]:
    print(f"\n--- 温度 {t}{'（贪心：每次都选概率最高的）' if t == 0 else ''} ---")
    print(write("One day, Ben found a box", t, seed=1))
