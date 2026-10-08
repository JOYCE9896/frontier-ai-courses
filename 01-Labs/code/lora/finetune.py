"""把练习 01 的小模型适配成"龙的故事"专家：全量微调 vs 不同秩的 LoRA。"""
import os, sys, re, json, time
import numpy as np
import torch
import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinyllama"))
from model import Config, TinyLlama
from lora import apply_lora

dev = "mps"
D = os.path.expanduser("~/.labs_data/tinystories")
tok = Tokenizer.from_file(os.path.join(D, "prepared", "tokenizer.json"))
EOT = tok.token_to_id("<|endoftext|>")
load = lambda pre: pq.read_table(os.path.join(D, [x for x in os.listdir(D) if x.startswith(pre)][0])).column("text").to_pylist()
is_dragon = lambda s: re.search(r"\bdragon", s, re.I) is not None

def to_ids(texts):
    ids = []
    for e in tok.encode_batch(texts):
        ids.extend(e.ids); ids.append(EOT)
    return np.array(ids, dtype=np.int64)

train_stories = load("train-00000")
val_stories = load("validation")
dragon_train = to_ids([s for s in train_stories if is_dragon(s)])          # 微调数据：训练集里所有提到龙的故事
dragon_val = to_ids([s for s in val_stories if is_dragon(s)])              # 没见过的龙故事：看学得好不好
general_val = to_ids([s for s in val_stories if not is_dragon(s)][:3000])  # 没见过的普通故事：看有没有忘掉原来的本事

BATCH, SEQ, STEPS = 16, 256, 600

def batch(data, gen):
    ix = torch.randint(len(data) - SEQ - 1, (BATCH,), generator=gen).tolist()
    x = torch.from_numpy(np.stack([data[i:i+SEQ] for i in ix]))
    y = torch.from_numpy(np.stack([data[i+1:i+1+SEQ] for i in ix]))
    return x.to(dev), y.to(dev)

@torch.no_grad()
def evaluate(model, data, n=15):
    model.eval(); g = torch.Generator().manual_seed(0)
    l = sum(model(*batch(data, g))[1].item() for _ in range(n)) / n
    model.train(); return l

@torch.no_grad()
def dragon_rate(model, n=40):
    """以"Once upon a time"开头写 n 个故事，数有几个提到了龙。"""
    model.eval(); torch.manual_seed(123); hits, first = 0, None
    for i in range(n):
        idx = torch.tensor([tok.encode("Once upon a time").ids], device=dev)
        out = model.generate(idx, 150, temperature=0.8, top_k=50, eot_id=EOT)[0].tolist()
        s = tok.decode([t for t in out if t != EOT])
        hits += is_dragon(s); first = first or s
    model.train(); return hits / n, first

def fresh():
    m = TinyLlama(Config()).to(dev)
    m.load_state_dict(torch.load(os.path.expanduser("~/.labs_data/tinyllama_run/model.pt"), map_location=dev))
    return m

if __name__ == "__main__":
    print(f"微调数据：{len(dragon_train):,} 个 token（训练集中提到 dragon 的故事）")
    print(f"评估数据：龙故事 {len(dragon_val):,} 个 token，普通故事 {len(general_val):,} 个 token（都来自验证集，训练时没见过）\n")
    results = {}
    base = fresh()
    rate, s = dragon_rate(base)
    results["原模型（不微调）"] = dict(trainable=0, dragon=evaluate(base, dragon_val), general=evaluate(base, general_val), rate=rate, minutes=0, sample=s, curve=[])
    print("原模型：", {k: v for k, v in results["原模型（不微调）"].items() if k not in ("sample", "curve")}, flush=True)

    runs = [("全量微调", None, 3e-4), ("LoRA r=1", 1, 2e-3), ("LoRA r=4", 4, 2e-3), ("LoRA r=16", 16, 2e-3), ("LoRA r=64", 64, 2e-3)]
    for name, r, lr in runs:
        torch.manual_seed(0)
        model = fresh()
        if r is not None:
            apply_lora(model, r).to(dev)   # 新加的 A、B 矩阵也要搬到 GPU 上
        params = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        g = torch.Generator().manual_seed(1); t0 = time.time(); curve = []
        for step in range(STEPS + 1):
            if step % 100 == 0:
                curve.append([step, evaluate(model, dragon_val), evaluate(model, general_val)])
                print(f"  {name} step {step:3d}  龙故事损失 {curve[-1][1]:.3f}  普通故事损失 {curve[-1][2]:.3f}", flush=True)
            if step == STEPS: break
            _, loss = model(*batch(dragon_train, g))
            opt.zero_grad(); loss.backward(); opt.step()
        minutes = (time.time() - t0) / 60
        rate, s = dragon_rate(model)
        results[name] = dict(trainable=n_train, dragon=curve[-1][1], general=curve[-1][2], rate=rate, minutes=minutes, sample=s, curve=curve)
        print(f"{name}: 可训练参数 {n_train:,}，龙故事损失 {curve[-1][1]:.3f}，普通故事损失 {curve[-1][2]:.3f}，"
              f"写到龙的比例 {rate:.0%}，用时 {minutes:.1f} 分钟\n", flush=True)
    json.dump(results, open(os.path.expanduser("~/.labs_data/lora_results.json"), "w"), ensure_ascii=False, indent=1)
