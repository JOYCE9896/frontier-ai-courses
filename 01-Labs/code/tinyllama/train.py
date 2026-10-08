"""训练小 LLaMA：随机取一段段文本，让模型预测每个位置的下一个 token。"""
import os, json, math, time
import numpy as np
import torch
from tokenizers import Tokenizer
from model import Config, TinyLlama

DATA = os.path.expanduser("~/.labs_data/tinystories/prepared")
OUT = os.path.expanduser("~/.labs_data/tinyllama_run")
os.makedirs(OUT, exist_ok=True)

# 训练设置
BATCH, SEQ = 32, 256
STEPS, WARMUP = 4000, 200
LR_MAX, LR_MIN = 1e-3, 1e-4
EVAL_EVERY, EVAL_BATCHES = 250, 20
SAMPLE_AT = {0, 100, 300, 1000, 2000, 4000}
PROMPT = "Once upon a time"
dev = "mps"

torch.manual_seed(1337)
tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
eot = tok.token_to_id("<|endoftext|>")
train_data = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")

def get_batch(data, gen):
    # 随机选 BATCH 个起点，每个取 SEQ+1 个 token；输入是前 SEQ 个，目标是往后错一位的 SEQ 个
    ix = torch.randint(len(data) - SEQ - 1, (BATCH,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i+SEQ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+SEQ].astype(np.int64)) for i in ix])
    return x.to(dev), y.to(dev)

def lr_at(step):
    # 先线性升温，再按余弦曲线从 LR_MAX 降到 LR_MIN
    if step < WARMUP:
        return LR_MAX * (step + 1) / WARMUP
    p = (step - WARMUP) / (STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * p))

@torch.no_grad()
def evaluate(model):
    model.eval()
    gen = torch.Generator().manual_seed(0)   # 每次用同样的验证批次，结果可比
    losses = [model(*get_batch(val_data, gen))[1].item() for _ in range(EVAL_BATCHES)]
    model.train()
    return sum(losses) / len(losses)

def sample(model):
    model.eval()
    torch.manual_seed(42)
    idx = torch.tensor([tok.encode(PROMPT).ids], device=dev)
    out = model.generate(idx, 120, temperature=0.8, top_k=50, eot_id=eot)[0].tolist()
    model.train()
    return tok.decode([t for t in out if t != eot])

model = TinyLlama(Config()).to(dev)
# 权重衰减只加在矩阵上，不加在归一化层的缩放上
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=LR_MAX, betas=(0.9, 0.95))

log = {"train": [], "val": [], "samples": {}}
train_gen = torch.Generator().manual_seed(1)
t0 = time.time()
for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0 or step == STEPS:
        v = evaluate(model)
        log["val"].append([step, v])
        print(f"step {step:5d} | 验证损失 {v:.3f} | 已用 {(time.time()-t0)/60:.1f} 分钟", flush=True)
    if step in SAMPLE_AT:
        s = sample(model)
        log["samples"][step] = s
        print(f"--- 第 {step} 步的生成 ---\n{s}\n", flush=True)
    if step == STEPS:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = get_batch(train_data, train_gen)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止某一步更新过猛
    opt.step()
    log["train"].append([step, loss.item()])
    if step % 50 == 0:
        print(f"step {step:5d} | 训练损失 {loss.item():.3f} | 学习率 {lr_at(step):.2e}", flush=True)

log["minutes"] = (time.time() - t0) / 60
log["tokens_seen"] = STEPS * BATCH * SEQ
torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
json.dump(log, open(os.path.join(OUT, "log.json"), "w"), ensure_ascii=False, indent=1)
print(f"训练结束：共 {STEPS} 步，见过 {log['tokens_seen']:,} 个 token，用时 {log['minutes']:.1f} 分钟")
