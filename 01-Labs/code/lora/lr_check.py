"""补充实验：全量微调和 LoRA r=64 改用更小的学习率，结果会怎样？"""
import os, json, time, torch
from finetune import fresh, apply_lora, evaluate, dragon_rate, batch, dragon_train, dragon_val, general_val, STEPS, dev

out = {}
for name, r, lr in [("全量微调，学习率 5e-5", None, 5e-5), ("LoRA r=64，学习率 3e-4", 64, 3e-4), ("LoRA r=1，学习率 5e-3", 1, 5e-3)]:
    torch.manual_seed(0)
    model = fresh()
    if r is not None:
        apply_lora(model, r).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(1)
    for step in range(STEPS):
        _, loss = model(*batch(dragon_train, g))
        opt.zero_grad(); loss.backward(); opt.step()
    d, gl = evaluate(model, dragon_val), evaluate(model, general_val)
    rate, _ = dragon_rate(model)
    out[name] = [d, gl, rate]
    print(f"{name}: 龙故事损失 {d:.3f}，普通故事损失 {gl:.3f}，写到龙的比例 {rate:.0%}", flush=True)
json.dump(out, open(os.path.expanduser("~/.labs_data/lora_lr.json"), "w"), ensure_ascii=False)
