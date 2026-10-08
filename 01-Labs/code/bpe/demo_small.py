"""从最小的例子开始：字节、预切分、在一句绕口令上一步步训练 BPE。"""
from bpe import pretokenize, train

print("=== 1. 文字在计算机里是字节 ===")
for s in ["a", "cat", "é", "地", "猫咪", "😀"]:
    b = s.encode("utf-8")
    print(f"{s!r:6} -> {len(s)} 个字符，{len(b)} 个字节: {list(b)}")

print("\n=== 2. 预切分 ===")
print([w.replace(" ", "␣") for w in pretokenize("Once upon a time, Lily had 3 cats!  她很开心。")])

print("\n=== 3. 在一句绕口令上训练 BPE ===")
text = "Peter Piper picked a peck of pickled peppers"
print("训练文本:", text)
vis = lambda b: b.decode().replace(" ", "␣")   # 用 ␣ 显示空格，看得清楚
show = lambda toks: " ".join(vis(t) for t in toks)

def log(step, pair, count, words):
    a, b = vis(pair[0]), vis(pair[1])
    print(f"\n第 {step} 次合并：{a} + {b} -> {a+b}（出现 {count} 次）")
    print("  合并前各词的切分：", " | ".join(show(t) for t in words.values()))

merges = train(text, 256 + 8, log=log)
print("\n学到的 8 条合并规则:", "，".join(vis(a + b) for a, b in merges))
