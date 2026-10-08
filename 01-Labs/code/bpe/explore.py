"""用训练好的分词器做实验：编码解码、词表大小与压缩率、切分的怪现象、中英文对比。"""
import os, pickle
import pyarrow.parquet as pq
import tiktoken
from bpe import Tokenizer

merges = pickle.load(open(os.path.expanduser("~/.labs_data/bpe_merges.pkl"), "rb"))
tok = Tokenizer(merges)
def vis(b):
    """把一个 token 显示出来：空格显示成 ␣；如果它只是半个汉字的字节，没法单独显示，就写成 <e5> 这样的十六进制。"""
    try:
        return b.decode("utf-8").replace(" ", "␣")
    except UnicodeDecodeError:
        return "".join(f"<{x:02x}>" for x in b)
show = lambda t, s: " | ".join(vis(p) for p in t.pieces(s))

print("=== 1. 编码与解码 ===")
s = "Once upon a time, a little rabbit named Benny found a shiny key."
ids = tok.encode(s)
print("原文:", s)
print("切分:", show(tok, s))
print("ID:  ", ids)
print("解码:", tok.decode(ids))
print("解码后和原文一样吗？", tok.decode(ids) == s)

print("\n=== 2. 词表越大，序列越短 ===")
DATA = os.path.expanduser("~/.labs_data/tinystories")
f = [x for x in os.listdir(DATA) if x.startswith("validation")][0]
val = "\n".join(pq.read_table(os.path.join(DATA, f)).column("text").to_pylist()[:1000])
nbytes = len(val.encode("utf-8"))
print(f"测试文本：验证集前 1000 篇故事（训练时没见过），共 {nbytes:,} 个字节")
for v in [256, 512, 1024, 2048, 4096]:
    t = tok.truncated(v)
    n = len(t.encode(val))
    print(f"词表 {v:5d}：{n:8,} 个 token，平均每个 token {nbytes/n:.2f} 个字节   例句切分：{show(t, 'The little girl was happy.')}")

print("\n=== 3. 切分的怪现象 ===")
for w in ["strawberry", " strawberry", " Strawberry", " strawbery", " hippopotamus", " 2026", " 12345", " Kallini"]:
    print(f"{w!r:18} -> {show(tok, w)}")

print("\n=== 4. 同样的意思，中文和英文要多少 token ===")
pairs = [("The little cat is playing in the garden.", "小猫在花园里玩。"),
         ("Thank you very much for your help today.", "非常感谢你今天的帮助。")]
encs = {"本练习的分词器（只在英文故事上训练，词表 4096）": tok.encode,
        "GPT-2 的分词器（词表 50257）": tiktoken.get_encoding("gpt2").encode,
        "GPT-4o 的分词器（词表约 20 万）": tiktoken.get_encoding("o200k_base").encode}
for en, zh in pairs:
    print(f"\n英文：{en}\n中文：{zh}")
    for name, enc in encs.items():
        print(f"  {name}：英文 {len(enc(en))} 个 token，中文 {len(enc(zh))} 个 token")
print("\n本练习的分词器怎样切中文：", show(tok, "小猫在花园里玩。"))
g4 = tiktoken.get_encoding("o200k_base")
print("GPT-4o 的分词器怎样切中文：", " | ".join(vis(g4.decode_single_token_bytes(i)) for i in g4.encode("小猫在花园里玩。")))
