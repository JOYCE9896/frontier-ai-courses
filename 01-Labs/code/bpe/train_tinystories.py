"""在 2 万篇 TinyStories 上训练一个词表大小为 4096 的 BPE 分词器。"""
import os, time, pickle
import pyarrow.parquet as pq
from bpe import train, pretokenize

DATA = os.path.expanduser("~/.labs_data/tinystories")
f = [x for x in os.listdir(DATA) if x.startswith("train-00000")][0]
stories = pq.read_table(os.path.join(DATA, f)).column("text").to_pylist()[:20000]
text = "\n".join(stories)
print(f"训练文本：{len(stories):,} 篇故事，{len(text):,} 个字符，{len(text.encode()):,} 个字节")
words = pretokenize(text)
print(f"预切分后共 {len(words):,} 个词，其中不同的词 {len(set(words)):,} 个")

t0 = time.time()
def log(step, pair, count, _):
    if step <= 30 or step % 500 == 0 or step == 4096 - 256:
        a, b = pair[0].decode(errors="replace"), pair[1].decode(errors="replace")
        print(f"第 {step:4d} 条：{a!r:>10} + {b!r:<10} -> {(a+b)!r:<14} 出现 {count:>7,} 次   （{time.time()-t0:.0f} 秒）", flush=True)
merges = train(text, 4096, log=log)
print(f"训练完成：共 {len(merges)} 条合并规则，用时 {time.time()-t0:.0f} 秒")
pickle.dump(merges, open(os.path.expanduser("~/.labs_data/bpe_merges.pkl"), "wb"))
