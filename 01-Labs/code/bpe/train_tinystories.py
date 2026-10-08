"""Train a BPE tokenizer with a 4096-token vocabulary on 20,000 TinyStories."""
import os, time, pickle
import pyarrow.parquet as pq
from bpe import train, pretokenize

DATA = os.path.expanduser("~/.labs_data/tinystories")
f = [x for x in os.listdir(DATA) if x.startswith("train-00000")][0]
stories = pq.read_table(os.path.join(DATA, f)).column("text").to_pylist()[:20000]
text = "\n".join(stories)
print(f"training text: {len(stories):,} stories, {len(text):,} characters, {len(text.encode()):,} bytes")
words = pretokenize(text)
print(f"after pre-tokenization: {len(words):,} words, {len(set(words)):,} distinct")

t0 = time.time()
def log(step, pair, count, _):
    if step <= 30 or step % 500 == 0 or step == 4096 - 256:
        a, b = pair[0].decode(errors="replace"), pair[1].decode(errors="replace")
        print(f"merge {step:4d}: {a!r:>10} + {b!r:<10} -> {(a+b)!r:<14} {count:>7,} occurrences   ({time.time()-t0:.0f}s)", flush=True)
merges = train(text, 4096, log=log)
print(f"done: {len(merges)} merge rules in {time.time()-t0:.0f}s")
pickle.dump(merges, open(os.path.expanduser("~/.labs_data/bpe_merges.pkl"), "wb"))
