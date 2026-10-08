"""Start small: bytes, pre-tokenization, and BPE trained step by step on one tongue twister."""
import json, os
from bpe import pretokenize, train

# example strings (some of them non-English) live in a separate data file
SAMPLES = json.load(open(os.path.join(os.path.dirname(__file__), "samples.json"), encoding="utf-8"))

print("=== 1. Text is stored as bytes ===")
for s in SAMPLES["byte_examples"]:
    b = s.encode("utf-8")
    print(f"{s!r:6} -> {len(s)} character(s), {len(b)} byte(s): {list(b)}")

print("\n=== 2. Pre-tokenization ===")
print([w.replace(" ", "␣") for w in pretokenize(SAMPLES["pretokenize_example"])])

print("\n=== 3. Training BPE on a tongue twister ===")
text = "Peter Piper picked a peck of pickled peppers"
print("training text:", text)
vis = lambda b: b.decode().replace(" ", "␣")   # show spaces as ␣ so they are visible
show = lambda toks: " ".join(vis(t) for t in toks)

def log(step, pair, count, words):
    a, b = vis(pair[0]), vis(pair[1])
    print(f"\nmerge {step}: {a} + {b} -> {a+b} ({count} occurrences)")
    print("  words before this merge:", " | ".join(show(t) for t in words.values()))

merges = train(text, 256 + 8, log=log)
print("\nthe 8 merge rules learned:", ", ".join(vis(a + b) for a, b in merges))
