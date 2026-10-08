"""Experiments with the trained tokenizer: encode/decode, vocabulary size vs. compression, odd splits, Chinese vs. English."""
import json, os, pickle
import pyarrow.parquet as pq
import tiktoken
from bpe import Tokenizer

SAMPLES = json.load(open(os.path.join(os.path.dirname(__file__), "samples.json"), encoding="utf-8"))
merges = pickle.load(open(os.path.expanduser("~/.labs_data/bpe_merges.pkl"), "rb"))
tok = Tokenizer(merges)
def vis(b):
    """Show one token: spaces as ␣; a lone byte of a multi-byte character cannot be shown, so print it in hex like <e5>."""
    try:
        return b.decode("utf-8").replace(" ", "␣")
    except UnicodeDecodeError:
        return "".join(f"<{x:02x}>" for x in b)
show = lambda t, s: " | ".join(vis(p) for p in t.pieces(s))

print("=== 1. Encoding and decoding ===")
s = "Once upon a time, a little rabbit named Benny found a shiny key."
ids = tok.encode(s)
print("text:   ", s)
print("pieces: ", show(tok, s))
print("IDs:    ", ids)
print("decoded:", tok.decode(ids))
print("identical to the original?", tok.decode(ids) == s)

print("\n=== 2. Bigger vocabulary, shorter sequences ===")
DATA = os.path.expanduser("~/.labs_data/tinystories")
f = [x for x in os.listdir(DATA) if x.startswith("validation")][0]
val = "\n".join(pq.read_table(os.path.join(DATA, f)).column("text").to_pylist()[:1000])
nbytes = len(val.encode("utf-8"))
print(f"test text: the first 1000 validation stories (not seen in training), {nbytes:,} bytes")
for v in [256, 512, 1024, 2048, 4096]:
    t = tok.truncated(v)
    n = len(t.encode(val))
    print(f"vocab {v:5d}: {n:8,} tokens, {nbytes/n:.2f} bytes per token   example: {show(t, 'The little girl was happy.')}")

print("\n=== 3. Odd splits ===")
for w in ["strawberry", " strawberry", " Strawberry", " strawbery", " hippopotamus", " 2026", " 12345", " Kallini"]:
    print(f"{w!r:18} -> {show(tok, w)}")

print("\n=== 4. The same meaning in English and Chinese: how many tokens? ===")
pairs = SAMPLES["translation_pairs"]          # (English, Chinese) sentences with the same meaning
encs = {"this lab's tokenizer (English stories only, vocab 4096)": tok.encode,
        "GPT-2 tokenizer (vocab 50,257)": tiktoken.get_encoding("gpt2").encode,
        "GPT-4o tokenizer (vocab about 200,000)": tiktoken.get_encoding("o200k_base").encode}
for en, zh in pairs:
    print(f"\nEnglish: {en}\nChinese: {zh}")
    for name, enc in encs.items():
        print(f"  {name}: English {len(enc(en))} tokens, Chinese {len(enc(zh))} tokens")
print("\nhow this lab's tokenizer splits the Chinese sentence:", show(tok, pairs[0][1]))
g4 = tiktoken.get_encoding("o200k_base")
print("how the GPT-4o tokenizer splits it:                 ", " | ".join(vis(g4.decode_single_token_bytes(i)) for i in g4.encode(pairs[0][1])))
