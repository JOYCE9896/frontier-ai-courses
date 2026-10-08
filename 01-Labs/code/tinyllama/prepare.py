"""Prepare the data: train a BPE tokenizer and save TinyStories as token IDs in a binary file."""
import os, time
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATA = os.path.expanduser("~/.labs_data/tinystories")
OUT = os.path.join(DATA, "prepared")
VOCAB_SIZE = 4096
os.makedirs(OUT, exist_ok=True)

def load(name):
    f = [x for x in os.listdir(DATA) if x.startswith(name) and x.endswith(".parquet")][0]
    return pq.read_table(os.path.join(DATA, f)).column("text").to_pylist()

train_texts = load("train-00000")
val_texts = load("validation")
print(f"train: {len(train_texts):,} stories, validation: {len(val_texts):,} stories")

# 1. Train a byte-level BPE tokenizer (lab 02 writes BPE by hand; here a library is used for speed)
t0 = time.time()
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["<|endoftext|>"],
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tok.train_from_iterator(train_texts, trainer=trainer)
tok.save(os.path.join(OUT, "tokenizer.json"))
print(f"tokenizer trained: vocabulary size {tok.get_vocab_size()}, took {time.time()-t0:.0f}s")

example = "Once upon a time, there was a little dog named Max."
enc = tok.encode(example)
print("sentence: ", example)
print("tokens:   ", enc.tokens)
print("token IDs:", enc.ids)

# 2. Encode every story; stories are separated by the special token <|endoftext|>
eot = tok.token_to_id("<|endoftext|>")
def encode_all(texts, path):
    ids = []
    for start in range(0, len(texts), 10000):
        for e in tok.encode_batch(texts[start:start + 10000]):
            ids.extend(e.ids); ids.append(eot)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)
    return len(arr)

t0 = time.time()
n_train = encode_all(train_texts, os.path.join(OUT, "train.bin"))
n_val = encode_all(val_texts, os.path.join(OUT, "val.bin"))
print(f"encoded: {n_train:,} training tokens, {n_val:,} validation tokens, took {time.time()-t0:.0f}s")
print(f"average story length: {n_train/len(train_texts):.0f} tokens")
