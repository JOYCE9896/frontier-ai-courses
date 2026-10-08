"""准备数据：训练一个 BPE 分词器，把 TinyStories 编码成 token ID 存成二进制文件。"""
import os, sys, json, time
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
print(f"训练集 {len(train_texts):,} 篇故事，验证集 {len(val_texts):,} 篇")

# 1. 训练字节级 BPE 分词器（BPE 算法本身在练习 2 里手写，这里用现成库加快速度）
t0 = time.time()
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["<|endoftext|>"],
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tok.train_from_iterator(train_texts, trainer=trainer)
tok.save(os.path.join(OUT, "tokenizer.json"))
print(f"分词器训练完成，词表大小 {tok.get_vocab_size()}，用时 {time.time()-t0:.0f} 秒")

example = "Once upon a time, there was a little dog named Max."
enc = tok.encode(example)
print("示例句子:", example)
print("切成的 token:", enc.tokens)
print("token ID:", enc.ids)

# 2. 编码全部故事，故事之间用 <|endoftext|> 隔开
eot = tok.token_to_id("<|endoftext|>")
def encode_all(texts, path):
    ids = []
    for batch_start in range(0, len(texts), 10000):
        for e in tok.encode_batch(texts[batch_start:batch_start+10000]):
            ids.extend(e.ids); ids.append(eot)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)
    return len(arr)

t0 = time.time()
n_train = encode_all(train_texts, os.path.join(OUT, "train.bin"))
n_val = encode_all(val_texts, os.path.join(OUT, "val.bin"))
print(f"编码完成：训练集 {n_train:,} 个 token，验证集 {n_val:,} 个 token，用时 {time.time()-t0:.0f} 秒")
print(f"平均每篇故事 {n_train/len(train_texts):.0f} 个 token")
