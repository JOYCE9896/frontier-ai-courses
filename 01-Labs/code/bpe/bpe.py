"""手写的字节级 BPE 分词器：训练（学合并规则）、编码（文字 -> token ID）、解码（token ID -> 文字）。"""
import re
from collections import Counter

# 预切分：先把文本粗切成"词"，BPE 只在词的内部合并，不会跨词。
# 依次匹配：带可选前导空格的英文字母串、数字串、其他符号串（包括中文），以及空白。
PRETOKEN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

def pretokenize(text):
    return PRETOKEN.findall(text)


def merge_pair(tokens, pair, new):
    """在一个 token 序列里，把所有相邻的 pair 换成 new。"""
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(new); i += 2
        else:
            out.append(tokens[i]); i += 1
    return out


def train(text, vocab_size, log=None):
    """从文本学 vocab_size - 256 条合并规则。返回合并列表，每条是 (左, 右)，都是 bytes。"""
    # 1. 统计每个"词"出现几次。同一个词不管出现多少次，只需要处理一次，再乘上次数。
    word_counts = Counter(pretokenize(text))
    # 2. 每个词先拆成单个字节。词表一开始就是 256 种字节。
    words = {w: [bytes([b]) for b in w.encode("utf-8")] for w in word_counts}
    merges = []
    while 256 + len(merges) < vocab_size:
        # 3. 数所有相邻的两个 token 一起出现了多少次
        pairs = Counter()
        for w, toks in words.items():
            for a, b in zip(toks, toks[1:]):
                pairs[(a, b)] += word_counts[w]
        if not pairs:
            break
        # 4. 取次数最多的一对；次数相同时取字典序大的，保证结果确定
        best = max(pairs, key=lambda p: (pairs[p], p))
        new = best[0] + best[1]
        merges.append(best)
        if log is not None:
            log(len(merges), best, pairs[best], words)
        # 5. 在所有词里把这一对合并成新 token
        for w in words:
            if len(words[w]) > 1:
                words[w] = merge_pair(words[w], best, new)
    return merges


class Tokenizer:
    def __init__(self, merges):
        self.merges = merges
        self.rank = {pair: i for i, pair in enumerate(merges)}   # 合并规则的先后顺序
        self.vocab = [bytes([b]) for b in range(256)] + [a + b for a, b in merges]
        self.id_of = {tok: i for i, tok in enumerate(self.vocab)}
        self.cache = {}

    def encode_word(self, word):
        if word in self.cache:
            return self.cache[word]
        toks = [bytes([b]) for b in word.encode("utf-8")]
        while len(toks) > 1:
            # 在当前相邻对里，找最早学到的那条合并规则来用；一条都用不了就结束
            pairs = [(self.rank.get(p, float("inf")), p) for p in zip(toks, toks[1:])]
            r, p = min(pairs)
            if r == float("inf"):
                break
            toks = merge_pair(toks, p, p[0] + p[1])
        ids = [self.id_of[t] for t in toks]
        self.cache[word] = ids
        return ids

    def encode(self, text):
        return [i for w in pretokenize(text) for i in self.encode_word(w)]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def pieces(self, text):
        """返回每个 token 对应的字节，方便看切分结果。"""
        return [self.vocab[i] for i in self.encode(text)]

    def truncated(self, vocab_size):
        """BPE 的合并是一条条按顺序学的，所以只取前 vocab_size-256 条，就得到一个更小词表的分词器。"""
        return Tokenizer(self.merges[: vocab_size - 256])
