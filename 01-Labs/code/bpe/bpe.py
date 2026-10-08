"""A byte-level BPE tokenizer written by hand: training (learn merge rules), encoding (text -> token IDs), decoding (IDs -> text)."""
import re
from collections import Counter

# Pre-tokenization: first cut the text into rough "words"; BPE merges only inside a word, never across words.
# Alternatives, in order: letters with an optional leading space, digits, other symbols (including Chinese), whitespace.
PRETOKEN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

def pretokenize(text):
    return PRETOKEN.findall(text)


def merge_pair(tokens, pair, new):
    """Replace every adjacent occurrence of pair in a token sequence with new."""
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(new); i += 2
        else:
            out.append(tokens[i]); i += 1
    return out


def train(text, vocab_size, log=None):
    """Learn vocab_size - 256 merge rules from text. Returns a list of (left, right) pairs of bytes."""
    # 1. Count how often each word occurs. A word that occurs 100,000 times is processed once and weighted by its count.
    word_counts = Counter(pretokenize(text))
    # 2. Split every word into single bytes. The vocabulary starts as the 256 possible bytes.
    words = {w: [bytes([b]) for b in w.encode("utf-8")] for w in word_counts}
    merges = []
    while 256 + len(merges) < vocab_size:
        # 3. Count how often each pair of adjacent tokens occurs
        pairs = Counter()
        for w, toks in words.items():
            for a, b in zip(toks, toks[1:]):
                pairs[(a, b)] += word_counts[w]
        if not pairs:
            break
        # 4. Take the most frequent pair; on ties take the lexicographically larger one so results are deterministic
        best = max(pairs, key=lambda p: (pairs[p], p))
        new = best[0] + best[1]
        merges.append(best)
        if log is not None:
            log(len(merges), best, pairs[best], words)
        # 5. Apply the merge in every word
        for w in words:
            if len(words[w]) > 1:
                words[w] = merge_pair(words[w], best, new)
    return merges


class Tokenizer:
    def __init__(self, merges):
        self.merges = merges
        self.rank = {pair: i for i, pair in enumerate(merges)}   # the order in which merges were learned
        self.vocab = [bytes([b]) for b in range(256)] + [a + b for a, b in merges]
        self.id_of = {tok: i for i, tok in enumerate(self.vocab)}
        self.cache = {}

    def encode_word(self, word):
        if word in self.cache:
            return self.cache[word]
        toks = [bytes([b]) for b in word.encode("utf-8")]
        while len(toks) > 1:
            # among the current adjacent pairs, apply the merge that was learned earliest; stop when none applies
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
        """The bytes of each token, handy for looking at how a text is split."""
        return [self.vocab[i] for i in self.encode(text)]

    def truncated(self, vocab_size):
        """Merges are learned one after another, so keeping the first vocab_size-256 gives a smaller tokenizer."""
        return Tokenizer(self.merges[: vocab_size - 256])
