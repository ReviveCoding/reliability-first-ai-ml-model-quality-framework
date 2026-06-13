from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str):
    return TOKEN_RE.findall(str(text).lower())

class SimpleBM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tokens = [tokenize(d) for d in docs]
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(t) for t in self.tokens) / max(1, len(self.tokens))
        self.df = defaultdict(int)
        for toks in self.tokens:
            for tok in set(toks):
                self.df[tok] += 1
        self.N = len(docs)
        self.tfs = [Counter(t) for t in self.tokens]

    def score(self, query: str, idx: int) -> float:
        q = tokenize(query)
        dl = len(self.tokens[idx]) or 1
        score = 0.0
        for tok in q:
            if tok not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[tok] + 0.5) / (self.df[tok] + 0.5))
            tf = self.tfs[idx][tok]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (tf * (self.k1 + 1)) / (denom or 1)
        return score

    def retrieve(self, query: str, top_k: int = 20):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
