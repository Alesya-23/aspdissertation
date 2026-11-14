"""BM25 для точного совпадения юридических терминов, статей и аббревиатур.

Векторный поиск плохо ловит редкие токены («ст. 105.1», «ч. 2», «УК»),
поэтому используется гибрид: dense + BM25, объединение через RRF.

Для русского применяется стеммер Snowball: «кражи» и «кража» дают один токен.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

try:
    import snowballstemmer

    _STEMMER = snowballstemmer.stemmer("russian")
except Exception:  # pragma: no cover - на случай отсутствия пакета
    _STEMMER = None

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+(?:[.\-][а-яёa-z0-9]+)*", re.IGNORECASE)

STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из",
    "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам",
    "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где",
    "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб",
    "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда",
    "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним", "здесь",
    "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда",
    "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой",
    "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между", "это",
}


def normalize(word: str) -> str:
    return word.lower().replace("ё", "е")


def stem(word: str) -> str:
    word = normalize(word)
    if _STEMMER is None or not re.search(r"[а-я]", word):
        return word
    return _STEMMER.stemWord(word)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(text):
        token = normalize(raw)
        if drop_stopwords and token in STOPWORDS:
            continue
        tokens.append(stem(token))
    return tokens


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.avg_len = 0.0
        self.idf: dict[str, float] = {}
        self.term_freqs: list[Counter] = []

    def fit(self, tokenized_docs: Sequence[Sequence[str]]) -> "BM25":
        self.term_freqs = [Counter(doc) for doc in tokenized_docs]
        self.doc_len = [sum(counter.values()) for counter in self.term_freqs]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        doc_freq: Counter = Counter()
        for counter in self.term_freqs:
            doc_freq.update(counter.keys())
        total = len(self.term_freqs)
        self.idf = {
            term: math.log((total - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in doc_freq.items()
        }
        return self

    def score(self, query_tokens: Iterable[str], doc_index: int) -> float:
        counter = self.term_freqs[doc_index]
        length = self.doc_len[doc_index] or 1
        score = 0.0
        for token in query_tokens:
            tf = counter.get(token, 0)
            if not tf:
                continue
            idf = self.idf.get(token, 0.0)
            score += idf * tf * (self.k1 + 1) / (
                tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            )
        return score

    def top_k(self, query_tokens: Sequence[str], k: int = 50) -> list[tuple[int, float]]:
        scored = [(idx, self.score(query_tokens, idx)) for idx in range(len(self.term_freqs))]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(idx, score) for idx, score in scored[:k] if score > 0]
