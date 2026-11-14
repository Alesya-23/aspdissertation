"""Поиск: dense + BM25, объединение RRF и подсказки из номера статьи.

Юридический запрос часто содержит номер статьи («ст. 158», «часть 2 статьи
105») — это самый сильный сигнал. Поэтому номера извлекаются из запроса и
дают буст соответствующим чанкам, а не жёсткий фильтр: если пользователь
ошибся номером, поиск всё равно вернёт осмысленные нормы.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .store import HybridIndex

ARTICLE_RE = re.compile(
    r"(?:ст\.?|стать[яиею]|статьей)\s*(\d{1,3}(?:\.\d{1,2})?)", re.IGNORECASE
)
PART_RE = re.compile(r"(?:ч\.?|част[ьию]|частью)\s*(\d{1,2}(?:\.\d{1,2})?)", re.IGNORECASE)

RRF_K = 2
W_DENSE = 1.0
W_BM25 = 0.8
ARTICLE_BOOST = 0.010
PART_BOOST = 0.004


@dataclass
class Hit:
    score: float
    chunk: dict
    dense_rank: int | None = None
    bm25_rank: int | None = None
    article_match: bool = False
    part_match: bool = False


def query_hints(query: str) -> tuple[set[str], set[str]]:
    articles = {m.group(1) for m in ARTICLE_RE.finditer(query)}
    parts = {m.group(1) for m in PART_RE.finditer(query)}
    return articles, parts


class Retriever:
    def __init__(self, index: HybridIndex, embedder, score_ratio: float = 0.6) -> None:
        self.index = index
        self.embedder = embedder
        self.score_ratio = score_ratio

    def search(
        self,
        query: str,
        k: int = 6,
        candidates: int = 50,
        per_article: int = 2,
    ) -> list[Hit]:
        query_vector = self.embedder.encode_query(query)
        dense = self.index.search_dense(query_vector, candidates)
        sparse = self.index.search_bm25(query, candidates)
        articles, parts = query_hints(query)

        fused: dict[int, Hit] = {}
        for rank, (idx, _) in enumerate(dense):
            hit = fused.setdefault(idx, Hit(score=0.0, chunk=self.index.chunks[idx]))
            hit.score += W_DENSE / (RRF_K + rank + 1)
            hit.dense_rank = rank
        for rank, (idx, _) in enumerate(sparse):
            hit = fused.setdefault(idx, Hit(score=0.0, chunk=self.index.chunks[idx]))
            hit.score += W_BM25 / (RRF_K + rank + 1)
            hit.bm25_rank = rank

        asks_plenum = bool(re.search(r"пленум|постановлени|разъяснени", query, re.IGNORECASE))
        # Если пользователь явно назвал статью УК РФ — показываем именно её
        # (при наличии таких чанков среди кандидатов). Если же он спрашивает
        # о разъяснениях Пленума, фильтр не применяем: ППВС может толковать
        # несколько статей, и ответ должен остаться в корпусе ППВС.
        if articles and not asks_plenum:
            explicit = {
                idx: hit
                for idx, hit in fused.items()
                if hit.chunk.get("act", "УК РФ") == "УК РФ" and hit.chunk["article"] in articles
            }
            if explicit:
                fused = explicit

        for hit in fused.values():
            chunk = hit.chunk
            if chunk["article"] in articles:
                hit.score += ARTICLE_BOOST
                hit.article_match = True
            if chunk.get("act") == "ППВС" and asks_plenum:
                hit.score += ARTICLE_BOOST
            if parts and parts & set(re.findall(r"\d+(?:\.\d+)?", chunk.get("part", ""))):
                hit.score += PART_BOOST
                hit.part_match = True

        ordered = sorted(fused.values(), key=lambda h: h.score, reverse=True)
        all_ordered = list(ordered)
        # Отсекаем явный шум: всё, что сильно ниже лучшего результата.
        if ordered and self.score_ratio:
            threshold = ordered[0].score * self.score_ratio
            ordered = [hit for hit in ordered if hit.score >= threshold]
        selected: list[Hit] = []
        seen_docs: dict[str, int] = {}
        for hit in ordered:
            key = hit.chunk.get("doc_id") or hit.chunk["article"]
            if seen_docs.get(key, 0) >= per_article:
                continue
            seen_docs[key] = seen_docs.get(key, 0) + 1
            selected.append(hit)
            if len(selected) >= k:
                break

        # Балансировка актов: на вопрос о наказании/составе нужен ещё и УК РФ,
        # а на вопрос о разъяснениях — хотя бы одно ППВС. Иначе один акт
        # вытесняет другой и ответ теряет норму либо практику.
        wants_code = bool(
            re.search(r"наказани|санкци|ответственност|квалифиц|состав|признак", query, re.IGNORECASE)
        )
        has_code = any(hit.chunk.get("act", "УК РФ") == "УК РФ" for hit in selected)
        has_plenum = any(hit.chunk.get("act") == "ППВС" for hit in selected)

        def add_first(predicate) -> None:
            for hit in all_ordered:
                if hit in selected or not predicate(hit):
                    continue
                key = hit.chunk.get("doc_id") or hit.chunk["article"]
                if seen_docs.get(key, 0) >= per_article:
                    continue
                selected.append(hit)
                seen_docs[key] = seen_docs.get(key, 0) + 1
                return

        if wants_code and not has_code:
            add_first(lambda h: h.chunk.get("act", "УК РФ") == "УК РФ")
        if asks_plenum and not has_plenum:
            add_first(lambda h: h.chunk.get("act") == "ППВС")
        return selected


def format_context(hits: list[Hit], max_chars: int | None = None) -> str:
    """Собирает фрагменты в пронумерованный контекст для LLM."""
    blocks: list[str] = []
    for number, hit in enumerate(hits, 1):
        chunk = hit.chunk
        header = f"[S{number}] {chunk_source(chunk)}"
        body = chunk["text"]
        if max_chars and len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + " …"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def chunk_source(chunk: dict) -> str:
    if chunk.get("act") == "ППВС":
        date = chunk.get("doc_date", "")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""):
            year, month, day = date.split("-")
            date = f"{day}.{month}.{year}"
        number = chunk.get("doc_number", "")
        paragraph = chunk.get("paragraph", "")
        source = f"ППВС от {date} № {number} «{chunk.get('doc_title', '')}»"
        if paragraph:
            source += f", п. {paragraph}"
        return source
    source = f"{chunk.get('act', 'УК РФ')}, ст. {chunk['article']}"
    if chunk.get("article_title"):
        source += f" «{chunk['article_title']}»"
    if chunk.get("part"):
        source += f", {chunk['part']}"
    if chunk.get("kind") == "note":
        source += " (примечания)"
    return source
