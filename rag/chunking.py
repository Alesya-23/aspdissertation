"""Чанкинг нормативного текста.

Два режима:

* ``structure`` (основной) — режем по структуре нормы: части статьи и
  примечания, никогда не пересекая границу статьи. Каждый чанк получает
  заголовок-контекст «УК РФ -> раздел -> глава -> статья -> часть», поэтому
  короткая часть («2. Кража, совершенная:») не теряет смысл без соседей.
* ``fixed`` — наивный baseline: скользящее окно фиксированного размера с
  перекрытием. Нужен только для сравнения в экспериментах.

Перекрытие в основном режиме — на уровне частей (по умолчанию 1 часть),
а внутри длинной части — на уровне предложений (15%).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .corpus import Article, Element

CountTokens = Callable[[str], int]

PART_LABEL_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)$")


@dataclass
class Chunk:
    id: str
    act: str
    article: str
    article_title: str
    section: str
    chapter: str
    part: str
    kind: str  # part | note
    body: str
    text: str
    tokens: int
    url: str
    strategy: str
    parts: list[str] = field(default_factory=list)
    doc_id: str = ""
    doc_number: str = ""
    doc_date: str = ""
    doc_title: str = ""
    paragraph: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "act": self.act,
            "article": self.article,
            "article_title": self.article_title,
            "section": self.section,
            "chapter": self.chapter,
            "part": self.part,
            "kind": self.kind,
            "body": self.body,
            "text": self.text,
            "tokens": self.tokens,
            "url": self.url,
            "strategy": self.strategy,
            "parts": self.parts,
            "doc_id": self.doc_id,
            "doc_number": self.doc_number,
            "doc_date": self.doc_date,
            "doc_title": self.doc_title,
            "paragraph": self.paragraph,
        }

    def citation(self) -> str:
        if self.act == "ППВС":
            date = _format_date(self.doc_date)
            number = f" № {self.doc_number}" if self.doc_number else ""
            paragraph = f", п. {self.paragraph}" if self.paragraph else ""
            return f"ППВС от {date}{number}{paragraph}"
        base = f"{self.act}, ст. {self.article}"
        if self.kind == "note":
            return f"{base}, примечания"
        if self.part:
            return f"{base}, {self.part}"
        return base


def _format_date(iso: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", iso or ""):
        year, month, day = iso.split("-")
        return f"{day}.{month}.{year}"
    return iso


def _header(article: Article, act: str, part_label: str = "", notes: bool = False) -> str:
    pieces = [act]
    if article.part:
        pieces.append(article.part)
    if article.section:
        pieces.append(article.section)
    if article.chapter:
        pieces.append(article.chapter)
    article_part = f"Статья {article.number}. {article.title}".strip()
    pieces.append(article_part)
    if notes:
        pieces.append("Примечания")
    elif part_label:
        pieces.append(part_label)
    return ". ".join(p for p in pieces if p).strip()


def _part_label(elements: Sequence[Element]) -> str:
    labels: list[str] = []
    for element in elements:
        if element.kind == "part" and element.label and element.label not in labels:
            labels.append(element.label)
    if not labels:
        return ""
    if len(labels) == 1:
        return f"ч. {labels[0]}"
    return "ч. " + ", ".join(labels)


def _split_sentences(text: str) -> list[str]:
    # Не режем после «ст.», «ч.», «т.е.» и т.п.: требуем заглавную букву/кавычку.
    parts = re.split(r"(?<=[.!?;])\s+(?=[А-ЯЁA-Z«\"(])", text)
    # «2. Кража...» не должно распадаться на «2.» и «Кража...».
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and re.fullmatch(r"\d{1,2}(?:\.\d{1,2})?\.", merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


class StructureChunker:
    """Структурный чанкер: части статьи как атомы, заголовок как контекст."""

    name = "structure"

    def __init__(
        self,
        count_tokens: CountTokens,
        act: str = "УК РФ",
        max_tokens: int = 480,
        target_tokens: int = 360,
        overlap_parts: int = 1,
        sentence_overlap: float = 0.15,
    ) -> None:
        self.count_tokens = count_tokens
        self.act = act
        self.max_tokens = max_tokens
        self.target_tokens = min(target_tokens, max_tokens)
        self.overlap_parts = max(0, overlap_parts)
        self.sentence_overlap = sentence_overlap

    def chunk_article(self, article: Article) -> list[Chunk]:
        if article.repealed or not article.elements:
            return []

        parts = [e for e in article.elements if e.kind in ("part", "preamble", "whole")]
        notes = [e for e in article.elements if e.kind == "note"]

        chunks: list[Chunk] = []
        if parts:
            chunks.extend(self._chunk_kind(article, parts, kind="part"))
        if notes:
            chunks.extend(self._chunk_kind(article, notes, kind="note"))
        return chunks

    def _chunk_kind(self, article: Article, elements: list[Element], kind: str) -> list[Chunk]:
        header_fn = lambda label="": _header(article, self.act, label, notes=(kind == "note"))
        prepared = self._split_oversized(article, elements, header_fn, kind)

        groups: list[list[Element]] = []
        current: list[Element] = []
        for element in prepared:
            candidate = current + [element]
            if current and self._tokens(header_fn(_part_label(candidate)), candidate) > self.max_tokens:
                groups.append(current)
                current = [element]
            else:
                current = candidate
        if current:
            groups.append(current)

        chunks: list[Chunk] = []
        for index, group in enumerate(groups):
            prefix: list[Element] = []
            if self.overlap_parts and index > 0 and kind == "part":
                prefix = groups[index - 1][-self.overlap_parts :]
                while prefix and self._tokens(header_fn(_part_label(prefix + group)), prefix + group) > self.max_tokens:
                    prefix = prefix[1:]
            elements_in_chunk = prefix + group
            label = _part_label(elements_in_chunk) if kind == "part" else ""
            header = header_fn(label)
            body = "\n".join(e.text for e in elements_in_chunk)
            text = f"{header}\n{body}"
            chunks.append(
                Chunk(
                    id=self._chunk_id(article, group[0], index, kind),
                    act=self.act,
                    article=article.number,
                    article_title=article.title,
                    section=article.section,
                    chapter=article.chapter,
                    part=label,
                    kind=kind,
                    body=body,
                    text=text,
                    tokens=self.count_tokens(text),
                    url=article.url,
                    strategy=self.name,
                    parts=[e.label for e in elements_in_chunk if e.label],
                )
            )
        return chunks

    def _split_oversized(
        self,
        article: Article,
        elements: list[Element],
        header_fn: Callable[[str], str],
        kind: str,
    ) -> list[Element]:
        result: list[Element] = []
        for element in elements:
            header = header_fn("" if kind == "note" else f"ч. {element.label}")
            budget = max(32, self.max_tokens - self.count_tokens(header) - 8)
            if self.count_tokens(element.text) <= budget:
                result.append(element)
                continue
            pieces = self._split_text(element.text, budget)
            for piece in pieces:
                result.append(
                    Element(label=element.label, text=piece, kind=element.kind, items=element.items)
                )
        return result

    def _split_text(self, text: str, budget: int) -> list[str]:
        sentences = _split_sentences(text)
        if len(sentences) == 1:
            return self._split_words(text, budget)
        pieces: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if self.count_tokens(sentence) > budget:
                if current:
                    pieces.append(" ".join(current))
                    current = []
                pieces.extend(self._split_words(sentence, budget))
                continue
            candidate = " ".join(current + [sentence])
            if current and self.count_tokens(candidate) > budget:
                pieces.append(" ".join(current))
                overlap_n = max(1, int(len(current) * self.sentence_overlap))
                current = current[-overlap_n:] + [sentence]
                while len(current) > 1 and self.count_tokens(" ".join(current)) > budget:
                    current.pop(0)
            else:
                current.append(sentence)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _split_words(self, text: str, budget: int) -> list[str]:
        """Аварийное дробление по словам с перекрытием (когда нет предложений)."""
        words = text.split()
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if current and self.count_tokens(candidate) > budget:
                pieces.append(" ".join(current))
                overlap = max(1, int(len(current) * self.sentence_overlap))
                current = current[-overlap:]
                while current and self.count_tokens(" ".join(current + [word])) > budget:
                    current.pop(0)
            current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _tokens(self, header: str, elements: Sequence[Element]) -> int:
        return self.count_tokens(header + "\n" + "\n".join(e.text for e in elements))

    def _chunk_id(self, article: Article, first: Element, index: int, kind: str) -> str:
        label = re.sub(r"[^0-9а-яёa-z._-]+", "", (first.label or "all").lower())
        return f"ukrf-{article.number}-{kind}-{label}-{index}"


class FixedChunker:
    """Baseline: скользящее окно фиксированного размера с перекрытием."""

    name = "fixed"

    def __init__(
        self,
        encode: Callable[[str], list[int]],
        decode: Callable[[list[int]], str],
        count_tokens: CountTokens,
        act: str = "УК РФ",
        chunk_tokens: int = 384,
        overlap: float = 0.15,
    ) -> None:
        self.encode = encode
        self.decode = decode
        self.count_tokens = count_tokens
        self.act = act
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap

    def chunk_article(self, article: Article) -> list[Chunk]:
        if article.repealed or not article.elements:
            return []
        header = _header(article, self.act)
        body = "\n".join(e.text for e in article.elements)
        ids = self.encode(f"{header}\n{body}")
        step = max(1, int(self.chunk_tokens * (1 - self.overlap)))
        chunks: list[Chunk] = []
        for index, start in enumerate(range(0, len(ids), step)):
            window = ids[start : start + self.chunk_tokens]
            if not window:
                break
            piece = self.decode(window).strip()
            text = f"{header}\n{piece}"
            chunks.append(
                Chunk(
                    id=f"ukrf-{article.number}-fixed-{start}",
                    act=self.act,
                    article=article.number,
                    article_title=article.title,
                    section=article.section,
                    chapter=article.chapter,
                    part="",
                    kind="part",
                    body=piece,
                    text=text,
                    tokens=self.count_tokens(text),
                    url=article.url,
                    strategy=self.name,
                    parts=[],
                )
            )
            if start + self.chunk_tokens >= len(ids):
                break
        return chunks


class PlenumChunker:
    """Чанкер постановлений Пленума ВС РФ: пункты + заголовок с реквизитами."""

    name = "structure"

    def __init__(
        self,
        count_tokens: CountTokens,
        max_tokens: int = 480,
        overlap_paras: int = 1,
        sentence_overlap: float = 0.15,
    ) -> None:
        self.count_tokens = count_tokens
        self.max_tokens = max_tokens
        self.overlap_paras = max(0, overlap_paras)
        self.sentence_overlap = sentence_overlap

    def chunk_document(self, doc) -> list[Chunk]:
        elements = [p for p in doc.paragraphs if p.text.strip()]
        if not elements:
            return []
        prepared: list = []
        for element in elements:
            header = self._header(doc, [element.label] if element.label else [])
            budget = max(64, self.max_tokens - self.count_tokens(header) - 8)
            if self.count_tokens(element.text) <= budget:
                prepared.append(element)
                continue
            for piece in self._split_text(element.text, budget):
                prepared.append(type(element)(label=element.label, text=piece, kind=element.kind))

        groups: list[list] = []
        current: list = []
        for element in prepared:
            candidate = current + [element]
            if current and self._tokens(doc, _labels(candidate), candidate) > self.max_tokens:
                groups.append(current)
                current = [element]
            else:
                current = candidate
        if current:
            groups.append(current)

        chunks: list[Chunk] = []
        for index, group in enumerate(groups):
            prefix: list = []
            if self.overlap_paras and index > 0:
                previous = groups[index - 1]
                same_paragraph = bool(
                    group[0].label and previous and previous[-1].label == group[0].label
                )
                if not same_paragraph:
                    prefix = previous[-self.overlap_paras :]
                    while prefix and self._tokens(doc, _labels(prefix + group), prefix + group) > self.max_tokens:
                        prefix = prefix[1:]
            items = prefix + group
            labels = _labels(items)
            header = self._header(doc, labels)
            body = "\n".join(item.text for item in items)
            text = f"{header}\n{body}"
            first = group[0]
            paragraph = ", ".join(labels)
            chunks.append(
                Chunk(
                    id=f"ppvs-{doc.doc_id}-p{re.sub(r'[^0-9.]+', '', paragraph) or 'all'}-{index}",
                    act="ППВС",
                    article="",
                    article_title=doc.title,
                    section="",
                    chapter="",
                    part="",
                    kind="paragraph",
                    body=body,
                    text=text,
                    tokens=self.count_tokens(text),
                    url=doc.url,
                    strategy=self.name,
                    parts=labels,
                    doc_id=doc.doc_id,
                    doc_number=doc.number,
                    doc_date=doc.date,
                    doc_title=doc.title,
                    paragraph=paragraph,
                )
            )
        return chunks

    def _header(self, doc, labels: Sequence[str]) -> str:
        title = doc.title if len(doc.title) <= 160 else doc.title[:157] + "…"
        base = f"Постановление Пленума ВС РФ от {_format_date(doc.date)} № {doc.number} «{title}»"
        if not labels:
            return f"{base}. Преамбула"
        if len(labels) == 1:
            return f"{base}. Пункт {labels[0]}"
        return f"{base}. Пункты {', '.join(labels)}"

    def _split_text(self, text: str, budget: int) -> list[str]:
        sentences = _split_sentences(text)
        if len(sentences) == 1:
            return _split_words(text, budget, self.count_tokens, self.sentence_overlap)
        pieces: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if self.count_tokens(sentence) > budget:
                if current:
                    pieces.append(" ".join(current))
                    current = []
                pieces.extend(_split_words(sentence, budget, self.count_tokens, self.sentence_overlap))
                continue
            candidate = " ".join(current + [sentence])
            if current and self.count_tokens(candidate) > budget:
                pieces.append(" ".join(current))
                overlap_n = max(1, int(len(current) * self.sentence_overlap))
                current = current[-overlap_n:] + [sentence]
                while len(current) > 1 and self.count_tokens(" ".join(current)) > budget:
                    current.pop(0)
            else:
                current.append(sentence)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _tokens(self, doc, labels: Sequence[str], elements: Sequence) -> int:
        header = self._header(doc, labels)
        return self.count_tokens(header + "\n" + "\n".join(e.text for e in elements))


def _labels(elements: Sequence) -> list[str]:
    labels: list[str] = []
    for element in elements:
        if element.label and element.label not in labels:
            labels.append(element.label)
    return labels


def _split_words(text: str, budget: int, count_tokens: CountTokens, overlap_ratio: float) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and count_tokens(candidate) > budget:
            pieces.append(" ".join(current))
            overlap = max(1, int(len(current) * overlap_ratio))
            current = current[-overlap:]
            while current and count_tokens(" ".join(current + [word])) > budget:
                current.pop(0)
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def build_chunks(articles: Iterable[Article], chunker) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(chunker.chunk_article(article))
    return chunks


def chunk_stats(chunks: Sequence[Chunk]) -> dict:
    if not chunks:
        return {"count": 0}
    tokens = sorted(c.tokens for c in chunks)
    return {
        "count": len(chunks),
        "articles": len({c.article for c in chunks}),
        "tokens_mean": round(sum(tokens) / len(tokens), 1),
        "tokens_p50": tokens[len(tokens) // 2],
        "tokens_p95": tokens[int(len(tokens) * 0.95) - 1],
        "tokens_max": tokens[-1],
        "notes": sum(1 for c in chunks if c.kind == "note"),
    }
