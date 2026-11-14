"""Сборка RAG: индекс + поиск + (опционально) локальная LLM."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .corpus import Article, corpus_text, load_corpus
from .embeddings import Embedder
from .generate import Generator
from .retriever import Hit, Retriever, chunk_source, format_context
from .store import HybridIndex


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    sources: list[dict] = field(default_factory=list)
    unsupported_articles: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        strategy: str = "structure",
        embedder: Embedder | None = None,
        index: HybridIndex | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.strategy = strategy
        self.embedder = embedder or Embedder(
            self.settings.embedding_model,
            device=self.settings.device,
            batch_size=self.settings.batch_size,
            hf_home=self.settings.data_dir / "models" / "hf",
        )
        self.index = index or HybridIndex.load(self.settings.index_dir(strategy))
        self.retriever = Retriever(self.index, self.embedder)
        self._corpus: dict[str, Article] | None = None
        self._generator: Generator | None = None

    @property
    def corpus(self) -> dict[str, Article]:
        if self._corpus is None:
            self._corpus = {a.number: a for a in load_corpus(self.settings.corpus_path)}
        return self._corpus

    def load_generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(
                self.settings.llm_model,
                n_ctx=self.settings.n_ctx,
                n_gpu_layers=self.settings.n_gpu_layers,
                temperature=self.settings.temperature,
                max_new_tokens=self.settings.max_new_tokens,
            )
        return self._generator

    def context_for(self, hits: list[Hit], expand: str = "article", max_tokens: int = 450) -> str:
        if expand != "article":
            return format_context(hits)
        blocks: list[str] = []
        for number, hit in enumerate(hits, 1):
            chunk = hit.chunk
            if chunk.get("act", "УК РФ") != "УК РФ":
                body = chunk["text"]
            else:
                article = self.corpus.get(chunk["article"])
                body = corpus_text(article) if article is not None else chunk["text"]
                body = _truncate_tokens(body, max_tokens, self.embedder.count_tokens)
            header = f"[S{number}] {chunk_source(chunk)}"
            blocks.append(f"{header}\n{body}")
        return "\n\n".join(blocks)

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.retriever.search(question, k=k or self.settings.top_k)

    def ask(
        self,
        question: str,
        k: int | None = None,
        expand: str = "article",
        generate: bool = True,
        stream: bool = True,
    ) -> Answer:
        hits = self.retrieve(question, k=k)
        context = self.context_for(hits, expand=expand)
        sources = _sources(hits)
        if not generate:
            return Answer(text=context, hits=hits, sources=sources)

        generator = self.load_generator()
        result = generator.answer(question, context, stream=stream)
        if stream:
            pieces: list[str] = []
            for piece in result:
                pieces.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
            sys.stdout.write("\n")
            text = "".join(pieces).strip()
            return Answer(
                text=text,
                hits=hits,
                sources=sources,
                unsupported_articles=_check_articles(text, sources, hits),
            )
        text = result
        return Answer(
            text=text,
            hits=hits,
            sources=sources,
            unsupported_articles=_check_articles(text, sources, hits),
        )


def _truncate_tokens(text: str, max_tokens: int, count_tokens) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    low, high = 1, len(words)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return " ".join(words[:low]) + " …"


def _check_articles(text: str, sources: list[dict], hits: list[Hit] | None = None) -> list[str]:
    """Ищет в ответе номера статей, которых не было в контексте (признак галлюцинации)."""
    import re

    pattern = r"(?:ст\.?|стать[яиею])\s*(\d{1,3}(?:\.\d{1,2})?)"
    mentioned = set(re.findall(pattern, text, flags=re.IGNORECASE))
    available = {source["article"] for source in sources if source.get("article")}
    # ППВС цитирует статьи УК РФ — это допустимый контекст, даже если
    # соответствующий чанк УК не попал в топ.
    for hit in hits or []:
        available |= set(re.findall(pattern, hit.chunk.get("text", ""), flags=re.IGNORECASE))
    return sorted(mentioned - available, key=lambda x: float(x))


def _sources(hits: list[Hit]) -> list[dict]:
    sources: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for number, hit in enumerate(hits, 1):
        chunk = hit.chunk
        key = (
            chunk.get("doc_id") or chunk["article"],
            chunk.get("paragraph") or chunk.get("part", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "n": number,
                "act": chunk.get("act", "УК РФ"),
                "article": chunk["article"],
                "title": chunk.get("article_title", ""),
                "part": chunk.get("part", ""),
                "paragraph": chunk.get("paragraph", ""),
                "doc_number": chunk.get("doc_number", ""),
                "doc_date": chunk.get("doc_date", ""),
                "url": chunk.get("url", ""),
                "score": round(hit.score, 5),
            }
        )
    return sources
