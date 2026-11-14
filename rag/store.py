"""Гибридный индекс: FAISS (косинус) + BM25, персистентность на диск."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from .bm25 import BM25, tokenize


class HybridIndex:
    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.index: faiss.Index | None = None
        self.bm25: BM25 | None = None
        self.meta: dict = {}

    @classmethod
    def build(cls, chunks: Sequence[dict], embedder, show_progress: bool = True) -> "HybridIndex":
        obj = cls()
        obj.chunks = list(chunks)
        texts = [c["text"] for c in obj.chunks]
        vectors = embedder.encode_documents(texts, show_progress=show_progress)
        faiss.normalize_L2(vectors)
        obj.index = faiss.IndexFlatIP(vectors.shape[1])
        obj.index.add(vectors)
        obj.bm25 = BM25().fit([tokenize(t) for t in texts])
        obj.meta = {
            "embedding_model": embedder.model_name,
            "dim": int(vectors.shape[1]),
            "count": len(obj.chunks),
            "strategy": obj.chunks[0]["strategy"] if obj.chunks else "",
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return obj

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise RuntimeError("индекс не построен")
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path | str) -> "HybridIndex":
        directory = Path(directory)
        obj = cls()
        obj.index = faiss.read_index(str(directory / "index.faiss"))
        obj.chunks = [
            json.loads(line)
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        obj.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        obj.bm25 = BM25().fit([tokenize(c["text"]) for c in obj.chunks])
        return obj

    def search_dense(self, query_vector: np.ndarray, k: int = 50) -> list[tuple[int, float]]:
        if self.index is None:
            return []
        vector = np.asarray([query_vector], dtype=np.float32)
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(k, len(self.chunks)))
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def search_bm25(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        return self.bm25.top_k(tokenize(query), k=k)

    def __len__(self) -> int:
        return len(self.chunks)
