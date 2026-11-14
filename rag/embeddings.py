"""Эмбеддинги: sentence-transformers + русско-ориентированные модели.

По умолчанию — ``intfloat/multilingual-e5-base`` (768d, 512 токенов):
хороший баланс качества на русском и размера. Для e5 обязательны префиксы
``query:`` / ``passage:`` — они добавляются автоматически.

Можно переключить модель переменной ``RAG_EMBEDDING_MODEL``:
    deepvk/USER-bge-m3      — сильнее на русском, ~1.4 ГБ
    intfloat/multilingual-e5-large — точнее, но медленнее
    cointegrated/rubert-tiny2 — быстрый CPU-вариант для отладки
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cuda",
        batch_size: int = 32,
        hf_home: Path | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        if hf_home is not None:
            os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        if device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.dim = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length
        self._e5 = "e5" in model_name.lower()

    def _prefix(self, kind: str) -> str:
        if not self._e5:
            return ""
        return "query: " if kind == "query" else "passage: "

    def encode_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode([self._prefix("passage") + t for t in texts], show_progress)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([self._prefix("query") + text], False)[0]

    def _encode(self, texts: Sequence[str], show_progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode_ids(self, text: str) -> list[int]:
        return self.model.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode_ids(self, ids: list[int]) -> str:
        return self.model.tokenizer.decode(ids, skip_special_tokens=True)
