"""Конфигурация RAG-системы.

Все пути и параметры можно переопределить переменными окружения:
    RAG_DATA_DIR, RAG_EMBEDDING_MODEL, RAG_DEVICE, RAG_LLM_MODEL,
    RAG_LLM_CTX, RAG_LLM_GPU_LAYERS
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _default_llm_path(base: Path) -> Path:
    env = os.environ.get("RAG_LLM_MODEL")
    if env:
        return Path(env).expanduser()
    preferred = base / "YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf"
    if preferred.exists():
        return preferred
    candidates = sorted(base.glob("*.gguf"), key=lambda p: p.stat().st_size)
    if candidates:
        return candidates[0]
    return preferred


@dataclass
class Settings:
    """Настройки по умолчанию для корпуса УК РФ."""

    base_dir: Path = field(default_factory=_project_root)
    data_dir: Path = field(default_factory=lambda: _env_path("RAG_DATA_DIR", _project_root() / "data"))
    act: str = "УК РФ"
    act_full: str = "Уголовный кодекс Российской Федерации"
    source_url: str = "https://ukodeksrf.ru"
    request_delay: float = 0.25

    embedding_model: str = os.environ.get("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    device: str = os.environ.get("RAG_DEVICE", "cuda")
    batch_size: int = 32

    llm_model: Path = field(default_factory=lambda: _default_llm_path(_project_root()))
    n_ctx: int = int(os.environ.get("RAG_LLM_CTX", "4096"))
    n_gpu_layers: int = int(os.environ.get("RAG_LLM_GPU_LAYERS", "-1"))
    temperature: float = 0.1
    top_k: int = 5
    max_new_tokens: int = 512

    # Параметры чанкинга
    chunk_max_tokens: int = 480
    chunk_target_tokens: int = 360
    chunk_overlap_parts: int = 1
    sentence_overlap: float = 0.15

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "raw" / "uk_rf.jsonl"

    @property
    def html_dir(self) -> Path:
        return self.data_dir / "raw" / "html"

    @property
    def index_root(self) -> Path:
        return self.data_dir / "index"

    @property
    def eval_questions_path(self) -> Path:
        return self.base_dir / "eval" / "questions.jsonl"

    def index_dir(self, strategy: str) -> Path:
        return self.index_root / strategy
