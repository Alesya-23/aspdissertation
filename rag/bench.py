"""Сравнение локальных GGUF-моделей на одном и том же RAG-контексте.

Метрики: время ответа, токенов/с, длина ответа, наличие ссылок [S…] и
номеров норм. Контекст для всех моделей одинаковый (один retrieval).
"""

from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path

from .config import Settings
from .generate import Generator
from .pipeline import RAGPipeline

MODELS = {
    "yandex": "YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf",
    "saiga8": "saiga_yandexgpt_8b-q4_k_m.gguf",
    "saiga12": "saiga_gemma3_12b-q4_k_m.gguf",
    "tiny": "model-2-q8_0.gguf",
}
# 12B в Q4_K_M не влезает в 8 ГБ целиком — часть слоёв оставляем на CPU.
GPU_LAYERS = {"saiga12": 20}


def run_benchmark(
    settings: Settings,
    questions: list[dict],
    model_names: list[str],
    max_new_tokens: int = 300,
    gpu_layers: int | None = None,
) -> dict:
    pipeline = RAGPipeline(settings, strategy="structure")
    results: dict = {"embedding_model": settings.embedding_model, "models": {}}

    for name in model_names:
        filename = MODELS.get(name, name)
        path = Path(filename)
        if not path.is_absolute():
            path = settings.base_dir / filename
        if not path.exists():
            print(f"[bench] нет файла модели: {path}")
            continue

        layers = gpu_layers if gpu_layers is not None else GPU_LAYERS.get(name, settings.n_gpu_layers)
        print(f"\n[bench] {name}: {path.name}, n_gpu_layers={layers}")
        try:
            generator = Generator(
                path,
                n_ctx=settings.n_ctx,
                n_gpu_layers=layers,
                temperature=settings.temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            print(f"[bench] не удалось загрузить {name}: {exc}")
            results["models"][name] = {"error": str(exc)}
            _free()
            continue

        model_results = []
        for item in questions:
            question = item["q"]
            hits = pipeline.retrieve(question, k=settings.top_k)
            context = pipeline.context_for(hits, expand="article")
            started = time.perf_counter()
            try:
                answer = generator.answer(question, context, stream=False)
            except Exception as exc:
                model_results.append({"q": question, "error": str(exc)})
                continue
            elapsed = time.perf_counter() - started
            completion_tokens = generator.count_tokens(answer)
            model_results.append(
                {
                    "q": question,
                    "answer": answer,
                    "seconds": round(elapsed, 2),
                    "completion_tokens": completion_tokens,
                    "tokens_per_second": round(completion_tokens / elapsed, 2) if elapsed else None,
                    "citations": len(re.findall(r"\[S\d+\]", answer)),
                    "articles": sorted(set(re.findall(r"ст\.?\s*(\d{1,3}(?:\.\d{1,2})?)", answer))),
                    "sources": [h.chunk.get("doc_number") or h.chunk.get("article") for h in hits],
                }
            )
            print(
                f"  {question[:60]}… {elapsed:.1f}s, "
                f"{model_results[-1]['tokens_per_second']} tok/s"
            )
        results["models"][name] = {
            "file": path.name,
            "gpu_layers": layers,
            "answers": model_results,
        }
        _free(generator)

    output = settings.data_dir / "bench_results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[bench] сохранено: {output}")
    return results


def _free(generator=None) -> None:
    try:
        if generator is not None:
            generator.llm.close()
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
