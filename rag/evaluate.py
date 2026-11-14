"""Оценка поиска: recall@k и MRR по размеченным вопросам.

Скрипт сравнивает стратегии чанкинга на одном и том же наборе вопросов —
это и есть эксперимент, который показывает, нужен ли структурный чанкинг
и перекрытие.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Sequence

from .chunking import FixedChunker, StructureChunker, build_chunks, chunk_stats
from .corpus import Article, load_corpus
from .embeddings import Embedder
from .retriever import Retriever
from .store import HybridIndex


def load_questions(path: Path) -> list[dict]:
    questions = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                questions.append(json.loads(line))
    return questions


def make_chunker(strategy: str, embedder: Embedder, settings) -> object:
    if strategy == "fixed":
        return FixedChunker(
            encode=embedder.encode_ids,
            decode=embedder.decode_ids,
            count_tokens=embedder.count_tokens,
            act=settings.act,
            chunk_tokens=384,
            overlap=0.15,
        )
    overlap = 0 if strategy in ("structure-nooverlap", "structure-0") else settings.chunk_overlap_parts
    return StructureChunker(
        count_tokens=embedder.count_tokens,
        act=settings.act,
        max_tokens=settings.chunk_max_tokens,
        target_tokens=settings.chunk_target_tokens,
        overlap_parts=overlap,
        sentence_overlap=settings.sentence_overlap,
    )


def build_index_for(
    articles: Sequence[Article],
    strategy: str,
    embedder: Embedder,
    settings,
) -> tuple[HybridIndex, dict]:
    chunker = make_chunker(strategy, embedder, settings)
    chunks = build_chunks(articles, chunker)
    stats = chunk_stats(chunks)
    index = HybridIndex.build([c.to_dict() for c in chunks], embedder, show_progress=True)
    return index, stats


def evaluate_index(index: HybridIndex, embedder: Embedder, questions: Iterable[dict], k: int = 10) -> dict:
    retriever = Retriever(index, embedder)
    recalls = {1: 0, 5: 0, 10: 0}
    reciprocal = 0.0
    part_hits = 0
    part_total = 0
    per_question = []

    for question in questions:
        hits = retriever.search(question["q"], k=k)
        rank = None
        for position, hit in enumerate(hits, 1):
            if hit.chunk["article"] in question["articles"]:
                rank = position
                break
        per_question.append({"q": question["q"], "expected": question["articles"], "rank": rank})
        if rank:
            for cutoff in recalls:
                if rank <= cutoff:
                    recalls[cutoff] += 1
            reciprocal += 1.0 / rank
        if question.get("parts"):
            part_total += 1
            for hit in hits[:5]:
                chunk_part = set(hit.chunk.get("part", "").replace("ч. ", "").replace(" ", "").split(","))
                if hit.chunk["article"] in question["articles"] and set(question["parts"]) & {
                    p for p in chunk_part if p
                }:
                    part_hits += 1
                    break

    total = max(1, len(per_question))
    return {
        "n": len(per_question),
        "recall@1": round(recalls[1] / total, 4),
        "recall@5": round(recalls[5] / total, 4),
        "recall@10": round(recalls[10] / total, 4),
        "mrr@10": round(reciprocal / total, 4),
        "part_accuracy@5": round(part_hits / part_total, 4) if part_total else None,
        "per_question": per_question,
    }


def run_comparison(
    settings,
    strategies: Sequence[str],
    k: int = 10,
    limit_questions: int | None = None,
    verbose: bool = True,
) -> dict:
    articles = [a for a in load_corpus(settings.corpus_path) if not a.repealed and a.elements]
    questions = load_questions(settings.eval_questions_path)
    if limit_questions:
        questions = questions[:limit_questions]

    embedder = Embedder(
        settings.embedding_model,
        device=settings.device,
        batch_size=settings.batch_size,
        hf_home=settings.data_dir / "models" / "hf",
    )

    results: dict = {"embedding_model": settings.embedding_model, "questions": len(questions), "strategies": {}}
    for strategy in strategies:
        if verbose:
            print(f"\n[eval] стратегия: {strategy}")
        started = time.time()
        index, stats = build_index_for(articles, strategy, embedder, settings)
        metrics = evaluate_index(index, embedder, questions, k=k)
        metrics["chunk_stats"] = stats
        metrics["build_seconds"] = round(time.time() - started, 1)
        results["strategies"][strategy] = metrics
        if verbose:
            print(
                f"[eval] {strategy}: recall@1={metrics['recall@1']:.3f} "
                f"recall@5={metrics['recall@5']:.3f} mrr={metrics['mrr@10']:.3f} "
                f"чанков={stats['count']}"
            )

    output = settings.data_dir / "eval_results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"\n[eval] результаты сохранены: {output}")
    return results


def evaluate_ppvs(index: HybridIndex, embedder: Embedder, questions: Iterable[dict], k: int = 10) -> dict:
    """Оценка поиска по ППВС: попал ли нужный документ (дата+номер) в топ."""
    retriever = Retriever(index, embedder)
    recalls = {1: 0, 5: 0, 10: 0}
    reciprocal = 0.0
    per_question = []
    for question in questions:
        hits = retriever.search(question["q"], k=k)
        rank = None
        for position, hit in enumerate(hits, 1):
            chunk = hit.chunk
            key = f"{chunk.get('doc_date', '')}|{chunk.get('doc_number', '')}"
            if key in question["docs"]:
                rank = position
                break
        per_question.append({"q": question["q"], "expected": question["docs"], "rank": rank})
        if rank:
            for cutoff in recalls:
                if rank <= cutoff:
                    recalls[cutoff] += 1
            reciprocal += 1.0 / rank
    total = max(1, len(per_question))
    return {
        "n": len(per_question),
        "recall@1": round(recalls[1] / total, 4),
        "recall@5": round(recalls[5] / total, 4),
        "recall@10": round(recalls[10] / total, 4),
        "mrr@10": round(reciprocal / total, 4),
        "per_question": per_question,
    }


def run_ppvs_evaluation(settings, k: int = 10) -> dict:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline(settings, strategy="structure")
    questions = [json.loads(line) for line in settings.eval_questions_path.with_name("ppvs_questions.jsonl").open(encoding="utf-8") if line.strip()]
    metrics = evaluate_ppvs(pipeline.index, pipeline.embedder, questions, k=k)
    output = settings.data_dir / "eval_ppvs_results.json"
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"ППВС: n={metrics['n']} R@1={metrics['recall@1']:.3f} "
        f"R@5={metrics['recall@5']:.3f} MRR={metrics['mrr@10']:.3f}"
    )
    for item in metrics["per_question"]:
        if item["rank"] != 1:
            print(f"  промах: {item['q']} -> rank={item['rank']}")
    print(f"сохранено: {output}")
    return metrics


def print_table(results: dict) -> None:
    header = f"{'strategy':22} {'chunks':>7} {'tok_p50':>8} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}"
    print(header)
    print("-" * len(header))
    for name, metrics in results["strategies"].items():
        stats = metrics["chunk_stats"]
        print(
            f"{name:22} {stats['count']:>7} {stats['tokens_p50']:>8} "
            f"{metrics['recall@1']:>6.3f} {metrics['recall@5']:>6.3f} "
            f"{metrics['recall@10']:>6.3f} {metrics['mrr@10']:>6.3f}"
        )
