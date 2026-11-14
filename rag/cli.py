"""CLI: python -m rag.cli <команда>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chunking import PlenumChunker, build_chunks, chunk_stats
from .config import Settings
from .corpus import build_corpus, load_corpus
from .embeddings import Embedder
from .evaluate import load_questions, make_chunker, print_table, run_comparison, run_ppvs_evaluation
from .pipeline import RAGPipeline
from .ppvs import build_ppvs, load_ppvs
from .retriever import chunk_source
from .store import HybridIndex


def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def cmd_corpus(args: argparse.Namespace) -> None:
    settings = Settings()
    stats = build_corpus(settings, refresh=args.refresh, limit=args.limit)
    print(json.dumps({k: v for k, v in stats.items() if k != "failed"}, ensure_ascii=False, indent=2))
    if args.ppvs:
        ppvs_stats = build_ppvs(settings, refresh=args.refresh)
        print(json.dumps(ppvs_stats, ensure_ascii=False, indent=2))


def cmd_ppvs(args: argparse.Namespace) -> None:
    settings = Settings()
    stats = build_ppvs(settings, refresh=args.refresh)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_index(args: argparse.Namespace) -> None:
    settings = Settings()
    if args.overlap is not None:
        settings.chunk_overlap_parts = args.overlap
    if args.max_tokens:
        settings.chunk_max_tokens = args.max_tokens
    articles = [a for a in load_corpus(settings.corpus_path) if not a.repealed and a.elements]
    if not articles:
        raise SystemExit("Корпус пуст. Сначала: python -m rag.cli corpus")

    embedder = Embedder(
        settings.embedding_model,
        device=settings.device,
        batch_size=settings.batch_size,
        hf_home=settings.data_dir / "models" / "hf",
    )
    strategies = ["structure", "fixed"] if args.strategy == "both" else [args.strategy]
    for strategy in strategies:
        print(f"\n[index] {strategy}: строю чанки...")
        chunker = make_chunker(strategy, embedder, settings)
        chunks = build_chunks(articles, chunker)
        stats = chunk_stats(chunks)
        print(f"[index] {strategy} (УК РФ): {stats}")

        ppvs_path = settings.data_dir / "raw" / "ppvs.jsonl"
        if args.with_ppvs and strategy == "structure" and ppvs_path.exists():
            docs = load_ppvs(ppvs_path)
            plenum_chunker = PlenumChunker(
                count_tokens=embedder.count_tokens,
                max_tokens=settings.chunk_max_tokens,
                overlap_paras=settings.chunk_overlap_parts,
                sentence_overlap=settings.sentence_overlap,
            )
            plenum_chunks = []
            for doc in docs:
                plenum_chunks.extend(plenum_chunker.chunk_document(doc))
            print(f"[index] {strategy} (ППВС): {chunk_stats(plenum_chunks)}")
            chunks = chunks + plenum_chunks

        print(f"[index] {strategy}: считаю эмбеддинги ({len(chunks)} чанков)...")
        index = HybridIndex.build([c.to_dict() for c in chunks], embedder, show_progress=True)
        target = settings.index_dir(strategy)
        index.save(target)
        print(f"[index] {strategy}: сохранено в {target}")


def cmd_ask(args: argparse.Namespace) -> None:
    settings = Settings()
    if args.k:
        settings.top_k = args.k
    if args.max_tokens:
        settings.max_new_tokens = args.max_tokens
    if args.ctx:
        settings.n_ctx = args.ctx
    pipeline = RAGPipeline(settings, strategy=args.strategy)

    if args.question:
        _answer_once(pipeline, args)
        return

    print("Интерактивный режим. Пустая строка — выход.")
    while True:
        try:
            question = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        _answer_once(pipeline, args, question)


def _answer_once(pipeline: RAGPipeline, args: argparse.Namespace, question: str | None = None) -> None:
    question = question or args.question
    if args.json:
        answer = pipeline.ask(
            question,
            k=args.k,
            expand=args.expand,
            generate=not args.no_llm,
            stream=False,
        )
        print(
            json.dumps(
                {"question": question, "answer": answer.text, "sources": answer.sources},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    answer = pipeline.ask(
        question,
        k=args.k,
        expand=args.expand,
        generate=not args.no_llm,
        stream=True,
    )
    if args.no_llm:
        print(answer.text)
    print("\nИсточники:")
    for source in answer.sources:
        if source.get("act") == "ППВС":
            date = source.get("doc_date", "")
            if len(date) == 10 and date[4] == "-":
                year, month, day = date.split("-")
                date = f"{day}.{month}.{year}"
            label = (
                f"ППВС от {date} № {source.get('doc_number', '')} "
                f"«{source.get('title', '')}»"
            )
            if source.get("paragraph"):
                label += f", п. {source['paragraph']}"
        else:
            label = (
                f"ст. {source['article']} «{source['title']}»"
                f"{', ' + source['part'] if source['part'] else ''}"
            )
        print(f"  [S{source['n']}] {label} — {source['url']}")
    if answer.unsupported_articles:
        print(
            "\n[!] В ответе упомянуты статьи вне найденного контекста: "
            + ", ".join(f"ст. {a}" for a in answer.unsupported_articles)
        )


def cmd_eval(args: argparse.Namespace) -> None:
    settings = Settings()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    results = run_comparison(
        settings,
        strategies,
        k=args.k,
        limit_questions=args.limit,
    )
    print("\nСравнение стратегий чанкинга:")
    print_table(results)


def cmd_eval_ppvs(args: argparse.Namespace) -> None:
    settings = Settings()
    run_ppvs_evaluation(settings, k=args.k)


def cmd_bench(args: argparse.Namespace) -> None:
    from .bench import run_benchmark

    settings = Settings()
    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = settings.base_dir / questions_path
    questions = [
        json.loads(line)
        for line in questions_path.open(encoding="utf-8")
        if line.strip()
    ]
    if args.limit:
        questions = questions[: args.limit]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_benchmark(
        settings,
        questions,
        models,
        max_new_tokens=args.max_tokens,
        gpu_layers=args.gpu_layers,
    )


def cmd_search(args: argparse.Namespace) -> None:
    """Поиск без LLM: показать найденные нормы и их тексты."""
    settings = Settings()
    pipeline = RAGPipeline(settings, strategy=args.strategy)
    hits = pipeline.retrieve(args.query, k=args.k)
    for number, hit in enumerate(hits, 1):
        chunk = hit.chunk
        print(f"\n[S{number}] {chunk_source(chunk)}  score={hit.score:.4f}")
        print(chunk["text"][: args.chars])


def cmd_stats(args: argparse.Namespace) -> None:
    settings = Settings()
    if settings.corpus_path.exists():
        articles = load_corpus(settings.corpus_path)
        print(
            f"Корпус: {len(articles)} статей, "
            f"{sum(len(a.elements) for a in articles)} элементов, "
            f"утратили силу: {sum(a.repealed for a in articles)}"
        )
    else:
        print("Корпус не собран")
    ppvs_path = settings.data_dir / "raw" / "ppvs.jsonl"
    if ppvs_path.exists():
        docs = load_ppvs(ppvs_path)
        print(f"ППВС: {len(docs)} постановлений, {sum(len(d.paragraphs) for d in docs)} пунктов")
    else:
        print("ППВС не собраны")
    for strategy in ("structure", "structure-nooverlap", "fixed"):
        meta_path = settings.index_dir(strategy) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(
                f"Индекс {strategy}: {meta['count']} чанков, "
                f"{meta['embedding_model']}, dim={meta['dim']}, собран {meta['built_at']}"
            )
    if settings.eval_questions_path.exists():
        print(f"Вопросов для оценки: {len(load_questions(settings.eval_questions_path))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG по уголовному праву РФ: корпус, чанки, гибридный поиск, локальная LLM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_corpus = sub.add_parser("corpus", help="скачать и разобрать УК РФ")
    p_corpus.add_argument("--refresh", action="store_true", help="перекачать страницы")
    p_corpus.add_argument("--limit", type=int, default=None, help="только первые N статей")
    p_corpus.add_argument("--ppvs", action="store_true", help="собрать ещё и ППВС")
    p_corpus.set_defaults(func=cmd_corpus)

    p_ppvs = sub.add_parser("ppvs", help="собрать корпус постановлений Пленума ВС РФ")
    p_ppvs.add_argument("--refresh", action="store_true")
    p_ppvs.set_defaults(func=cmd_ppvs)

    p_index = sub.add_parser("index", help="построить чанки и векторный индекс")
    p_index.add_argument("--strategy", choices=["structure", "structure-nooverlap", "fixed", "both"], default="structure")
    p_index.add_argument("--overlap", type=int, default=None, help="сколько частей дублировать в соседнем чанке")
    p_index.add_argument("--max-tokens", type=int, default=None)
    p_index.add_argument("--with-ppvs", dest="with_ppvs", action="store_true", default=True)
    p_index.add_argument("--no-ppvs", dest="with_ppvs", action="store_false")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="задать вопрос")
    p_ask.add_argument("question", nargs="?", help="вопрос; без него — интерактивный режим")
    p_ask.add_argument("-k", type=int, default=None, help="сколько фрагментов подать в LLM")
    p_ask.add_argument("--strategy", default="structure")
    p_ask.add_argument("--expand", choices=["article", "none"], default="article", help="раскрывать ли чанк до всей статьи")
    p_ask.add_argument("--no-llm", action="store_true", help="только поиск, без генерации")
    p_ask.add_argument("--json", action="store_true", help="вывод в JSON (без стриминга)")
    p_ask.add_argument("--max-tokens", type=int, default=None, help="лимит новых токенов")
    p_ask.add_argument("--ctx", type=int, default=None, help="размер контекста LLM")
    p_ask.set_defaults(func=cmd_ask)

    p_search = sub.add_parser("search", help="только поиск, без LLM")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=6)
    p_search.add_argument("--chars", type=int, default=800)
    p_search.add_argument("--strategy", default="structure")
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="сравнить стратегии чанкинга")
    p_eval.add_argument(
        "--strategies",
        default="structure,structure-nooverlap,fixed",
        help="список через запятую",
    )
    p_eval.add_argument("--k", type=int, default=10)
    p_eval.add_argument("--limit", type=int, default=None, help="ограничить число вопросов")
    p_eval.set_defaults(func=cmd_eval)

    p_eval_ppvs = sub.add_parser("eval-ppvs", help="оценить поиск по постановлениям Пленума")
    p_eval_ppvs.add_argument("--k", type=int, default=10)
    p_eval_ppvs.set_defaults(func=cmd_eval_ppvs)

    p_bench = sub.add_parser("bench", help="сравнить локальные модели на одинаковом контексте")
    p_bench.add_argument("--models", default="yandex,saiga8", help="yandex,saiga8,saiga12,tiny или путь к .gguf")
    p_bench.add_argument("--questions", default="eval/bench_questions.jsonl")
    p_bench.add_argument("--max-tokens", type=int, default=300)
    p_bench.add_argument("--gpu-layers", type=int, default=None, help="переопределить n_gpu_layers")
    p_bench.add_argument("--limit", type=int, default=None)
    p_bench.set_defaults(func=cmd_bench)

    p_stats = sub.add_parser("stats", help="что уже собрано")
    p_stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> None:
    _setup_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
