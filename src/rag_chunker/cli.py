import argparse
import json
from pathlib import Path

from rag_chunker.evaluator import ChunkEvaluator
from rag_chunker.preprocessor import TextPreprocessor
from rag_chunker.splitters import SentenceSplitter, WordSplitter


def run_split(args: argparse.Namespace) -> None:
    """Команда split: очищает корпус документов и сохраняет чанки в JSON."""
    raw_text = Path(args.input).read_text(encoding="utf-8")
    preprocessor = TextPreprocessor(keep_punctuation=True, lowercase=True)

    # Выбираем нужный сплиттер в зависимости от флага --strategy
    splitter = (
        SentenceSplitter(args.size, args.overlap)
        if args.strategy == "sentence"
        else WordSplitter(args.size, args.overlap)
    )

    all_chunks = []
    # Документы внутри одного файла разделяются маркером "---"
    for idx, doc in enumerate(raw_text.split("---"), start=1):
        if not doc.strip():
            continue
        cleaned = preprocessor.process(doc)
        chunks = splitter.split(cleaned, source_id=f"doc_{idx}")
        all_chunks.extend(chunk.to_dict() for chunk in chunks)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Готово: сохранено {len(all_chunks)} чанков в {output_path}")


def run_benchmark(args: argparse.Namespace) -> None:
    """Команда benchmark: сравнивает пословную и синтаксическую нарезку на одном документе."""
    raw_text = Path(args.input).read_text(encoding="utf-8")
    first_doc = raw_text.split("---")[0]

    preprocessor = TextPreprocessor(keep_punctuation=True, lowercase=True)
    cleaned = preprocessor.process(first_doc)

    # Стратегия A: 3 предложения на чанк с перекрытием в 1 предложение
    sent_splitter = SentenceSplitter(window_size=3, overlap=1)
    # Стратегия B: 100 слов на чанк с перекрытием в 20 слов
    word_splitter = WordSplitter(window_size=100, overlap=20)

    # Эталонные списки юнитов нужны для расчёта метрик SSR и TRR
    ref_sentences = sent_splitter.extract_units(cleaned)
    ref_words = word_splitter.extract_units(cleaned)

    sent_chunks = sent_splitter.split(cleaned, source_id="benchmark")
    word_chunks = word_splitter.split(cleaned, source_id="benchmark")

    evaluator = ChunkEvaluator()
    sent_metrics = evaluator.evaluate(
        "По предложениям (3/1)", sent_chunks, ref_sentences, len(ref_sentences)
    )
    word_metrics = evaluator.evaluate(
        "По словам (100/20)", word_chunks, ref_sentences, len(ref_words)
    )

    print("\n" + evaluator.format_comparison(sent_metrics, word_metrics) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-chunker",
        description="CLI-утилита для очистки текста, чанкинга и оценки метрик в RAG-пайплайнах",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Подкоманда нарезки корпуса в JSON
    split_cmd = subparsers.add_parser("split", help="Нарезать документы на чанки в JSON")
    split_cmd.add_argument("--input", default="data/sample_documents.txt", help="Путь к входному .txt")
    split_cmd.add_argument("--output", default="data/chunks.json", help="Путь для сохранения JSON")
    split_cmd.add_argument(
        "--strategy", choices=["word", "sentence"], default="word", help="Стратегия нарезки"
    )
    split_cmd.add_argument("--size", type=int, default=50, help="Размер окна (в словах или предложениях)")
    split_cmd.add_argument("--overlap", type=int, default=10, help="Размер перекрытия между чанками")
    split_cmd.set_defaults(func=run_split)

    # Подкоманда запуска сравнительного бенчмарка
    bench_cmd = subparsers.add_parser(
        "benchmark", help="Сравнить стратегии чанкинга и посчитать SSR/TRR"
    )
    bench_cmd.add_argument("--input", default="data/large_document.txt", help="Путь к документу")
    bench_cmd.set_defaults(func=run_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
