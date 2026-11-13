import statistics

from rag_chunker.models import BenchmarkMetrics, TextChunk


class ChunkEvaluator:
    """Расчёт количественных и структурных метрик качества нарезки для RAG."""

    @staticmethod
    def sentence_break_rate(
        reference_sentences: list[str], chunks: list[TextChunk]
    ) -> float:
        """
        SSR (Sentence Splitting Rate) — доля предложений из исходного текста,
        которые не сохранились целиком ни в одном из чанков (в процентах).
        Чем ниже показатель, тем меньше разорванных правил и фактов.
        """
        if not reference_sentences or not chunks:
            return 0.0

        broken_count = 0
        for sentence in reference_sentences:
            # Проверяем, встречается ли эталонное предложение целиком хотя бы в одном чанке
            is_intact = any(sentence in chunk.content for chunk in chunks)
            if not is_intact:
                broken_count += 1

        return round((broken_count / len(reference_sentences)) * 100, 2)

    @staticmethod
    def redundancy_ratio(chunks: list[TextChunk], source_unit_count: int) -> float:
        """
        TRR (Token Redundancy Ratio) — коэффициент избыточности данных из-за overlap.
        Показывает отношение суммы токенов во всех чанках к исходному числу токенов.
        Значение 1.00 = перекрытия нет; 1.48 = переплата +48% по памяти и токенам.
        """
        if not chunks or source_unit_count <= 0:
            return 0.0

        total_chunk_units = sum(chunk.unit_count for chunk in chunks)
        return round(total_chunk_units / source_unit_count, 2)

    def evaluate(
        self,
        strategy_name: str,
        chunks: list[TextChunk],
        reference_sentences: list[str],
        source_unit_count: int,
    ) -> BenchmarkMetrics:
        # Собираем полный набор метрик по списку готовых чанков
        if not chunks:
            return BenchmarkMetrics(strategy_name, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        unit_lengths = [chunk.unit_count for chunk in chunks]
        char_lengths = [len(chunk.content) for chunk in chunks]

        mean_units = round(statistics.mean(unit_lengths), 2)
        # Стандартное отклонение можно посчитать только если чанков минимум два
        stdev_units = (
            round(statistics.stdev(unit_lengths), 2) if len(unit_lengths) > 1 else 0.0
        )
        mean_chars = round(statistics.mean(char_lengths), 2)
        break_pct = self.sentence_break_rate(reference_sentences, chunks)
        redundancy = self.redundancy_ratio(chunks, source_unit_count)

        return BenchmarkMetrics(
            strategy_name=strategy_name,
            total_chunks=len(chunks),
            mean_units=mean_units,
            stdev_units=stdev_units,
            mean_chars=mean_chars,
            sentence_break_pct=break_pct,
            redundancy_ratio=redundancy,
        )

    @staticmethod
    def format_comparison(first: BenchmarkMetrics, second: BenchmarkMetrics) -> str:
        # Форматируем сравнение двух стратегий в ровную консольную таблицу
        header = f"{'Метрика':<36} | {first.strategy_name:<22} | {second.strategy_name:<22}"
        line = "-" * len(header)
        rows = [
            "=" * len(header),
            header,
            line,
            f"{'Всего чанков (total_chunks)':<36} | {first.total_chunks:<22} | {second.total_chunks:<22}",
            f"{'Средний размер (юнитов)':<36} | {first.mean_units:<22} | {second.mean_units:<22}",
            f"{'Станд. отклонение (stdev)':<36} | {first.stdev_units:<22} | {second.stdev_units:<22}",
            f"{'Средняя длина (символов)':<36} | {first.mean_chars:<22} | {second.mean_chars:<22}",
            f"{'Разрыв предложений (SSR)':<36} | {f'{first.sentence_break_pct}%':<22} | {f'{second.sentence_break_pct}%':<22}",
            f"{'Избыточность токенов (TRR)':<36} | {first.redundancy_ratio:<22} | {second.redundancy_ratio:<22}",
            "=" * len(header),
        ]
        return "\n".join(rows)
