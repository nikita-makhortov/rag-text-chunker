from dataclasses import asdict, dataclass


@dataclass
class TextChunk:
    """Единица нарезанного текста (чанк) с метаданными для индексации в RAG."""

    source_id: str   # Идентификатор исходного документа (например, "doc_1")
    index: int       # Порядковый номер чанка внутри документа (начиная с 0)
    content: str     # Готовый текст чанка после склейки токенов
    unit_count: int  # Количество базовых единиц (слов или предложений) в чанке

    def to_dict(self) -> dict:
        # Преобразуем объект в обычный словарь для удобной выгрузки в JSON
        return asdict(self)


@dataclass
class BenchmarkMetrics:
    """Сводные метрики качества и памяти для конкретной стратегии чанкинга."""

    strategy_name: str         # Название стратегии для вывода в таблицу
    total_chunks: int          # Сколько всего чанков получилось из документа
    mean_units: float          # Среднее число слов/предложений в одном чанке
    stdev_units: float         # Разброс (стандартное отклонение) размера чанков
    mean_chars: float          # Средняя длина чанка в символах
    sentence_break_pct: float  # SSR: процент предложений, разорванных на стыках
    redundancy_ratio: float    # TRR: во сколько раз раздулся объём данных из-за overlap
