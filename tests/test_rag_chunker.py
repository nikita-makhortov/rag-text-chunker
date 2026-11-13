import pytest

from rag_chunker import (
    ChunkEvaluator,
    SentenceSplitter,
    TextPreprocessor,
    WordSplitter,
)


def test_preprocessor_pipeline():
    # Проверяем очистку мусорных символов, схлопывание пробелов и сохранение дефисов
    processor = TextPreprocessor(keep_punctuation=True, lowercase=True)
    raw = "#\n  Bodor   A-3   мощность  $$1000$ Вт!  "
    assert processor.process(raw) == "bodor a-3 мощность 1000 вт!"

    # При keep_punctuation=False должны удаляться все знаки препинания и дефисы
    strict_processor = TextPreprocessor(keep_punctuation=False, lowercase=False)
    assert strict_processor.process("Bodor-A3#, скидка 100!") == "BodorA3 скидка 100"


def test_word_splitter_sliding_window():
    # Эталонная проверка скользящего окна: размер 3, перекрытие 1 -> шаг сдвига = 2
    splitter = WordSplitter(window_size=3, overlap=1)
    chunks = splitter.split("один два три четыре пять", source_id="doc_1")
    assert [c.content for c in chunks] == ["один два три", "три четыре пять"]
    assert [c.unit_count for c in chunks] == [3, 3]


def test_sentence_splitter_preserves_decimals():
    # Точка внутри десятичных дробей (0.02, 16.5) не должна разрывать предложение
    splitter = SentenceSplitter(window_size=2, overlap=1)
    text = "Точность 0.02 мм. Давление 16.5 бар! Готово?"
    chunks = splitter.split(text)
    assert [c.content for c in chunks] == [
        "Точность 0.02 мм. Давление 16.5 бар!",
        "Давление 16.5 бар! Готово?",
    ]


def test_invalid_window_parameters_raise():
    # Защита от бесконечного цикла: overlap >= window_size или неположительный размер окна
    with pytest.raises(ValueError):
        WordSplitter(window_size=3, overlap=3)
    with pytest.raises(ValueError):
        WordSplitter(window_size=0, overlap=0)
    with pytest.raises(ValueError):
        SentenceSplitter(window_size=4, overlap=-1)


def test_evaluator_metrics():
    # Проверяем расчёт SSR и TRR на тексте из 3 предложений (окно 2, перекрытие 1 -> 2 чанка по 2 предложения)
    text = "Первое предложение. Второе предложение. Третье предложение."
    sent_splitter = SentenceSplitter(window_size=2, overlap=1)
    ref_sentences = sent_splitter.extract_units(text)
    chunks = sent_splitter.split(text)

    evaluator = ChunkEvaluator()
    metrics = evaluator.evaluate("Sentence", chunks, ref_sentences, len(ref_sentences))
    assert metrics.total_chunks == 2
    assert metrics.sentence_break_pct == 0.0
    # В 2 чанках по 2 предложения суммарно 4 юнита при 3 исходных -> 4 / 3 = 1.33
    assert metrics.redundancy_ratio == 1.33
