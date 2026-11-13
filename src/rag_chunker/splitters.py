import re

from rag_chunker.models import TextChunk


class BaseSplitter:
    """Базовый класс нарезки текста скользящим окном с перекрытием (overlap)."""

    def __init__(self, window_size: int, overlap: int) -> None:
        # Проверяем корректность параметров окна:
        # если overlap >= window_size, шаг сдвига станет <= 0 и цикл уйдёт в бесконечность
        if window_size <= 0 or overlap < 0 or overlap >= window_size:
            raise ValueError(
                f"Некорректные параметры окна: window_size={window_size}, overlap={overlap}"
            )
        self.window_size = window_size
        self.overlap = overlap

    @property
    def stride(self) -> int:
        # Шаг, на который сдвигается окно на каждой итерации
        return self.window_size - self.overlap

    def extract_units(self, text: str) -> list[str]:
        # Метод переопределяется в наследниках (по словам или по предложениям)
        raise NotImplementedError("Нужно реализовать extract_units в дочернем классе")

    def _slice_windows(self, units: list[str]) -> list[list[str]]:
        # Нарезаем плоский список токенов на пересекающиеся окна
        if not units:
            return []

        windows: list[list[str]] = []
        cursor = 0

        while True:
            # Берём срез длиной window_size от текущей позиции курсора
            current_window = units[cursor : cursor + self.window_size]
            windows.append(current_window)

            # Если текущий срез уже дошёл до конца списка — выходим,
            # чтобы не плодить лишние мелкие хвосты из перекрытия
            if cursor + self.window_size >= len(units):
                break

            # Сдвигаем окно вперёд с учётом перекрытия
            cursor += self.stride

        return windows

    def split(self, text: str, source_id: str = "doc_1") -> list[TextChunk]:
        # Полный цикл: разбиваем текст на юниты -> режем окнами -> упаковываем в TextChunk
        if not text.strip():
            return []

        units = self.extract_units(text)
        windows = self._slice_windows(units)

        chunks: list[TextChunk] = []
        for idx, window in enumerate(windows):
            chunks.append(
                TextChunk(
                    source_id=source_id,
                    index=idx,
                    content=" ".join(window),
                    unit_count=len(window),
                )
            )
        return chunks


class WordSplitter(BaseSplitter):
    """Разбивает текст на окна фиксированного размера по словам."""

    def extract_units(self, text: str) -> list[str]:
        units: list[str] = []
        for raw_token in text.split():
            # Срезаем знаки препинания по краям слова, но не трогаем дефис внутри слова
            cleaned = raw_token.strip("!?.,-")
            if cleaned:
                units.append(cleaned)
        return units


class SentenceSplitter(BaseSplitter):
    """Разбивает текст на окна по законченным предложениям."""

    def extract_units(self, text: str) -> list[str]:
        if not text.strip():
            return []

        # Регулярка с lookbehind (?<=[.!?]): делим только по пробелам после знака конца предложения.
        # Благодаря этому десятичные дроби вроде "0.02 мм" или "16.5 бар" не разрываются на части
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]
