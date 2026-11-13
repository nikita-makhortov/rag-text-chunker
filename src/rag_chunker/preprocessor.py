import re


class TextPreprocessor:
    """Конвейер очистки и нормализации сырого текста перед токенизацией."""

    def __init__(self, keep_punctuation: bool = True, lowercase: bool = True) -> None:
        # Флаги позволяют гибко настраивать очистку под разные задачи:
        # например, для нарезки по предложениям пунктуацию обязательно оставляем
        self.keep_punctuation = keep_punctuation
        self.lowercase = lowercase

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        # Заменяем любые последовательности пробелов, табов и переносов строк на один пробел
        return re.sub(r"\s+", " ", text).strip()

    def _filter_noise(self, text: str) -> str:
        # Удаляем мусорные спецсимволы (#, $, @, % и т.д.), оставляя буквы и цифры.
        # При keep_punctuation=True сохраняем базовые знаки препинания и дефис,
        # чтобы не склеивать технические маркировки вроде "Bodor-A3" и не ломать границы фраз
        if self.keep_punctuation:
            return re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9.,!?\s\-]", "", text)
        return re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]", "", text)

    def process(self, raw_text: str) -> str:
        # Последовательно прогоняем текст через все этапы конвейера
        text = self._collapse_whitespace(raw_text)
        text = self._filter_noise(text)
        if self.lowercase:
            text = text.lower()
        # Финальная зачистка пробелов на случай, если после удаления спецсимволов появились двойные пробелы
        return self._collapse_whitespace(text)
