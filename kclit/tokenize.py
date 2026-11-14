"""한국어 형태소 분석, 중국어 분사, 고전 한문 글자 단위 분할 (2장)."""

from functools import lru_cache

from .text import detect_script, han_chars, normalize

KOREAN_CONTENT_TAGS = ("NNG", "NNP", "VV", "VA", "XR", "SH")


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def korean(text: str, tags: tuple = KOREAN_CONTENT_TAGS) -> list[str]:
    """Kiwi로 형태소를 나누고 내용어만 남긴다. 용언은 기본형에 '다'를 붙인다."""
    out = []
    for tok in _kiwi().tokenize(text):
        if tok.tag not in tags:
            continue
        out.append(tok.form + "다" if tok.tag in ("VV", "VA") else tok.form)
    return out


def add_chinese_words(words) -> None:
    """작품 고유의 인명·지명을 jieba 사용자 사전에 등록한다."""
    import jieba

    for w in words:
        jieba.add_word(w)


def chinese(text: str, min_len: int = 1) -> list[str]:
    """jieba로 현대 중국어·백화문을 단어 단위로 나눈다."""
    import jieba

    return [w for w in jieba.lcut(text) if han_chars(w) and len(w) >= min_len]


def classical(text: str) -> list[str]:
    """고전 한문은 글자 하나를 분석 단위로 삼는다."""
    return han_chars(normalize(text))


def tokenize(text: str, lang: str | None = None) -> list[str]:
    """lang을 지정하지 않으면 문자 비율로 판별한다. 'zh'는 jieba, 'classical'은 글자 단위."""
    lang = lang or detect_script(text)
    if lang in ("ko", "mixed"):
        return korean(text)
    if lang == "classical":
        return classical(text)
    if lang == "zh":
        return chinese(text)
    raise ValueError(f"지원하지 않는 언어입니다: {lang}")
