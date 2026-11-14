"""문자 정규화, 문자 체계 판별, 문장 분리 (2장)."""

import re
import unicodedata
from functools import lru_cache

from opencc import OpenCC

# 한국 한문 자료에 자주 나오는 이체자 → OpenCC 번체 표준형
VARIANTS = {
    "爲": "為",
    "眞": "真",
    "竝": "並",
    "敎": "教",
    "靑": "青",
    "淸": "清",
}

_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_KANA = re.compile(r"[぀-ヿ]")
_SENT_END = re.compile(r"(?<=[。！？!?\.])\s*")


@lru_cache(maxsize=None)
def _converter(config: str) -> OpenCC:
    return OpenCC(config)


def normalize(text: str, variants: dict | None = None) -> str:
    """NFC 정규화 후 이체자를 대표 글자로 바꾼다.

    NFC는 한국 자료에 흔한 CJK 호환 한자(U+F900대)를 통합 한자로 바꿔 준다.
    """
    text = unicodedata.normalize("NFC", text)
    table = VARIANTS if variants is None else variants
    return "".join(table.get(ch, ch) for ch in text)


def to_simplified(text: str) -> str:
    return _converter("t2s").convert(text)


def to_traditional(text: str) -> str:
    return _converter("s2t").convert(text)


def script_profile(text: str) -> dict:
    """한글·한자·가나 비율을 센다."""
    counted = {
        "hangul": len(_HANGUL.findall(text)),
        "han": len(_HAN.findall(text)),
        "kana": len(_KANA.findall(text)),
    }
    total = sum(counted.values()) or 1
    return {k: v / total for k, v in counted.items()}


def detect_script(text: str) -> str:
    """'ko'(한글 위주), 'mixed'(국한문), 'zh'(한자 위주), 'ja', 'other' 중 하나."""
    p = script_profile(text)
    if p["kana"] > 0.1:
        return "ja"
    if p["hangul"] >= 0.7:
        return "ko"
    if p["han"] >= 0.9:
        return "zh"
    if p["hangul"] > 0 and p["han"] > 0:
        return "mixed"
    return "other"


def split_sentences(text: str) -> list[str]:
    """문장 부호 기준 분리. 부호가 없는 고전 한문은 줄 단위로 나눈다."""
    parts = []
    for line in text.splitlines():
        parts.extend(s.strip() for s in _SENT_END.split(line))
    return [s for s in parts if s]


def han_chars(text: str) -> list[str]:
    return _HAN.findall(text)
