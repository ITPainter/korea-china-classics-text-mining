"""텍스트 재사용과 전고(典故) 탐지 (5장).

조선 한시와 중국 고전 사이의 인용·변용을 찾는다.
한시는 글자를 바꾸거나 줄여 쓰는 경우가 많아 완전 일치 대신 글자 n-gram 겹침과
부분 정렬을 함께 쓴다.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

from .text import han_chars, normalize


def char_ngrams(text: str, n: int = 2) -> set[str]:
    chars = han_chars(normalize(text))
    return {"".join(chars[i : i + n]) for i in range(len(chars) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def longest_shared(a: str, b: str) -> str:
    """두 구절이 공유하는 가장 긴 연속 글자열."""
    ca, cb = "".join(han_chars(normalize(a))), "".join(han_chars(normalize(b)))
    m = SequenceMatcher(None, ca, cb, autojunk=False).find_longest_match(0, len(ca), 0, len(cb))
    return ca[m.a : m.a + m.size]


@dataclass
class Match:
    target_id: str
    target_line: str
    source_id: str
    source_line: str
    score: float
    shared: str


def find_reuse(
    targets: dict[str, list[str]],
    sources: dict[str, list[str]],
    n: int = 2,
    min_score: float = 0.15,
    min_shared: int = 3,
) -> list[Match]:
    """targets(예: 조선 한시)의 각 행을 sources(예: 중국 고전)의 각 행과 비교해 재사용 후보를 찾는다.

    후보는 '확정된 인용'이 아니다. 연구자가 직접 읽고 인용·변용·우연한 일치로 분류해야 한다.
    """
    index = [(sid, line, char_ngrams(line, n)) for sid, lines in sources.items() for line in lines]
    found = []
    for tid, lines in targets.items():
        for line in lines:
            grams = char_ngrams(line, n)
            for sid, sline, sgrams in index:
                if not grams & sgrams:
                    continue
                shared = longest_shared(line, sline)
                s = jaccard(grams, sgrams)
                if s >= min_score or len(shared) >= min_shared:
                    found.append(Match(tid, line, sid, sline, round(s, 3), shared))
    return sorted(found, key=lambda m: (len(m.shared), m.score), reverse=True)


def to_frame(matches: list[Match]):
    import pandas as pd

    return pd.DataFrame([m.__dict__ for m in matches])
