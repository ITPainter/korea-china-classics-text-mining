"""계량 문체론: Burrows의 Delta (3장)."""

from collections import Counter

import numpy as np
import pandas as pd


def relative_freqs(tokens: list[str], vocab: list[str]) -> np.ndarray:
    c = Counter(tokens)
    n = len(tokens) or 1
    return np.array([c[w] / n for w in vocab])


def burrows_delta(candidates: dict[str, list[str]], disputed: list[str], n_mfw: int = 100) -> pd.Series:
    """후보 작가별 토큰 목록과 논란 작품 토큰으로 Delta 거리를 계산한다. 작을수록 문체가 가깝다.

    빈도 상위 n_mfw개 단어(주로 기능어)의 상대 빈도를 z-점수로 바꾼 뒤,
    후보와 논란 작품 사이 z-점수 차이의 절댓값 평균을 구한다.
    """
    pooled = Counter()
    for toks in candidates.values():
        pooled.update(toks)
    vocab = [w for w, _ in pooled.most_common(n_mfw)]
    matrix = np.vstack([relative_freqs(t, vocab) for t in candidates.values()])
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    std[std == 0] = 1
    z_cand = (matrix - mean) / std
    z_test = (relative_freqs(disputed, vocab) - mean) / std
    delta = np.abs(z_cand - z_test).mean(axis=1)
    return pd.Series(delta, index=list(candidates), name="Delta").sort_values()


def outlier_chapters(docs: list[list[str]], n_mfw: int = 100) -> pd.Series:
    """회차 중 문체가 '평균 회차'에서 가장 먼 순서(Delta). docs는 회차별 토큰 목록, 번호는 1회부터."""
    pooled = Counter()
    for toks in docs:
        pooled.update(toks)
    vocab = [w for w, _ in pooled.most_common(n_mfw)]
    matrix = np.vstack([relative_freqs(t, vocab) for t in docs])
    std = matrix.std(axis=0)
    std[std == 0] = 1
    z = (matrix - matrix.mean(axis=0)) / std
    return pd.Series(np.abs(z).mean(axis=1), index=range(1, len(docs) + 1), name="Delta").sort_values(ascending=False)
