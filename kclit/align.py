"""원문-번역문 병렬 정렬 (6장).

다국어 문장 임베딩의 유사도 행렬 위에서 순서를 지키는 동적 계획법으로 1:1, 1:2, 2:1 짝을 찾는다.
짝이 없는 문장은 '생략' 또는 '첨가' 후보로 표시된다.
"""

import numpy as np


def align(sim: np.ndarray, skip_penalty: float = 0.2, merge_penalty: float = 0.25) -> list[tuple[list[int], list[int], float]]:
    """sim[i, j] = 원문 i와 번역문 j의 유사도. (원문 번호들, 번역문 번호들, 점수) 목록을 돌려준다."""
    n, m = sim.shape
    best = np.full((n + 1, m + 1), -np.inf)
    back = {}
    best[0, 0] = 0.0
    moves = [(1, 1), (1, 0), (0, 1), (2, 1), (1, 2)]
    for i in range(n + 1):
        for j in range(m + 1):
            if best[i, j] == -np.inf:
                continue
            for di, dj in moves:
                a, b = i + di, j + dj
                if a > n or b > m:
                    continue
                if di == 0 or dj == 0:
                    gain = -skip_penalty
                else:
                    gain = sim[i:a, j:b].mean() - (merge_penalty if di + dj == 3 else 0)
                if best[i, j] + gain > best[a, b]:
                    best[a, b] = best[i, j] + gain
                    back[(a, b)] = (i, j)
    pairs, cell = [], (n, m)
    while cell != (0, 0):
        i, j = back[cell]
        a, b = cell
        src, tgt = list(range(i, a)), list(range(j, b))
        score = float(sim[i:a, j:b].mean()) if src and tgt else 0.0
        pairs.append((src, tgt, round(score, 3)))
        cell = (i, j)
    return pairs[::-1]


def label(pair: tuple[list[int], list[int], float], low: float = 0.4) -> str:
    src, tgt, score = pair
    if not tgt:
        return "생략"
    if not src:
        return "첨가"
    if score < low:
        return "의역 또는 오역 의심"
    return "대응"
