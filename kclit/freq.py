"""빈도, 키워드(로그우도), TF-IDF, 공기어 (3장)."""

import math
from collections import Counter
from itertools import combinations

import pandas as pd


def frequencies(docs: list[list[str]]) -> Counter:
    c = Counter()
    for tokens in docs:
        c.update(tokens)
    return c


def keyness(target: Counter, reference: Counter, min_count: int = 2) -> pd.DataFrame:
    """Dunning 로그우도(G2)로 대상 코퍼스에서 상대적으로 많이 쓰인 단어를 찾는다."""
    n1, n2 = sum(target.values()), sum(reference.values())
    rows = []
    for w, a in target.items():
        if a < min_count:
            continue
        b = reference.get(w, 0)
        e1 = n1 * (a + b) / (n1 + n2)
        e2 = n2 * (a + b) / (n1 + n2)
        g2 = 2 * (a * math.log(a / e1) + (b * math.log(b / e2) if b else 0))
        if a / n1 > b / n2:
            rows.append((w, a, b, g2))
    df = pd.DataFrame(rows, columns=["단어", "대상 빈도", "참조 빈도", "G2"])
    return df.sort_values("G2", ascending=False, ignore_index=True)


def tfidf(docs: list[list[str]], doc_ids: list[str] | None = None, top: int = 10) -> dict[str, list[tuple[str, float]]]:
    """문서별 TF-IDF 상위 단어. 이미 토큰화된 문서를 받는다."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer=lambda x: x)
    m = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    ids = doc_ids or [str(i) for i in range(len(docs))]
    out = {}
    for i, doc_id in enumerate(ids):
        row = m.getrow(i).toarray()[0]
        order = row.argsort()[::-1][:top]
        out[doc_id] = [(vocab[j], round(float(row[j]), 4)) for j in order if row[j] > 0]
    return out


def cooccurrence(units: list[list[str]], min_count: int = 1) -> Counter:
    """한 단위(문장·행·문서) 안에 함께 나온 단어 쌍을 센다."""
    c = Counter()
    for tokens in units:
        for a, b in combinations(sorted(set(tokens)), 2):
            c[(a, b)] += 1
    return Counter({k: v for k, v in c.items() if v >= min_count})


def cooccurrence_graph(pairs: Counter):
    import networkx as nx

    g = nx.Graph()
    for (a, b), w in pairs.items():
        g.add_edge(a, b, weight=w)
    return g


def term_counts(chapters: list[str], terms: list[str]) -> pd.DataFrame:
    """회차 x 낱말 표. 낱말이 글자 그대로 몇 번 나오는지 센다(호칭 사전이 필요 없는 단순 검색)."""
    return pd.DataFrame({t: [c.count(t) for c in chapters] for t in terms}, index=range(1, len(chapters) + 1))


def filter_common(docs: list[list[str]], min_docs: int = 3, max_ratio: float = 0.4) -> list[list[str]]:
    """너무 드문 말(min_docs회 미만)과 어디에나 나오는 말(전체 문서의 max_ratio 초과)을 뺀다. 토픽 모델링 전처리용."""
    n = len(docs)
    df = Counter(w for tokens in docs for w in set(tokens))
    keep = {w for w, c in df.items() if c >= min_docs and c / n <= max_ratio}
    return [[w for w in tokens if w in keep] for tokens in docs]
