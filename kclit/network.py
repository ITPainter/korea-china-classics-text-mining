"""인물망과 문인 교류망 (5장)."""

from itertools import combinations

import networkx as nx
import pandas as pd


def character_network(scenes: list[str], characters: dict[str, list[str]]) -> nx.Graph:
    """같은 장면(문단)에 함께 나온 인물을 잇는다.

    characters: {'대표 이름': ['별칭1', '별칭2', ...]} — 소설 속 호칭이 여럿이라 별칭을 함께 센다.
    """
    g = nx.Graph()
    g.add_nodes_from(characters)
    for scene in scenes:
        present = sorted(name for name, aliases in characters.items() if any(a in scene for a in [name, *aliases]))
        for a, b in combinations(present, 2):
            w = g.get_edge_data(a, b, {"weight": 0})["weight"]
            g.add_edge(a, b, weight=w + 1)
    return g


def exchange_network(records: pd.DataFrame, source: str = "from", target: str = "to") -> nx.DiGraph:
    """사행록·증답시 기록(누가 누구에게)으로 방향이 있는 교류망을 만든다."""
    g = nx.DiGraph()
    for (a, b), n in records.groupby([source, target]).size().items():
        g.add_edge(a, b, weight=int(n))
    return g


def strong_edges(g: nx.Graph, min_weight: int = 1) -> nx.Graph:
    """함께 나온 횟수가 min_weight 미만인 선을 걸러 낸다. 인물이 많으면 거의 모든 쌍이 이어지기 때문에 필요하다."""
    h = nx.Graph() if not g.is_directed() else nx.DiGraph()
    h.add_nodes_from(g.nodes)
    h.add_edges_from((a, b, d) for a, b, d in g.edges(data=True) if d.get("weight", 1) >= min_weight)
    return h


def centrality_table(g: nx.Graph, min_weight: int = 1) -> pd.DataFrame:
    """연결 중심성(얼마나 많이 이어졌나), 매개 중심성(집단을 잇는 다리 역할), 가중 연결(함께 나온 횟수의 합).

    min_weight를 올리면 약한 연결을 뺀 관계도로 중심성을 계산한다. 가중 연결은 걸러 내기 전의 값이다.
    """
    h = strong_edges(g, min_weight)
    df = pd.DataFrame(
        {
            "연결 중심성": nx.degree_centrality(h),
            "매개 중심성": nx.betweenness_centrality(h, weight=None),
            "가중 연결": dict(g.degree(weight="weight")),
        }
    )
    df[["연결 중심성", "매개 중심성"]] = df[["연결 중심성", "매개 중심성"]].round(3)
    return df.sort_values(["가중 연결", "매개 중심성"], ascending=False)


def draw(g: nx.Graph, ax=None, seed: int = 0):
    """한글·한자 라벨을 쓰려면 Colab에서 setup_korean_font()를 먼저 실행한다."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    pos = nx.spring_layout(g, seed=seed)
    widths = [g[u][v].get("weight", 1) for u, v in g.edges()]
    nx.draw_networkx(g, pos, ax=ax, node_color="#cfe0f3", edge_color="#888", width=widths, font_family=plt.rcParams["font.family"])
    ax.axis("off")
    return ax
