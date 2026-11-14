"""이 책의 두 작품과 『장자』 원문을 위키문헌에서 받아 와 보관한다 (2장).

세 작품 모두 저작권 보호 기간이 끝난 고전이다. 한 번 받은 회차는 cache_dir에 저장해 두고 다시 받지 않는다.
처음 『서유기』 100회를 받을 때는 요청 사이에 2초씩 쉬어서 4~5분쯤 걸린다.
"""

from pathlib import Path

import pandas as pd

from .collect import fetch_wikisource

XIYOU_COUNT = 100
GU_COUNT = 16


def _chapters(title_pattern: str, file_pattern: str, numbers, cache_dir: str, progress: bool) -> list[str]:
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    numbers = list(numbers)
    texts = []
    for k, n in enumerate(numbers, 1):
        path = folder / file_pattern.format(n=n)
        if not (path.exists() and path.stat().st_size > 0):
            path.write_text(fetch_wikisource(title_pattern.format(n=n), "zh"), encoding="utf-8")
            if progress and (k % 10 == 0 or k == len(numbers)):
                print(f"  {k}/{len(numbers)}회 받음")
        texts.append(path.read_text(encoding="utf-8"))
    return texts


def xiyou(cache_dir: str = "data", chapters=None, progress: bool = True) -> list[str]:
    """吳承恩 『西遊記』 회차별 원문 목록(백화문). chapters로 일부 회차(예: range(1, 21))만 받을 수 있다."""
    return _chapters("西遊記/第{n:03d}回", "xiyou_{n:03d}.txt", chapters or range(1, XIYOU_COUNT + 1), cache_dir, progress)


def gu(cache_dir: str = "data", chapters=None, progress: bool = True) -> list[str]:
    """金萬重 『九雲夢』 회차별 한문 원문 목록(16회)."""
    return _chapters("九雲夢/第{n:02d}回", "gu_{n:02d}.txt", chapters or range(1, GU_COUNT + 1), cache_dir, progress)


def qiwulun(cache_dir: str = "data") -> str:
    """『莊子』 「齊物論」 원문. 『구운몽』 16회가 끌어온 '호접지몽'이 들어 있다."""
    path = Path(cache_dir) / "zhuangzi_qiwulun.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.stat().st_size > 0):
        path.write_text(fetch_wikisource("莊子/齊物論", "zh"), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def corpus_frame(xiyou_chapters: list[str] | None = None, gu_chapters: list[str] | None = None) -> pd.DataFrame:
    """회차 하나를 한 행으로 하는 코퍼스 표(이름표 + 본문). 연도는 판본·학설에 따라 달라 비워 둔다."""
    rows = []
    for n, t in enumerate(xiyou_chapters or [], 1):
        rows.append(("xiyou_%03d" % n, "西遊記 第%d回" % n, "吳承恩", "중국", "明", None, "소설", "백화문", "위키문헌", t))
    for n, t in enumerate(gu_chapters or [], 1):
        rows.append(("gu_%02d" % n, "九雲夢 第%d回" % n, "金萬重", "한국", "朝鮮", None, "소설", "한문", "위키문헌", t))
    cols = ["id", "title", "author", "country", "period", "year", "genre", "script", "source", "text"]
    return pd.DataFrame(rows, columns=cols)
