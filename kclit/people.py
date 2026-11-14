"""인물 호칭 사전과 호칭 집계 (2~3장, 5장).

한 인물이 여러 이름으로 불리는 작품에서는 '孫悟空' 하나만 검색하면 대부분을 놓친다.
호칭을 사람별로 묶은 사전을 만들고, 겹치는 호칭(예: '孫行者' 안의 '行者')은 긴 것부터 한 번만 센다.
"""

import re

import pandas as pd

XIYOU_GROUPS = {
    "손오공": ["孫悟空", "齊天大聖", "孫行者", "孫大聖", "美猴王", "石猴", "弼馬溫", "老孫", "悟空", "行者", "大聖", "猴王", "猴子"],
    "삼장": ["唐三藏", "唐僧", "三藏", "唐長老", "長老"],
    "저팔계": ["豬八戒", "豬悟能", "八戒", "悟能", "呆子"],
    "사오정": ["沙和尚", "沙悟淨", "沙僧", "悟淨"],
}

GU_GROUPS = {
    "양소유": ["楊少遊", "楊尚書", "楊生", "楊丞相", "丞相", "相公", "少遊"],
    "진채봉": ["秦彩鳳", "秦氏", "秦女", "秦小姐"],
    "계섬월": ["桂蟾月", "蟾月", "桂氏"],
    "적경홍": ["狄驚鴻", "驚鴻", "狄氏"],
    "정경패": ["鄭瓊貝", "鄭小姐", "鄭女", "鄭氏", "瓊貝", "英陽公主", "英陽"],
    "가춘운": ["賈春雲", "春雲", "春娘", "賈氏"],
    "난양공주": ["蘭陽公主", "蘭陽", "李簫和"],
    "심요연": ["沈嫋煙", "嫋煙", "沈氏"],
    "백능파": ["白凌波", "凌波", "白氏"],
}


def _matcher(groups: dict[str, list[str]]):
    pairs = sorted(((a, g) for g, aliases in groups.items() for a in aliases), key=lambda x: -len(x[0]))
    pattern = re.compile("|".join(re.escape(a) for a, _ in pairs))
    return pattern, dict(pairs)


def count_names(chapters: list[str], groups: dict[str, list[str]]) -> pd.DataFrame:
    """회차 x 인물 표. 인물마다 모든 호칭의 등장 횟수를 합친다. 행 번호는 1회부터 시작한다."""
    pattern, owner = _matcher(groups)
    rows = []
    for text in chapters:
        row = dict.fromkeys(groups, 0)
        for m in pattern.finditer(text):
            row[owner[m.group()]] += 1
        rows.append(row)
    return pd.DataFrame(rows, index=range(1, len(chapters) + 1))


def alias_table(chapters: list[str], aliases: list[str]) -> pd.DataFrame:
    """한 인물의 호칭별 횟수와 비중. 겹치는 호칭은 긴 것부터 한 번만 센다."""
    pattern, _ = _matcher({"x": aliases})
    counts = {a: 0 for a in aliases}
    for text in chapters:
        for m in pattern.finditer(text):
            counts[m.group()] += 1
    s = pd.Series(counts).sort_values(ascending=False)
    return pd.DataFrame({"횟수": s, "비중": (s / s.sum()).round(3)})


def leaders(counts: pd.DataFrame) -> pd.Series:
    """회마다 가장 자주 나온 인물. 아무도 나오지 않은 회는 건너뛴다."""
    named = counts[counts.sum(axis=1) > 0]
    return named.idxmax(axis=1)


def absent(counts: pd.DataFrame, person: str) -> list[int]:
    """그 인물의 호칭이 한 번도 나오지 않는 회."""
    return counts.index[counts[person] == 0].tolist()
