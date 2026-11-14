"""코퍼스와 메타데이터 (2장)."""

from pathlib import Path

import pandas as pd

METADATA_COLUMNS = ["id", "title", "author", "country", "period", "year", "genre", "script", "source", "text"]


def load_folder(folder: str, metadata_csv: str | None = None) -> pd.DataFrame:
    """폴더의 .txt 파일을 읽고, 메타데이터 CSV가 있으면 파일 이름(id)으로 합친다."""
    rows = [{"id": p.stem, "text": p.read_text(encoding="utf-8")} for p in sorted(Path(folder).glob("*.txt"))]
    df = pd.DataFrame(rows, columns=["id", "text"])
    if metadata_csv:
        meta = pd.read_csv(metadata_csv, dtype={"id": str})
        df = meta.drop(columns=["text"], errors="ignore").merge(df, on="id", how="right")
    return df


def check_metadata(df: pd.DataFrame) -> list[str]:
    """빠진 메타데이터 열과 비어 있는 칸을 알려 준다."""
    problems = [f"열 없음: {c}" for c in METADATA_COLUMNS if c not in df.columns]
    for c in df.columns.intersection(METADATA_COLUMNS):
        n = int(df[c].isna().sum())
        if n:
            problems.append(f"{c}: 빈 칸 {n}개")
    return problems


def distribution(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """코퍼스 편향 점검용: 국가·시대 등 기준별 작품 수."""
    return df.groupby(by).size().rename("작품 수").reset_index()
