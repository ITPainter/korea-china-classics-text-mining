"""생성형 AI 활용: 질문 문장 만들기, AI가 인용한 구절 검증, AI 활용 기록 (6장).

API 키가 필요 없다. 이 모듈은 AI를 직접 호출하지 않는다. 질문 문장(프롬프트)을 만들어 주면
Claude, ChatGPT, Gemini 같은 챗봇의 대화창에 붙여 넣어 답을 받고, 그 답을 노트북에 붙여 넣어 검증한다.
"""

import csv
import datetime as dt
import re
from pathlib import Path

TRANSLATE_SYSTEM = (
    "당신은 한문과 고전 중국어를 한국어로 옮기는 번역 보조자입니다. "
    "원문의 구절 순서를 지켜 직역에 가깝게 옮기고, 확신이 없는 글자나 전고는 [?]로 표시합니다. "
    "출전을 지어내지 말고, 모르면 모른다고 씁니다."
)

ANNOTATE_SYSTEM = (
    "당신은 고전 구절의 전고(典故)와 출전을 찾는 연구 보조자입니다. "
    "각 후보마다 원문 구절, 추정 출전(서명·편명), 근거를 적습니다. "
    "출전이 확실하지 않으면 '불확실'이라고 쓰고, 존재 여부를 확인할 수 없는 문헌은 제시하지 않습니다. "
    "결과는 연구자가 원문 데이터베이스에서 직접 확인할 후보 목록입니다."
)


def translate_prompt(text: str) -> str:
    """번역 초안을 요청하는 질문 문장. 챗봇 대화창에 붙여 넣는다."""
    return f"{TRANSLATE_SYSTEM}\n\n다음 원문을 번역해 주세요.\n\n<원문>\n{text}\n</원문>"


def allusion_prompt(text: str) -> str:
    """출전 후보를 요청하는 질문 문장. 챗봇 대화창에 붙여 넣는다."""
    return f"{ANNOTATE_SYSTEM}\n\n다음 구절의 출전 후보를 찾아 주세요.\n\n<구절>\n{text}\n</구절>"


def _han(s: str) -> str:
    return "".join(re.findall(r"[㐀-鿿]", s))


def check_quotes(answer: str, source: str, min_len: int = 5):
    """AI의 답에 적힌 한자 구절(min_len자 이상)이 원문에 그대로 있는지 찾아본다.

    글자열이 있는지만 확인한다. 뜻이 맞는지, 출전이 맞는지는 사람이 원문을 읽어 판단한다.
    """
    import pandas as pd

    base = _han(source)
    rows, seen = [], set()
    for run in re.split(r"[^㐀-鿿]+", answer):
        if len(run) >= min_len and run not in seen:
            seen.add(run)
            rows.append({"구절": run, "원문에 있음": run in base})
    return pd.DataFrame(rows, columns=["구절", "원문에 있음"])


def log_use(path: str, task: str, prompt: str, verified_by: str, note: str = "", model: str = "") -> None:
    """AI 활용 기록표(CSV)에 한 줄을 덧붙인다: 날짜, 모델, 작업, 프롬프트, 검증 방법, 메모."""
    p = Path(path)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["날짜", "모델", "작업", "프롬프트", "검증 방법", "메모"])
        w.writerow([dt.date.today().isoformat(), model, task, prompt, verified_by, note])
