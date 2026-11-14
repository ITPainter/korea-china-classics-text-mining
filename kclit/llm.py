"""생성형 AI(Claude) 활용: 번역 초안, 전고 주석 후보, AI 활용 기록 (6장).

Colab에서는 왼쪽 열쇠 아이콘(보안 비밀)에 ANTHROPIC_API_KEY를 등록하고 load_api_key()를 실행한다.
키를 노트북 셀에 직접 적지 않는다.
"""

import csv
import datetime as dt
import os
from pathlib import Path

MODEL = "claude-opus-5-5"

TRANSLATE_SYSTEM = (
    "당신은 한문과 고전 중국어를 한국어로 옮기는 번역 보조자입니다. "
    "원문의 구절 순서를 지켜 직역에 가깝게 옮기고, 확신이 없는 글자나 전고는 [?]로 표시합니다. "
    "출전을 지어내지 말고, 모르면 모른다고 씁니다."
)

ANNOTATE_SYSTEM = (
    "당신은 한시의 전고(典故)를 찾는 연구 보조자입니다. "
    "각 후보마다 원문 구절, 추정 출전(서명·편명), 근거를 적습니다. "
    "출전이 확실하지 않으면 '불확실'이라고 쓰고, 존재 여부를 확인할 수 없는 문헌은 제시하지 않습니다. "
    "결과는 연구자가 원문 데이터베이스에서 직접 확인할 후보 목록입니다."
)


def load_api_key() -> None:
    """Colab 보안 비밀에서 키를 읽어 환경 변수로 등록한다. Colab 밖에서는 아무것도 하지 않는다."""
    try:
        from google.colab import userdata
    except ImportError:
        return
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")


def ask(prompt: str, system: str, max_tokens: int = 16000, effort: str = "high", client=None) -> str:
    """Claude에 한 번 질문하고 본문 텍스트를 돌려준다. client를 넘기면 그것을 쓴다(테스트용)."""
    import anthropic

    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("모델이 요청을 거절했습니다. 프롬프트를 확인하세요.")
    return "".join(b.text for b in response.content if b.type == "text")


def translate_draft(text: str, client=None) -> str:
    return ask(f"다음 원문을 번역해 주세요.\n\n<원문>\n{text}\n</원문>", TRANSLATE_SYSTEM, client=client)


def allusion_candidates(poem: str, client=None) -> str:
    return ask(f"다음 한시의 전고 후보를 찾아 주세요.\n\n<한시>\n{poem}\n</한시>", ANNOTATE_SYSTEM, client=client)


def log_use(path: str, task: str, prompt: str, verified_by: str, note: str = "") -> None:
    """AI 활용 기록표(CSV)에 한 줄을 덧붙인다: 날짜, 모델, 작업, 프롬프트, 검증 방법, 메모."""
    p = Path(path)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["날짜", "모델", "작업", "프롬프트", "검증 방법", "메모"])
        w.writerow([dt.date.today().isoformat(), MODEL, task, prompt, verified_by, note])
