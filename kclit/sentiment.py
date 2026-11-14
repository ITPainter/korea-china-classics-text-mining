"""정서 분석과 서사 정서 곡선 (4장).

모델 목록과 언어별 라우팅은 저자의 sentiment-analysis-multi-language(MIT) 프로젝트에서 가져왔다.
이 모델들은 금융 문장·상품 리뷰로 학습되어 문학 텍스트에는 맞지 않을 수 있다.
4장은 이 '도메인 불일치'를 직접 확인하는 실습으로 시작한다.
"""

from functools import lru_cache

import numpy as np

MODELS = {
    "ko": "matthewburke/korean_sentiment",
    "zh": "uer/roberta-base-finetuned-jd-binary-chinese",
}


@lru_cache(maxsize=4)
def _pipeline(model_name: str):
    from transformers import pipeline

    return pipeline("text-classification", model=model_name, truncation=True)


def _signed(result: dict) -> float:
    label = result["label"].lower()
    negative = any(k in label for k in ("neg", "label_0", "부정", "negative"))
    return -result["score"] if negative else result["score"]


def score(sentences: list[str], lang: str, model_name: str | None = None) -> list[float]:
    """문장마다 -1(부정)~+1(긍정) 사이 점수를 매긴다. Colab: transformers 기본 설치됨."""
    pipe = _pipeline(model_name or MODELS[lang])
    return [_signed(r) for r in pipe(sentences, batch_size=16)]


def segments(sentences: list[str], n: int = 20) -> list[list[str]]:
    """서사를 길이가 비슷한 n개 구간으로 나눈다."""
    n = max(1, min(n, len(sentences)))
    return [list(chunk) for chunk in np.array_split(np.array(sentences, dtype=object), n)]


def arc(scores: list[float], window: int = 3) -> np.ndarray:
    """구간 점수에 이동 평균을 씌워 정서 곡선을 만든다."""
    s = np.asarray(scores, dtype=float)
    if window <= 1 or len(s) < window:
        return s
    pad = window // 2
    padded = np.pad(s, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(s)]


def turning_points(curve: np.ndarray, k: int = 3) -> list[int]:
    """곡선이 가장 크게 꺾이는 구간 번호. 원문을 찾아 꼼꼼히 읽을 지점이다."""
    if len(curve) < 3:
        return []
    change = np.abs(np.diff(curve, n=2))
    return sorted((np.argsort(change)[::-1][:k] + 1).tolist())
