"""단어 임베딩과 다국어 문장 임베딩 (4장, 6장)."""

import numpy as np

MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def train_word2vec(docs: list[list[str]], vector_size: int = 100, window: int = 5, min_count: int = 2, seed: int = 0):
    """토큰화된 문서로 word2vec을 학습한다. Colab: !pip -q install gensim"""
    from gensim.models import Word2Vec

    return Word2Vec(docs, vector_size=vector_size, window=window, min_count=min_count, seed=seed, workers=1)


def neighbors(model, word: str, topn: int = 10) -> list[tuple[str, float]]:
    if word not in model.wv:
        return []
    return [(w, round(s, 3)) for w, s in model.wv.most_similar(word, topn=topn)]


def neighbor_overlap(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> float:
    """두 코퍼스에서 같은 개념어의 이웃 단어가 얼마나 겹치는지(자카드 지수)."""
    sa, sb = {w for w, _ in a}, {w for w, _ in b}
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def sentence_model(name: str = MULTILINGUAL_MODEL):
    """한국어·중국어 문장을 같은 공간에 놓는 다국어 모델. Colab: !pip -q install sentence-transformers"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def encode(model, sentences: list[str]) -> np.ndarray:
    return np.asarray(model.encode(sentences, normalize_embeddings=True))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T
