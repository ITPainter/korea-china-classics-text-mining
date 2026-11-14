"""토픽 모델링: scikit-learn LDA (3장). BERTopic은 노트북에서 선택 실습으로 다룬다."""

import pandas as pd


def lda(docs: list[list[str]], n_topics: int = 5, top_words: int = 8, seed: int = 0):
    """토큰화된 문서로 LDA를 학습해 (토픽별 상위 단어 표, 문서-토픽 비중 표)를 돌려준다."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(analyzer=lambda x: x)
    x = vec.fit_transform(docs)
    model = LatentDirichletAllocation(n_components=n_topics, random_state=seed, learning_method="batch")
    doc_topic = model.fit_transform(x)
    vocab = vec.get_feature_names_out()
    topics = pd.DataFrame(
        {f"토픽{k}": [vocab[i] for i in comp.argsort()[::-1][:top_words]] for k, comp in enumerate(model.components_)}
    )
    weights = pd.DataFrame(doc_topic, columns=[f"토픽{k}" for k in range(n_topics)]).round(3)
    return topics, weights
