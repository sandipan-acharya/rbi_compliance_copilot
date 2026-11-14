"""Pluggable text-similarity backends.

tfidf   : runs anywhere (scikit-learn only). Used for the measured results in the report.
minilm  : sentence-transformers/all-MiniLM-L6-v2, the SAME embedder used by the Part III RAG index.
          Needs `sentence-transformers` and access to Hugging Face. Thresholds must be re-calibrated
          (see evaluate.py --backend minilm --calibrate).
"""
from __future__ import annotations
import numpy as np


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
        self._fitted = False

    def fit(self, texts):
        self._vec.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts):
        if not self._fitted:
            self.fit(texts)
        m = self._vec.transform(texts)          # rows are L2-normalised
        return m.toarray().astype("float32")


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name)

    def fit(self, texts):  # no-op, pretrained
        return self

    def encode(self, texts):
        return np.asarray(self._m.encode(list(texts), normalize_embeddings=True), dtype="float32")


def get_embedder(name: str = "tfidf"):
    if name == "minilm":
        return MiniLMEmbedder()
    return TfidfEmbedder()


def minilm_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T
