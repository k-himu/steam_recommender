"""
Content-based similarity model.

Baseline implementation: TF-IDF + cosine similarity. This module exposes
a small `ContentModel` interface (fit / most_similar) specifically so
that a future embedding-based model can be swapped in without touching
candidate_generation.py or ranking.py -- those only depend on
`most_similar(appid, top_k)` returning (appid, similarity_score) pairs.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config

logger = logging.getLogger(__name__)


class ContentModel:
    """
    Interface every content-similarity backend must implement:
      - fit(texts, appids)
      - most_similar(appid, top_k) -> list[(appid, score)]
      - save(dir) / load(dir)

    Swapping TF-IDF for sentence embeddings later means writing a new
    class with this same interface -- no other module needs to change.
    """

    def fit(self, texts: pd.Series, appids: pd.Series) -> "ContentModel":
        raise NotImplementedError

    def most_similar(self, appid: int, top_k: int) -> list[tuple[int, float]]:
        raise NotImplementedError

    def similarity(self, appid_a: int, appid_b: int) -> Optional[float]:
        raise NotImplementedError


class TfidfContentModel(ContentModel):
    def __init__(self, tfidf_config: config.TfidfConfig = config.TFIDF_CONFIG):
        self.tfidf_config = tfidf_config
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix: Optional[sparse.csr_matrix] = None
        self.appid_to_row: dict[int, int] = {}
        self.row_to_appid: list[int] = []

    def fit(self, texts: pd.Series, appids: pd.Series) -> "TfidfContentModel":
        if len(texts) != len(appids):
            raise ValueError("texts and appids must be the same length")

        self.vectorizer = TfidfVectorizer(
            max_features=self.tfidf_config.max_features,
            ngram_range=self.tfidf_config.ngram_range,
            min_df=self.tfidf_config.min_df,
            max_df=self.tfidf_config.max_df,
            sublinear_tf=self.tfidf_config.sublinear_tf,
        )
        # texts may contain empty strings (no genres/tags/description at all)
        # -- TfidfVectorizer handles empty docs fine (all-zero row).
        self.matrix = self.vectorizer.fit_transform(texts.fillna(""))
        self.row_to_appid = list(appids)
        self.appid_to_row = {appid: i for i, appid in enumerate(self.row_to_appid)}

        logger.info(
            "Fit TF-IDF model: %d games, vocabulary size %d",
            self.matrix.shape[0], len(self.vectorizer.vocabulary_),
        )
        return self

    def _row_for(self, appid: int) -> Optional[int]:
        return self.appid_to_row.get(appid)

    def most_similar(self, appid: int, top_k: int = 20) -> list[tuple[int, float]]:
        if self.matrix is None:
            raise RuntimeError("Model not fit yet -- call fit() first.")
        row = self._row_for(appid)
        if row is None:
            return []

        query_vec = self.matrix[row]
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        sims[row] = -1.0  # exclude the game itself

        top_indices = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        results = [
            (self.row_to_appid[i], float(sims[i]))
            for i in top_indices
            if sims[i] > 0  # don't return completely unrelated (zero-similarity) games as "similar"
        ]
        return results

    def similarity(self, appid_a: int, appid_b: int) -> Optional[float]:
        row_a, row_b = self._row_for(appid_a), self._row_for(appid_b)
        if row_a is None or row_b is None:
            return None
        return float(cosine_similarity(self.matrix[row_a], self.matrix[row_b])[0, 0])

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "tfidf_matrix.npz", self.matrix)
        with open(directory / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(directory / "tfidf_appid_index.pkl", "wb") as f:
            pickle.dump(self.row_to_appid, f)
        logger.info("Saved TF-IDF model to %s", directory)

    @classmethod
    def load(cls, directory: Path, tfidf_config: config.TfidfConfig = config.TFIDF_CONFIG) -> "TfidfContentModel":
        model = cls(tfidf_config)
        model.matrix = sparse.load_npz(directory / "tfidf_matrix.npz")
        with open(directory / "tfidf_vectorizer.pkl", "rb") as f:
            model.vectorizer = pickle.load(f)
        with open(directory / "tfidf_appid_index.pkl", "rb") as f:
            model.row_to_appid = pickle.load(f)
        model.appid_to_row = {appid: i for i, appid in enumerate(model.row_to_appid)}
        logger.info("Loaded TF-IDF model from %s (%d games)", directory, len(model.row_to_appid))
        return model
