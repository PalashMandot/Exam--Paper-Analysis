"""TF-IDF + genuine K-means. Labels are suggestions, not question identities."""
from __future__ import annotations
from collections import Counter
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from threadpoolctl import threadpool_limits


class KMeansTopicModel:
    def __init__(self, n_clusters=None, random_state=42):
        self.requested_clusters = n_clusters
        self.random_state = random_state
        self.vectorizer_ = None
        self.model_ = None
        self.topic_labels_ = {}
        self.keywords_ = {}

    def fit(self, texts, topic_names=None):
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("K-means needs non-empty question text.")
        topic_names = list(topic_names or ["Unassigned"] * len(texts))
        if len(topic_names) != len(texts):
            raise ValueError("Topic names must align with questions.")
        unique = {}
        for text, topic in zip(texts, topic_names):
            unique.setdefault(text, topic)
        texts, topics = list(unique), list(unique.values())
        generic = {"explain", "describe", "define", "discuss", "example", "examples", "write", "given", "using", "following"}
        self.vectorizer_ = TfidfVectorizer(stop_words=sorted(set(ENGLISH_STOP_WORDS) | generic),
                                          ngram_range=(1, 2), max_features=4000, sublinear_tf=True)
        try:
            vectors = self.vectorizer_.fit_transform(texts)
        except ValueError as exc:
            raise ValueError("Questions do not contain enough usable words for topic grouping.") from exc
        distinct_vectors = len({(tuple(vectors.getrow(i).indices), tuple(np.round(vectors.getrow(i).data, 10)))
                                for i in range(vectors.shape[0])})
        known_topics = {t for t in topics if t and t != "Unassigned"}
        automatic = max(2, round(math.sqrt(len(texts))))
        desired = int(self.requested_clusters) if self.requested_clusters is not None else automatic
        if desired < 1:
            raise ValueError("The number of topic clusters must be positive.")
        count = min(desired, distinct_vectors, len(texts), 8)
        self.model_ = KMeans(n_clusters=count, random_state=self.random_state, n_init=10, max_iter=200)
        with threadpool_limits(limits=1):
            clusters = self.model_.fit_predict(vectors)
        terms = self.vectorizer_.get_feature_names_out()
        for cid in range(count):
            indices = self.model_.cluster_centers_[cid].argsort()[::-1]
            keywords = [str(terms[j]) for j in indices[:3] if self.model_.cluster_centers_[cid][j] > 0]
            self.keywords_[cid] = keywords
            labels = [topics[i] for i, c in enumerate(clusters) if int(c) == cid and topics[i] != "Unassigned"]
            label = Counter(labels).most_common(1)[0][0] if labels else f"Topic {cid+1}: " + ", ".join(keywords)
            self.topic_labels_[cid] = label[:100]
        return self

    def predict_topics(self, texts):
        if self.model_ is None:
            raise ValueError("Fit K-means before assigning topics.")
        if not texts:
            return []
        vectors = self.vectorizer_.transform(texts)
        with threadpool_limits(limits=1):
            clusters = self.model_.predict(vectors)
        return [self.topic_labels_[int(c)] if vectors.getrow(i).nnz else "Unassigned"
                for i, c in enumerate(clusters)]

    def predict_clusters(self, texts):
        if self.model_ is None:
            raise ValueError("Fit K-means before assigning clusters.")
        if not texts:
            return []
        vectors = self.vectorizer_.transform(texts)
        with threadpool_limits(limits=1):
            clusters = self.model_.predict(vectors)
        return [int(c) if vectors.getrow(i).nnz else -1 for i,c in enumerate(clusters)]

    def metadata(self):
        if self.model_ is None:
            raise ValueError("Fit K-means before reading metadata.")
        return {"algorithm": "K-means", "n_clusters": int(self.model_.n_clusters),
                "random_state": self.random_state, "vocabulary_size": len(self.vectorizer_.vocabulary_),
                "clusters": [{"cluster": cid, "topic": self.topic_labels_[cid], "keywords": self.keywords_[cid]}
                             for cid in sorted(self.topic_labels_)]}
