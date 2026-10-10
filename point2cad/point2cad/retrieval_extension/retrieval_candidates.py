"""kNN retrieval of candidate surface types (draft).

Idea: keep segmentation and the fitting modules as they are. Embed each
surface cluster with a frozen encoder, look up the nearest example patches in
a support set, and use the voted types as the candidate set passed to
`select_surface_type`. Adding a type = adding example embeddings (plus a
fitting module for that type); no retraining.
"""
import numpy as np


class SupportSet:
    def __init__(self):
        self.vectors = []
        self.labels = []

    def add(self, vectors, label):
        for v in np.atleast_2d(vectors):
            self.vectors.append(np.asarray(v, dtype=np.float32))
            self.labels.append(label)

    def candidates(self, query, k=10, min_votes=2):
        """Return type names that received at least `min_votes` among the k nearest examples."""
        if not self.vectors:
            return None
        x = np.stack(self.vectors)
        q = np.asarray(query, dtype=np.float32)
        d = np.linalg.norm(x - q, axis=1)
        nn = np.argsort(d)[: min(k, len(d))]
        votes = {}
        for i in nn:
            votes[self.labels[i]] = votes.get(self.labels[i], 0) + 1
        picked = {t for t, c in votes.items() if c >= min_votes}
        return picked or {max(votes, key=votes.get)}
