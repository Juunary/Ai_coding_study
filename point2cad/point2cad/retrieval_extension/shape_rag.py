"""Shape vector DB + retrieval-based type inference.

  store   : example patch -> ShapeEncoder.embed -> (vector, type) in ShapeVectorDB
  query   : new cluster   -> embed -> k nearest examples -> similarity-weighted vote
  verify  : voted candidates -> fitting errors -> select_surface_type(errors, candidates)

Adding a new surface type means adding its example vectors to the DB (and a
fitting module for it); the encoder is not retrained.
"""
import numpy as np

from retrieval_candidates import SupportSet
from select_type import select_surface_type


class ShapeVectorDB(SupportSet):
    def add_patches(self, encoder, patches, label):
        for p in patches:
            self.add(encoder.embed(p), label)

    def search(self, query, k=10):
        x = np.stack(self.vectors)
        sim = x @ np.asarray(query, dtype=np.float32)       # cosine (vectors are unit length)
        nn = np.argsort(-sim)[: min(k, len(sim))]
        return [(self.labels[i], float(sim[i])) for i in nn]

    def vote(self, query, k=10, temperature=0.1):
        scores = {}
        for label, sim in self.search(query, k):
            scores[label] = scores.get(label, 0.0) + np.exp(sim / temperature)
        total = sum(scores.values())
        return {t: s / total for t, s in sorted(scores.items(), key=lambda kv: -kv[1])}

    def save(self, path):
        np.savez(path, vectors=np.stack(self.vectors), labels=np.array(self.labels))

    @classmethod
    def load(cls, path):
        f = np.load(path)
        db = cls()
        db.add(f["vectors"], None)
        db.labels = [str(t) for t in f["labels"]]
        return db


def infer_shape_type(points, encoder, db, k=10, min_share=0.2, errors=None, min_similarity=None):
    """Returns {"type", "confidence", "similarity", "candidates", "votes"}.

    Without `errors` the type is the top vote. With `errors` (fit error per type,
    as in fitting_one_surface.py) the voted candidates restrict the usual
    error-based selection. If the nearest example is less similar than
    `min_similarity`, the type is "unknown" (e.g. several faces in one cluster).
    """
    z = encoder.embed(points)
    similarity = db.search(z, 1)[0][1]
    if min_similarity is not None and similarity < min_similarity:
        return {"type": "unknown", "confidence": 0.0, "similarity": similarity,
                "candidates": set(), "votes": {}}
    votes = db.vote(z, k)
    candidates = {t for t, s in votes.items() if s >= min_share} or {next(iter(votes))}
    if errors is None:
        best = next(iter(votes))
    else:
        best = select_surface_type(errors, candidates=candidates & set(errors))
    return {"type": best, "confidence": votes.get(best, 0.0), "similarity": similarity,
            "candidates": candidates, "votes": votes}
