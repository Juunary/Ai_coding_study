"""Shape embedding model (numpy only).

points --(shape_elements.shape_descriptor)--> descriptor --(MLP)--> L2-normalized embedding

The MLP is trained with a batch-hard triplet loss so that patches of the same
surface type end up close to each other. The output is used as the key of the
shape vector DB in `shape_rag.py`.

The final embedding concatenates the learned vector with the standardized
descriptor (weight `raw_weight`). The learned part separates the trained types
sharply; the raw part keeps element information the MLP had no reason to
preserve, which matters for types that were never in training (e.g. torus).
"""
import numpy as np

from shape_elements import shape_descriptor

CLIP = 5.0


class ShapeEncoder:
    def __init__(self, in_dim, hidden=64, out_dim=32, seed=0, raw_weight=1.0):
        rng = np.random.default_rng(seed)
        self.raw_weight = raw_weight
        self.params = {
            "W1": rng.normal(0, np.sqrt(2 / in_dim), (in_dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, np.sqrt(1 / hidden), (hidden, out_dim)),
            "b2": np.zeros(out_dim),
        }
        self.mu = np.zeros(in_dim)
        self.sigma = np.ones(in_dim)

    # ------------------------------------------------------------------ forward
    def _standardize(self, X):
        # clipped so that one out-of-range element cannot dominate the distance
        return np.clip((X - self.mu) / self.sigma, -CLIP, CLIP)

    def _forward(self, X):
        P = self.params
        x = self._standardize(X)
        pre = x @ P["W1"] + P["b1"]
        h = np.maximum(pre, 0)
        u = h @ P["W2"] + P["b2"]
        norm = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        return u / norm, (x, pre, h, norm)

    def embed_descriptors(self, X):
        X = np.atleast_2d(X)
        z = self._forward(X)[0]
        if self.raw_weight <= 0:
            return z
        x = self._standardize(X)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        e = np.hstack([z, self.raw_weight * x])
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def embed(self, points):
        return self.embed_descriptors(shape_descriptor(points))[0]

    # ----------------------------------------------------------------- training
    def fit(self, X, labels, epochs=300, per_class=48, lr=3e-3, margin=0.3, seed=0, verbose=False):
        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(labels)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-6
        rng = np.random.default_rng(seed)
        classes = np.unique(labels)
        by_class = {c: np.flatnonzero(labels == c) for c in classes}
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        history = []
        steps_per_epoch = max(1, len(X) // (per_class * len(classes)))
        step = 0
        for epoch in range(epochs):
            for _ in range(steps_per_epoch):
                idx = np.concatenate([
                    rng.choice(by_class[c], per_class, replace=len(by_class[c]) < per_class)
                    for c in classes
                ])
                loss, grads = self._triplet_step(X[idx], labels[idx], margin)
                step += 1
                for k in self.params:                      # Adam
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    mh = m[k] / (1 - 0.9 ** step)
                    vh = v[k] / (1 - 0.999 ** step)
                    self.params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            history.append(loss)
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  triplet loss {loss:.4f}")
        return history

    def _triplet_step(self, X, y, margin):
        P = self.params
        z, (x, pre, h, norm) = self._forward(X)
        D = np.maximum(2 - 2 * z @ z.T, 0)                 # squared distance of unit vectors
        same = y[:, None] == y[None, :]
        np.fill_diagonal(same, False)
        diff = y[:, None] != y[None, :]
        pos = np.where(same, D, -np.inf).argmax(axis=1)    # hardest positive
        neg = np.where(diff, D, np.inf).argmin(axis=1)     # hardest negative
        rows = np.arange(len(X))
        viol = D[rows, pos] - D[rows, neg] + margin
        active = viol > 0
        loss = np.maximum(viol, 0).mean()

        dz = np.zeros_like(z)
        scale = 1.0 / len(X)
        for i in np.flatnonzero(active):
            p, n = pos[i], neg[i]
            dz[i] += scale * 2 * ((z[i] - z[p]) - (z[i] - z[n]))
            dz[p] -= scale * 2 * (z[i] - z[p])
            dz[n] += scale * 2 * (z[i] - z[n])

        du = (dz - z * (z * dz).sum(axis=1, keepdims=True)) / norm
        grads = {"W2": h.T @ du, "b2": du.sum(axis=0)}
        dpre = (du @ P["W2"].T) * (pre > 0)
        grads["W1"] = x.T @ dpre
        grads["b1"] = dpre.sum(axis=0)
        return loss, grads

    # ------------------------------------------------------------- persistence
    def save(self, path):
        np.savez(path, mu=self.mu, sigma=self.sigma, raw_weight=self.raw_weight, **self.params)

    @classmethod
    def load(cls, path):
        f = np.load(path)
        enc = cls(f["W1"].shape[0], f["W1"].shape[1], f["W2"].shape[1])
        enc.params = {k: f[k] for k in ("W1", "b1", "W2", "b2")}
        enc.mu, enc.sigma = f["mu"], f["sigma"]
        enc.raw_weight = float(f["raw_weight"])
        return enc
