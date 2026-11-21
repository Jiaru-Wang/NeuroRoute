"""Lectures 6.4/6.6: whole-partition metrics and routing math.

No gradients/normalization fitting. Predictions retain original identities.
"""

import math

import numpy as np

from .contracts import require


def classification(truth, pred):
    truth, pred = np.asarray(truth), np.asarray(pred)
    require(
        truth.ndim == pred.ndim == 1 and truth.shape == pred.shape and truth.size > 0,
        "Nonempty aligned labels required",
    )
    require(
        all(
            np.issubdtype(x.dtype, np.integer) and np.isin(x, range(4)).all() for x in (truth, pred)
        ),
        "Integer labels 0..3 required",
    )
    cm = np.bincount(4 * truth.astype(np.int64) + pred, minlength=16).reshape(4, 4)
    tp, support = np.diag(cm).astype(float), cm.sum(1)
    require((support > 0).all(), "Whole partition must contain all four true classes")
    den = support + cm.sum(0)
    return {
        "accuracy": float(tp.sum() / truth.size),
        "macro_f1": float((2 * tp / den).mean()),
        "balanced_accuracy": float((tp / support).mean()),
        "confusion_matrix": cm.tolist(),
        "samples": int(truth.size),
    }


class EarlyStopping:
    def __init__(self, patience, min_delta):
        require(
            type(patience) is int and patience > 0 and math.isfinite(min_delta) and min_delta >= 0,
            "Invalid early stopping",
        )
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.bad, self.last, self.stopped = None, None, 0, 0, False

    def step(self, score, epoch):
        require(
            type(epoch) is int and epoch == self.last + 1 and not self.stopped,
            "Epoch must be consecutive; cannot step after stop",
        )
        require(math.isfinite(score) and 0 <= score <= 1, "Invalid selection score")
        improved = self.best is None or score > self.best + self.min_delta
        if improved:
            self.best, self.best_epoch, self.bad = score, epoch, 0
        else:
            self.bad += 1
        self.last, self.stopped = epoch, self.bad >= self.patience
        return improved, self.stopped


def routing_summary(probabilities, indices):
    p, idx = np.asarray(probabilities, dtype=float), np.asarray(indices)
    require(
        p.ndim == idx.ndim == 2 and p.shape[0] == idx.shape[0] and len(p) > 0,
        "Routing shape mismatch",
    )
    require(
        np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1),
        "Invalid routing probabilities",
    )
    n, e = p.shape
    require(
        e > 1 and np.issubdtype(idx.dtype, np.integer) and ((idx >= 0) & (idx < e)).all(),
        "Invalid expert IDs",
    )
    require(all(len(set(row)) == len(row) for row in idx), "Duplicate expert selection")
    importance = p.mean(0)
    load = np.bincount(idx.reshape(-1), minlength=e) / n
    logp = np.log(np.maximum(p, np.finfo(float).tiny))

    def cv2(x):
        return float(x.var() / (x.mean() ** 2 + 1e-12))

    return {
        "importance": importance.tolist(),
        "load": load.tolist(),
        "load_sum": float(load.sum()),
        "normalized_entropy": float(-(p * logp).sum(1).mean() / math.log(e)),
        "importance_cv2": cv2(importance),
        "hard_load_cv2_diagnostic": cv2(load),
    }


def js_divergence(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    require(
        p.shape == q.shape and p.ndim == 2 and len(p) > 0, "Aligned probability matrices required"
    )
    require(
        all(np.isfinite(x).all() and (x >= 0).all() and np.allclose(x.sum(1), 1) for x in (p, q)),
        "Invalid probability distribution",
    )
    m = (p + q) / 2

    def kl(x):
        return (x * (np.log(np.maximum(x, 1e-300)) - np.log(np.maximum(m, 1e-300)))).sum(1)

    return (kl(p) + kl(q)) / 2


def topk_switch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    require(
        a.shape == b.shape and a.ndim == 2 and len(a) > 0 and a.shape[1] > 0, "Invalid Top-K pairs"
    )
    require(
        all(np.issubdtype(x.dtype, np.integer) and (x >= 0).all() for x in (a, b)),
        "Expert indices must be nonnegative integers",
    )
    require(all(len(set(row)) == a.shape[1] for row in list(a) + list(b)), "Duplicate experts")
    return np.array([1 - len(set(x) & set(y)) / a.shape[1] for x, y in zip(a, b)])
