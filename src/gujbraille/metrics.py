"""Evaluation metrics: classification metrics, CER, WER and sequence alignment."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def levenshtein(a: Sequence, b: Sequence) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    prev = np.arange(m + 1)
    for i in range(1, n + 1):
        cur = np.empty(m + 1, dtype=np.int64)
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return int(prev[m])


def cer(ref: str, hyp: str) -> Tuple[int, int]:
    """Returns (edit distance, reference length) over characters (spaces kept)."""
    ref, hyp = " ".join(ref.split()), " ".join(hyp.split())
    return levenshtein(ref, hyp), len(ref)


def wer(ref: str, hyp: str) -> Tuple[int, int]:
    r, h = ref.split(), hyp.split()
    return levenshtein(r, h), len(r)


def align(ref: Sequence, hyp: Sequence) -> List[Tuple[int, int]]:
    """Levenshtein alignment; returns (i_ref, j_hyp) for match/substitution pairs."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ri != hyp[j - 1]))
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(y_true)  # macro averages over classes present in the test set
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 average=None, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "n_classes_in_test": int(len(labels)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),  # mean of per-class F1 (not F1 of the means)
    }
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                    average="weighted", zero_division=0)
    out.update(weighted_precision=float(pw), weighted_recall=float(rw), weighted_f1=float(fw))
    out["per_class"] = [
        {"label": int(l), "precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)}
        for l, a, b, c, d in zip(labels, p, r, f, s)
    ]
    return out


def confusion(y_true, y_pred, labels=None):
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])) if labels is None else labels
    return confusion_matrix(y_true, y_pred, labels=labels), labels


def mean_sd_ci(values: Sequence[float]) -> Dict[str, float]:
    from scipy import stats
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    if len(v) > 1:
        h = float(stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)))
    else:
        h = 0.0
    return {"mean": m, "sd": sd, "ci95_low": m - h, "ci95_high": m + h, "n_runs": int(len(v))}


def mcnemar(y_true, pred_a, pred_b) -> Dict[str, float]:
    """Exact McNemar test (two-sided binomial on discordant pairs)."""
    from scipy.stats import binomtest
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01 = int(np.sum(ca & ~cb))   # A right, B wrong
    n10 = int(np.sum(~ca & cb))   # A wrong, B right
    n = n01 + n10
    pval = float(binomtest(n01, n, 0.5).pvalue) if n else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": pval}
