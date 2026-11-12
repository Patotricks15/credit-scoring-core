import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

PSI_EPS = 1e-6


def gini(y, score) -> float:
    return float(2 * roc_auc_score(y, score) - 1)


def ks_statistic(y, score) -> float:
    """Max distance between the cumulative distributions of goods and bads (y = 1 for good)."""
    fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(score))
    return float(np.max(np.abs(tpr - fpr)))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        mask = bins == b
        total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def psi(expected, actual, n_bins: int = 10) -> float:
    """Population stability index. Edges are quantiles of the expected sample."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e_counts = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=len(edges) + 1)
    a_counts = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=len(edges) + 1)
    e_pct = np.maximum(e_counts / e_counts.sum(), PSI_EPS)
    a_pct = np.maximum(a_counts / a_counts.sum(), PSI_EPS)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def discrimination(y, score) -> dict:
    return {"auc": float(roc_auc_score(y, score)), "gini": gini(y, score), "ks": ks_statistic(y, score)}


def decile_table(y, score) -> pd.DataFrame:
    """Bad rate per score decile; decile 1 holds the lowest scores (highest risk when score is good-odds)."""
    df = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score)})
    df["decile"] = pd.qcut(df["score"].rank(method="first"), 10, labels=False) + 1
    table = df.groupby("decile").agg(n=("y", "size"), good_rate=("y", "mean"))
    return table.reset_index()
