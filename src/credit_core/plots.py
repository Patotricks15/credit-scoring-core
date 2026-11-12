"""Plot helpers shared by the credit notebooks. Each function returns the matplotlib figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def plot_roc(y, score, label: str = "model", ax=None):
    fig, ax = _axes(ax)
    fpr, tpr, _ = roc_curve(y, score)
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend()
    return fig


def plot_ks(y, score, ax=None):
    fig, ax = _axes(ax)
    y = np.asarray(y)
    order = np.argsort(score)
    cum_good = np.cumsum(y[order] == 1) / np.sum(y == 1)
    cum_bad = np.cumsum(y[order] == 0) / np.sum(y == 0)
    x = np.linspace(0, 1, len(order))
    ax.plot(x, cum_bad, label="bad")
    ax.plot(x, cum_good, label="good")
    ax.set(xlabel="Population share (lowest score first)", ylabel="Cumulative share", title="KS plot")
    ax.legend()
    return fig


def plot_reliability(y, p, n_bins: int = 10, ax=None):
    fig, ax = _axes(ax)
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot(g["pred"], g["obs"], "o-")
    ax.set(xlabel="Predicted probability", ylabel="Observed rate", title="Reliability diagram")
    return fig


def plot_score_distribution(score, y, ax=None):
    fig, ax = _axes(ax)
    y = np.asarray(y)
    score = np.asarray(score)
    ax.hist(score[y == 1], bins=40, alpha=0.6, density=True, label="good")
    ax.hist(score[y == 0], bins=40, alpha=0.6, density=True, label="bad")
    ax.set(xlabel="Score", ylabel="Density", title="Score distribution by outcome")
    ax.legend()
    return fig


def plot_psi_bars(psi_by_feature: pd.Series, ax=None):
    fig, ax = _axes(ax)
    s = psi_by_feature.sort_values()
    ax.barh(s.index.astype(str), s.values)
    ax.axvline(0.10, color="orange", ls="--", label="0.10 watch")
    ax.axvline(0.25, color="red", ls="--", label="0.25 alert")
    ax.set(xlabel="PSI", title="Population stability index")
    ax.legend()
    return fig


def plot_profit_curve(table: pd.DataFrame, ax=None):
    fig, ax = _axes(ax)
    ax.plot(table["approval_rate"], table["expected_profit"], label="expected")
    ax.plot(table["approval_rate"], table["realized_profit"], label="realized")
    ax.set(xlabel="Approval rate", ylabel="Profit (same units as EAD)", title="Profit vs approval rate")
    ax.legend()
    return fig


def plot_monthly_metric(periods, values, ylabel: str, title: str, threshold=None, ax=None):
    fig, ax = _axes(ax)
    ax.plot(list(periods), list(values), marker="o")
    if threshold is not None:
        ax.axhline(threshold, color="red", ls="--", lw=0.8)
    ax.set(xlabel="Period", ylabel=ylabel, title=title)
    ax.tick_params(axis="x", rotation=45)
    return fig


def _axes(ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        return fig, ax
    return ax.figure, ax
