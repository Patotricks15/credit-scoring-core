"""Binning, weight of evidence (WOE) and information value (IV).

Convention: the target is 1 for good and 0 for bad, so WOE = ln(%good / %bad)
and a higher WOE means lower risk.
"""

import numpy as np
import pandas as pd

MISSING = "missing"
OTHER = "other"
SMOOTHING = 0.5


def make_numeric_edges(x: pd.Series, n_bins: int = 5) -> np.ndarray:
    """Interior quantile edges computed on non-missing training values."""
    clean = x.dropna()
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(clean.quantile(qs).to_numpy())


def make_categorical_levels(x: pd.Series, min_count: int) -> list:
    """Levels with at least `min_count` training rows; the rest map to OTHER."""
    counts = x.dropna().value_counts()
    return list(counts[counts >= min_count].index)


def assign_numeric(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    values = x.to_numpy(dtype=float)
    codes = np.searchsorted(edges, values, side="right").astype(str)
    return np.where(np.isnan(values), MISSING, "bin" + codes)


def assign_categorical(x: pd.Series, levels: list) -> np.ndarray:
    values = x.to_numpy(dtype=object)
    known = x.isin(levels).to_numpy()
    labels = np.where(known, values, OTHER)
    return np.where(x.isna().to_numpy(), MISSING, labels)


def woe_table(bins: np.ndarray, y: pd.Series) -> pd.DataFrame:
    """Per-bin counts, WOE and IV contribution. Smoothing avoids zero counts."""
    df = pd.DataFrame({"bin": bins, "y": np.asarray(y)})
    grouped = df.groupby("bin")["y"].agg(good="sum", total="count")
    grouped["bad"] = grouped["total"] - grouped["good"]
    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()
    if total_good == 0 or total_bad == 0:
        raise ValueError("WOE needs both good and bad rows in the training data.")
    n_bins = len(grouped)
    pct_good = (grouped["good"] + SMOOTHING) / (total_good + SMOOTHING * n_bins)
    pct_bad = (grouped["bad"] + SMOOTHING) / (total_bad + SMOOTHING * n_bins)
    grouped["pct_good"] = pct_good
    grouped["pct_bad"] = pct_bad
    grouped["woe"] = np.log(pct_good / pct_bad)
    grouped["iv_part"] = (pct_good - pct_bad) * grouped["woe"]
    return grouped


class WoeEncoder:
    """Fits bins and WOE on training data only and keeps features with IV in range."""

    def __init__(
        self,
        numeric: list,
        categorical: list,
        n_bins: int = 5,
        min_count: int = 50,
        iv_min: float = 0.02,
        iv_max: float = 0.5,
    ):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.n_bins = n_bins
        self.min_count = min_count
        self.iv_min = iv_min
        self.iv_max = iv_max

    def _assign(self, col: str, x: pd.Series) -> np.ndarray:
        spec = self.specs_[col]
        if col in self.numeric:
            return assign_numeric(x, spec)
        return assign_categorical(x, spec)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoeEncoder":
        self.specs_ = {}
        self.tables_ = {}
        self.iv_ = {}
        for col in self.numeric:
            self.specs_[col] = make_numeric_edges(X[col], self.n_bins)
        for col in self.categorical:
            self.specs_[col] = make_categorical_levels(X[col], self.min_count)
        for col in self.numeric + self.categorical:
            table = woe_table(self._assign(col, X[col]), y)
            self.tables_[col] = table
            self.iv_[col] = float(table["iv_part"].sum())
        self.dropped_ = {
            col: ("IV below %.2f" % self.iv_min if iv < self.iv_min else "IV above %.2f" % self.iv_max)
            for col, iv in self.iv_.items()
            if iv < self.iv_min or iv > self.iv_max
        }
        self.selected_ = [c for c in self.numeric + self.categorical if c not in self.dropped_]
        return self

    def transform_bins(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({col: self._assign(col, X[col]) for col in self.selected_}, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        bins = self.transform_bins(X)
        out = {col: bins[col].map(self.tables_[col]["woe"]).fillna(0.0).to_numpy() for col in self.selected_}
        return pd.DataFrame(out, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        rows = [
            {"feature": c, "iv": self.iv_[c], "status": "dropped: " + self.dropped_[c] if c in self.dropped_ else "kept"}
            for c in self.numeric + self.categorical
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
