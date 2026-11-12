"""Points-based scorecard on top of WOE features.

Target: 1 = good. The logistic model estimates log(good / bad) as a linear
function of WOE, and points are a linear rescaling of that log-odds.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_core import RANDOM_STATE
from credit_core.binning import WoeEncoder


def points_scaling(pdo: float = 20.0, base_score: float = 600.0, base_odds: float = 50.0):
    """Return (factor, offset) so that score = offset + factor * ln(good/bad).

    At `base_odds` good-to-bad the score equals `base_score`; doubling the odds adds `pdo` points.
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


class Scorecard:
    def __init__(
        self,
        encoder: WoeEncoder,
        C: float = 1.0,
        pdo: float = 20.0,
        base_score: float = 600.0,
        base_odds: float = 50.0,
    ):
        self.encoder = encoder
        self.C = C
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        self.encoder.fit(X, y)
        Z = self.encoder.transform(X)
        self.model_ = LogisticRegression(C=self.C, max_iter=2000, random_state=RANDOM_STATE)
        self.model_.fit(Z, np.asarray(y))
        self.factor_, self.offset_ = points_scaling(self.pdo, self.base_score, self.base_odds)
        return self

    def predict_good(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.encoder.transform(X))[:, 1]

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 - self.predict_good(X)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Points score: higher means lower risk."""
        log_odds = self.model_.decision_function(self.encoder.transform(X))
        return self.offset_ + self.factor_ * log_odds

    def points_table(self) -> pd.DataFrame:
        """One row per feature bin with its points; the intercept row carries the base points."""
        coef = dict(zip(self.encoder.selected_, self.model_.coef_[0]))
        rows = [
            {"feature": "intercept", "bin": "-", "woe": np.nan, "points": self.offset_ + self.factor_ * self.model_.intercept_[0]}
        ]
        for col in self.encoder.selected_:
            table = self.encoder.tables_[col]
            for b, woe in table["woe"].items():
                rows.append({"feature": col, "bin": b, "woe": woe, "points": self.factor_ * coef[col] * woe})
        return pd.DataFrame(rows)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points per feature for each row, relative to that feature's best (highest-points) bin.

        Values are zero or negative; the most negative values are the main reasons for a lower score.
        """
        table = self.points_table().dropna(subset=["woe"])
        best = table.groupby("feature")["points"].max()
        pts_map = {(f, b): p for f, b, p in zip(table["feature"], table["bin"], table["points"])}
        bins = self.encoder.transform_bins(X)
        out = {}
        for col in self.encoder.selected_:
            # Bins never seen in training carry WOE 0, so their points are 0.
            pts = np.array([pts_map.get((col, b), 0.0) for b in bins[col]], dtype=float)
            out[col] = pts - best[col]
        return pd.DataFrame(out, index=X.index)

    def reason_codes(self, X: pd.DataFrame, top: int = 3) -> pd.DataFrame:
        contrib = self.contributions(X)
        codes = {}
        for idx, row in contrib.iterrows():
            negative = row[row < 0].sort_values()
            codes[idx] = list(negative.index[:top])
        width = top
        data = {f"reason_{k + 1}": [codes[i][k] if k < len(codes[i]) else None for i in X.index] for k in range(width)}
        return pd.DataFrame(data, index=X.index)
