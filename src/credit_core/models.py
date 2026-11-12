import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from credit_core import RANDOM_STATE
from credit_core.binning import WoeEncoder
from credit_core.scorecard import Scorecard

C_GRID = (0.01, 0.1, 1.0, 10.0)


def make_encoder(numeric, categorical, **kwargs) -> WoeEncoder:
    return WoeEncoder(numeric=numeric, categorical=categorical, **kwargs)


def tune_scorecard_C(numeric, categorical, X, y, grid=C_GRID, n_splits: int = 5) -> float:
    """Choose the logistic C by stratified CV ROC-AUC on the training data only."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    best_c, best_auc = None, -np.inf
    for c in grid:
        aucs = []
        for tr, va in cv.split(X, y):
            sc = Scorecard(make_encoder(numeric, categorical), C=c).fit(X.iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[va], sc.predict_good(X.iloc[va])))
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_c, best_auc = c, mean_auc
    return best_c


def fit_monotone_lgbm(X: pd.DataFrame, y, monotone: dict, params: dict | None = None):
    """LightGBM classifier with monotone constraints. `monotone` maps feature -> +1, -1 or 0.

    Target y = 1 is good, so the model predicts good; constraints are given for default risk
    and are flipped here to keep the sign meaning "higher value -> higher risk".
    """
    import lightgbm as lgb

    constraints = [-monotone.get(col, 0) for col in X.columns]
    default_params = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        monotone_constraints=constraints,
        monotone_constraints_method="advanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    if params:
        default_params.update(params)
    model = lgb.LGBMClassifier(**default_params)
    model.fit(X, np.asarray(y))
    return model


def lgbm_frame(X: pd.DataFrame, categories: dict | None = None):
    """Convert categorical columns to pandas category dtype with fixed categories."""
    out = X.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        cats = categories[col] if categories else sorted(out[col].dropna().unique())
        out[col] = pd.Categorical(out[col], categories=cats)
    return out
