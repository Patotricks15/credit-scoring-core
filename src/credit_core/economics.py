"""Expected loss and cutoff profit.

Amounts are expressed through EAD (exposure at default). `margin` is the profit on
EAD when a loan is repaid, and `lgd` is the loss given default as a fraction of EAD.
"""

import numpy as np
import pandas as pd


def expected_loss(pd_, lgd: float, ead) -> np.ndarray:
    return np.asarray(pd_) * lgd * np.asarray(ead)


def cutoff_profit(pd_, defaulted, ead, margin: float, lgd: float, cutoffs) -> pd.DataFrame:
    """Profit for each PD cutoff: approve applicants with PD <= cutoff.

    `expected_profit` uses predicted PD. `realized_profit` uses the observed default flag.
    """
    pd_ = np.asarray(pd_, dtype=float)
    defaulted = np.asarray(defaulted, dtype=float)
    ead = np.asarray(ead, dtype=float)
    rows = []
    for c in cutoffs:
        approved = pd_ <= c
        n_app = int(approved.sum())
        exp_p = float(np.sum(((1 - pd_[approved]) * margin - pd_[approved] * lgd) * ead[approved]))
        real_p = float(np.sum(((1 - defaulted[approved]) * margin - defaulted[approved] * lgd) * ead[approved]))
        rows.append(
            {
                "cutoff_pd": float(c),
                "approved": n_app,
                "approval_rate": n_app / len(pd_),
                "expected_profit": exp_p,
                "realized_profit": real_p,
                "realized_bad_rate": float(defaulted[approved].mean()) if n_app else np.nan,
            }
        )
    return pd.DataFrame(rows)
