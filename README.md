# credit-scoring-core

Shared, tested code for the credit score portfolio. Each project in the portfolio depends on this package instead of copying the logic.

## Contents

- `binning.py`: numeric and categorical binning, WOE and IV. Fitted on train data only.
- `scorecard.py`: logistic scorecard on WOE, points table, score, reason codes.
- `metrics.py`: Gini, KS, ECE, Brier, PSI, decile table.
- `economics.py`: expected loss and cutoff profit.
- `models.py`: encoder factory, C tuning by CV, monotone LightGBM.
- `datasets.py`: loaders for German Credit and LendingClub.
- `plots.py`: plot helpers used by the notebooks.

## Convention

Target `y = 1` means **good** (repaid). WOE is `ln(%good / %bad)`, so positive WOE means lower risk. Points are `offset + factor * ln(good/bad)`, with `factor = PDO / ln 2`.

## Data

The loaders read local copies from `CREDIT_DATA_DIR` (default `~/.local/share/credit-data`). The data is not in this repository. See `datasets.py` for the expected file names.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
