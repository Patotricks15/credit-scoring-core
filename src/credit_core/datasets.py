"""Loaders for the two real credit datasets. Files are read in place from CREDIT_DATA_DIR."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("CREDIT_DATA_DIR", Path.home() / ".local/share/credit-data"))

GERMAN_PATH = DATA_DIR / "german1" / "german_credit_data.csv"
LENDINGCLUB_PATH = DATA_DIR / "lendingclub" / "accepted_2007_to_2018Q4.csv.gz"

GERMAN_NUMERIC = ["Age", "Credit amount", "Duration"]
GERMAN_CATEGORICAL = ["Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
GERMAN_PROTECTED = ["Sex"]


def load_german(path=GERMAN_PATH):
    """German Credit (1,000 rows). Returns features (without Sex) and y (1 = good)."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["Job"] = df["Job"].astype(str)
    y = (df["Risk"] == "good").astype(int)
    X = df[GERMAN_NUMERIC + GERMAN_CATEGORICAL + GERMAN_PROTECTED]
    return X, y


LC_STATUS_TARGET = {
    "Fully Paid": 1,
    "Charged Off": 0,
    "Default": 0,
    "Does not meet the credit policy. Status:Fully Paid": 1,
    "Does not meet the credit policy. Status:Charged Off": 0,
}

LC_NUMERIC = [
    "loan_amnt",
    "term_months",
    "installment",
    "emp_length_years",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "credit_history_months",
]
LC_CATEGORICAL = ["home_ownership", "verification_status", "purpose"]

# Sign of the monotone constraint on the default risk: +1 means higher value -> higher risk.
LC_MONOTONE = {
    "annual_inc": -1,
    "fico_range_low": -1,
    "dti": 1,
    "delinq_2yrs": 1,
    "inq_last_6mths": 1,
    "pub_rec": 1,
    "revol_util": 1,
}

LC_COLUMNS = [
    "loan_amnt",
    "term",
    "installment",
    "emp_length",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "earliest_cr_line",
    "home_ownership",
    "verification_status",
    "purpose",
    "issue_d",
    "loan_status",
]


def _emp_length_years(s: pd.Series) -> pd.Series:
    text = s.astype("string")
    years = text.str.extract(r"(\d+)")[0].astype("float")
    years = years.where(~text.str.contains("< 1", na=False), 0.0)
    return years


def load_lendingclub(path=LENDINGCLUB_PATH) -> pd.DataFrame:
    """Accepted LendingClub loans with resolved outcomes only.

    Returns application-time features, `issue_date`, and `y` (1 = fully paid, 0 = charged off or default).
    Post-decision fields (payments, recoveries, hardship and settlement fields) are not loaded.
    Grade, sub-grade and interest rate are excluded because they are the lender's own risk price.
    """
    df = pd.read_csv(path, usecols=LC_COLUMNS, low_memory=False)
    df = df[df["loan_status"].isin(LC_STATUS_TARGET)].copy()
    df["y"] = df["loan_status"].map(LC_STATUS_TARGET).astype(int)
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["credit_history_months"] = (
        (df["issue_date"].dt.year - earliest.dt.year) * 12 + (df["issue_date"].dt.month - earliest.dt.month)
    ).astype(float)
    df["term_months"] = df["term"].astype("string").str.extract(r"(\d+)")[0].astype(float)
    df["emp_length_years"] = _emp_length_years(df["emp_length"])
    df["revol_util"] = pd.to_numeric(df["revol_util"].astype("string").str.rstrip("%"), errors="coerce")
    df["EAD"] = df["loan_amnt"].astype(float)
    keep = LC_NUMERIC + LC_CATEGORICAL + ["issue_date", "y", "EAD"]
    return df[keep].reset_index(drop=True)


def split_by_issue_date(df: pd.DataFrame, train_end: str, test_start: str):
    """Out-of-time split: train issued before `train_end`, test issued on or after `test_start`."""
    train = df[df["issue_date"] < pd.Timestamp(train_end)]
    test = df[df["issue_date"] >= pd.Timestamp(test_start)]
    return train, test


def to_frame_features(df: pd.DataFrame):
    X = df[LC_NUMERIC + LC_CATEGORICAL].copy()
    # Nullable Float64 columns (e.g. revol_util) break NumPy float conversion, so use plain float.
    X[LC_NUMERIC] = X[LC_NUMERIC].astype("float64")
    X[LC_CATEGORICAL] = X[LC_CATEGORICAL].astype(str).replace({"nan": np.nan})
    return X, df["y"]
