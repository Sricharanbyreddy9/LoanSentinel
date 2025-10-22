# train.py
# -----------------------------------------
# Trains a simple loan-risk classifier.
# - If ../../data/loan_data.csv exists, it is used.
#   Expected columns (case-insensitive, underscore/space tolerant):
#     income, credit_score, loan_amount, tenure_months, dti, risk
#   risk must be 0/1 (1 = high risk)
# - Otherwise, synthetic data is generated.
# Saves model → ../../models/rf_loan.joblib
# -----------------------------------------

import os
import sys
import re
import math
import json
import joblib
import numpy as np
import pandas as pd
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Feature names we will use consistently across code
FEATURES: List[str] = ["income", "credit_score", "loan_amount", "tenure_months", "dti"]
LABEL = "risk"

# Resolve important paths robustly (handles spaces in Windows paths)
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "loan_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "rf_loan.joblib")
META_PATH = os.path.join(MODELS_DIR, "rf_loan.meta.json")


def _normalize_cols(cols: List[str]) -> List[str]:
    """Lowercase, replace spaces with underscores, drop extra punctuation."""
    norm = []
    for c in cols:
        x = c.strip().lower()
        x = re.sub(r"\s+", "_", x)         # spaces -> _
        x = re.sub(r"[^a-z0-9_]", "", x)   # keep alnum + underscore
        norm.append(x)
    return norm


def load_or_generate_data() -> pd.DataFrame:
    """Load CSV if present; otherwise generate a synthetic dataset."""
    if os.path.exists(DATA_CSV):
        print(f"[INFO] Using CSV dataset: {DATA_CSV}")
        df = pd.read_csv(DATA_CSV)
        df.columns = _normalize_cols(df.columns.tolist())

        missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(missing)
                + "\nExpected: "
                + ", ".join(FEATURES + [LABEL])
            )

        # Basic sanity fixes
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [LABEL])
        # Ensure label is 0/1
        if df[LABEL].dtype != "int64" and df[LABEL].dtype != "int32":
            df[LABEL] = df[LABEL].astype(int)
        return df

    # Synthetic fallback
    print("[INFO] No CSV found. Generating synthetic data...")
    N = 5000
    income = np.random.normal(50000, 15000, N).clip(15000, 200000)
    credit = np.random.normal(680, 80, N).clip(300, 850)
    amount = np.random.normal(15000, 8000, N).clip(1000, 80000)
    tenure = np.random.randint(6, 84, N)
    dti = np.clip(np.random.normal(0.35, 0.15, N), 0.05, 0.95)

    # Simple (noisy) rule for risk label
    base = ((dti > 0.45) | (credit < 620) | (amount > 40000)).astype(int)
    noise = (np.random.rand(N) < 0.05).astype(int)  # 5% label noise
    risk = np.clip(base ^ noise, 0, 1)

    df = pd.DataFrame(
        {
            "income": income,
            "credit_score": credit,
            "loan_amount": amount,
            "tenure_months": tenure,
            "dti": dti,
            "risk": risk,
        }
    )
    return df


def train_model(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[LABEL].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # Class balance info
    pos_rate = float(np.mean(y_train))
    print(f"[INFO] Train size: {len(y_train)} | Test size: {len(y_test)} | Pos rate: {pos_rate:.3f}")

    # RandomForest works well without scaling; tune trees modestly for stability
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    # Metrics
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred).tolist()

    print(f"[METRIC] AUC: {auc:.3f}")
    print(f"[METRIC] ACC: {acc:.3f}")
    print(f"[METRIC] Confusion Matrix [ [tn, fp], [fn, tp] ]: {cm}")

    return clf, {"auc": float(auc), "accuracy": float(acc), "confusion_matrix": cm}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_or_generate_data()
    print(f"[INFO] Data shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    model, metrics = train_model(df)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"[SAVE] Model → {MODEL_PATH}")

    # Save metadata (features, label, metrics, training source)
    meta = {
        "features": FEATURES,
        "label": LABEL,
        "metrics": metrics,
        "data_source": DATA_CSV if os.path.exists(DATA_CSV) else "synthetic",
        "random_seed": RANDOM_SEED,
        "model_type": "RandomForestClassifier",
        "model_path": MODEL_PATH,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVE] Meta → {META_PATH}")

    print("\n[READY] You can now serve predictions with FastAPI using these same FEATURES.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
