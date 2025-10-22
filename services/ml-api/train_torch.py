import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Tiny MLP model
class RiskNet(nn.Module):
    def __init__(self, in_dim=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

BASE_DIR   = Path(__file__).resolve().parent
DATA_CSV   = (BASE_DIR / "../../data/loan_data.csv").resolve()
OUT_WEIGHTS= (BASE_DIR / "../../models/risknet.pt").resolve()
FEATURES   = ["income", "credit_score", "loan_amount", "tenure_months", "dti"]
LABEL      = "risk"  # 0/1

def load_data():
    """
    If the CSV doesn't exist, generate a synthetic dataset with the required columns:
    income, credit_score, loan_amount, tenure_months, dti, risk.
    """
    if not DATA_CSV.exists():
        DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        n = 20000

        income = rng.normal(50000, 15000, n).clip(5000, 150000)
        credit = rng.normal(680, 60, n).clip(300, 850)
        amount = rng.normal(15000, 8000, n).clip(1000, 60000)
        tenure = rng.integers(6, 60, n)
        dti    = rng.normal(0.35, 0.15, n).clip(0.0, 1.5)

        # underlying risk function to label data (heuristic)
        z = (
            -3.0
            - 0.00003 * income
            - 0.006   * credit
            + 0.00005 * amount
            + 1.8     * dti
            - 0.01    * tenure
        )
        p = 1.0 / (1.0 + np.exp(-z))
        risk = (rng.random(n) < p).astype(int)

        df = pd.DataFrame({
            "income": income,
            "credit_score": credit,
            "loan_amount": amount,
            "tenure_months": tenure,
            "dti": dti,
            "risk": risk,
        })
        df.to_csv(DATA_CSV, index=False)
        print(f"⚙️  Generated synthetic dataset at: {DATA_CSV} (rows={len(df)})")

    df = pd.read_csv(DATA_CSV)
    required = {"income","credit_score","loan_amount","tenure_months","dti","risk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    X = df[FEATURES].astype(float).values
    y = df[LABEL].astype(int).values
    return X, y

def make_loaders(X, y, batch=64, test_size=0.2, seed=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    y_val   = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    return DataLoader(train_ds, batch_size=batch, shuffle=True), DataLoader(val_ds, batch_size=batch)

def train():
    X, y = load_data()
    train_loader, val_loader = make_loaders(X, y)

    model = RiskNet(in_dim=len(FEATURES), hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    best_val = float("inf")
    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            preds = model(xb)
            loss = bce(preds, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for xb, yb in val_loader:
                preds = model(xb)
                vtotal += bce(preds, yb).item() * len(xb)
            val_loss = vtotal / len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUT_WEIGHTS)
            print(f"  ✓ saved {OUT_WEIGHTS}")

if __name__ == "__main__":
    train()
