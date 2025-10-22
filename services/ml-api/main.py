# services/ml-api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, confloat, constr
import json, joblib, re
import numpy as np
from pathlib import Path

# --- Torch (optional)
import torch
from torch_model import load_torch_model  # services/ml-api/torch_model.py

# --- Hugging Face transformers (for /nlp/score)
from transformers import pipeline

app = FastAPI(title="Loan Risk ML API")

# =========================
# Input Schemas
# =========================
class PredictIn(BaseModel):
    income: confloat(ge=0) = Field(..., description="Annual income")
    credit_score: conint(ge=0, le=850) = Field(..., description="Credit score 0..850")
    loan_amount: confloat(gt=0) = Field(..., description="Requested loan amount")
    tenure_months: conint(gt=0) = Field(..., description="Loan tenure in months")
    dti: confloat(ge=0, le=10) = Field(..., description="Debt-to-income ratio (0..10)")

class NLPIn(BaseModel):
    application_text: constr(strip_whitespace=True, min_length=1)

class TextIn(BaseModel):
    application_text: constr(strip_whitespace=True, min_length=1)

# =========================
# Paths & Model Loading
# =========================
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "../../models/rf_loan.joblib").resolve()
META_PATH  = (BASE_DIR / "../../models/rf_loan.meta.json").resolve()
TORCH_PATH = (BASE_DIR / "../../models/risknet.pt").resolve()
SCALER_PATH= (BASE_DIR / "../../models/torch_scaler.joblib").resolve()  # optional, if saved in trainer

DEFAULT_FEATURE_ORDER = ["income", "credit_score", "loan_amount", "tenure_months", "dti"]

# Load RF model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model load failed at {MODEL_PATH}: {e}")

# Load feature order (accepts 'feature_order' or 'features')
try:
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    feature_order = meta.get("feature_order") or meta.get("features") or DEFAULT_FEATURE_ORDER
except Exception:
    feature_order = DEFAULT_FEATURE_ORDER

# Try to load Torch weights & scaler (optional)
torch_model = None
torch_scaler = None
try:
    if TORCH_PATH.exists():
        torch_model = load_torch_model(str(TORCH_PATH), device="cpu")
except Exception:
    torch_model = None  # don't crash if missing

try:
    if SCALER_PATH.exists():
        torch_scaler = joblib.load(SCALER_PATH)
except Exception:
    torch_scaler = None

# Lazy HF pipeline (created only on first call)
_text_clf = None
def get_text_clf():
    global _text_clf
    if _text_clf is None:
        _text_clf = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _text_clf

# =========================
# Health & Schema
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "v": "1.5.0",
        "torch_loaded": bool(torch_model),
        "torch_scaler_loaded": bool(torch_scaler),
    }

@app.get("/schema")
def schema():
    return {"features": feature_order}

# =========================
# RandomForest predictor
# =========================
@app.post("/predict")
def predict(data: PredictIn):
    try:
        X = np.array([[getattr(data, f) for f in feature_order]])
        prob = float(model.predict_proba(X)[0][1])
        label = int(prob >= 0.5)
        return {"risk_probability": prob, "risk_label": label, "backend": "rf"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# =========================
# Torch predictor (uses scaler if available)
# =========================
@app.post("/predict_torch")
def predict_torch(data: PredictIn):
    if torch_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Torch model not available at {TORCH_PATH}. Train and save weights first."
        )
    try:
        X = np.array([[getattr(data, f) for f in feature_order]], dtype=np.float32)

        # Apply same preprocessing as training, if scaler is available
        if torch_scaler is not None:
            X = torch_scaler.transform(X)

        with torch.no_grad():
            p = torch_model(torch.from_numpy(X.astype(np.float32))).item()
        label = int(p >= 0.5)
        return {"risk_probability": float(p), "risk_label": label, "backend": "torch"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# =========================
# NLP: keyword flags + simple score
# =========================
RISK_TERMS = [
    r"\blate payment(s)?\b",
    r"\bdefault(ed|ing)?\b",
    r"\bcharge-?off\b",
    r"\bcollections?\b",
    r"\bbankrupt(cy)?\b",
    r"\bdelinquen(t|cy)\b",
]

def keyword_flags(text: str) -> dict:
    t = text.lower()
    flags = {}
    for i, pat in enumerate(RISK_TERMS):
        flags[f"flag_{i}"] = bool(re.search(pat, t))
    return flags

@app.post("/nlp/extract")
def nlp_extract(payload: NLPIn):
    """
    Returns a simple text risk score (0..1) = fraction of risk keywords found,
    plus a boolean flag map for transparency.
    """
    flags = keyword_flags(payload.application_text)
    score = sum(flags.values()) / max(1, len(flags))
    return {"text_risk_score": float(score), "flags": flags}

# =========================
# NLP: transformer score (/nlp/score)
# =========================
@app.post("/nlp/score")
def nlp_score(payload: TextIn):
    """
    Uses a small pretrained model to produce a smarter text score.
    For now we map sentiment to a risk proxy:
      NEGATIVE -> higher risk (score near 1), POSITIVE -> lower risk (near 0).
    """
    clf = get_text_clf()
    result = clf(payload.application_text, truncation=True)[0]  # {'label': 'NEGATIVE'/'POSITIVE', 'score': 0.xx}
    label = result["label"]
    conf  = float(result["score"])

    if label.upper().startswith("NEG"):
        text_risk_score = conf
    else:
        text_risk_score = 1.0 - conf

    return {
        "model": "distilbert-sst2",
        "label": label,
        "confidence": conf,
        "text_risk_score": float(text_risk_score),
    }
