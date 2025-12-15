from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Dict

app = FastAPI(title="Wine Quality API")

LABELS = {0: "low", 1: "medium", 2: "high"}

# Map logical model names to joblib files
MODEL_FILES: Dict[str, str] = {
    "white": "models/final_finetuned_white.joblib",
    # "red": "models/final_finetuned_red.joblib",
    # "combined": "models/final_finetuned_combined.joblib",
}

# Load models at import/startup
_models: Dict[str, object] = {}
FEATURES = None

for name, filepath in MODEL_FILES.items():
    bundle = joblib.load(filepath)
    _models[name] = bundle["model"]
    if FEATURES is None:
        FEATURES = bundle["features"]
    else:
        if bundle["features"] != FEATURES:
            raise RuntimeError(f"Feature mismatch between models: {name} differs")

class WineSample(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/{model_name}/predict")
def predict(
    sample: WineSample,
    model_name: str = Path(..., description="Model name: one of " + ", ".join(MODEL_FILES.keys()))
):
    model_name = model_name.lower()
    if model_name not in _models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Build DataFrame with exact columns/order
    row = {f: getattr(sample, f) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)

    y_pred = _models[model_name].predict(X)

    pred = int(y_pred[0])
    label = LABELS.get(pred, "unknown")
    return {"model": model_name, "predicted_class": pred, "predicted_label": label}