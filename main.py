from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Wine Quality API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABELS = {0: "low", 1: "medium", 2: "high"}

# Map logical model names to joblib files
MODEL_FILES: Dict[str, str] = {
    "white": "models/final_finetuned_white.joblib",
    "red": "models/final_finetuned_red.joblib",
    "combined": "models/final_finetuned_combined.joblib",
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


class WineSample(BaseModel, Field):
    fixed_acidity: float = Field(7.0, ge=0, description="Typical white wine acidity")
    volatile_acidity: float = Field(0.3, ge=0)
    citric_acid: float = Field(0.3, ge=0)
    residual_sugar: float = Field(6.0, ge=0)
    chlorides: float = Field(0.04, ge=0)
    free_sulfur_dioxide: float = Field(30.0, ge=0)
    total_sulfur_dioxide: float = Field(130.0, ge=0)
    density: float = Field(1.0, ge=0)
    pH: float = Field(3.3, ge=0)
    sulphates: float = Field(0.5, ge=0)
    alcohol: float = Field(10.0, gt=0)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(_models.keys()),
        "features": len(FEATURES) if FEATURES else 0
    }


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


@app.post("/predict-all")
def predict_all(sample: WineSample):
    # Build DataFrame once
    row = {f: getattr(sample, f) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)

    results = []
    for model_name, model in _models.items():
        y_pred = model.predict(X)
        pred = int(y_pred[0])
        label = LABELS.get(pred, "unknown")
        results.append({
            "model": model_name,
            "predicted_class": pred,
            "predicted_label": label
        })

    return {"results": results}
