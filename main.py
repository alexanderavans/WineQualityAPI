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

# Map logical model names to joblib files
MODEL_FILES: Dict[str, str] = {
    "white": "models/final_finetuned_white.joblib",
    "red": "models/final_finetuned_red.joblib",
}

_models: Dict[str, object] = {}
_model_features: Dict[str, list[str]] = {}

for name, filepath in MODEL_FILES.items():
    bundle = joblib.load(filepath)
    _models[name] = bundle["model"]
    _model_features[name] = bundle["features"]  # list with features per model

class RedWineSample(BaseModel):
    volatile_acidity: float = Field(0.3, ge=0)
    citric_acid: float = Field(0.3, ge=0)
    pH: float = Field(3.3, ge=0)
    sulphates: float = Field(0.5, ge=0)
    alcohol: float = Field(10.0, gt=0)

class WhiteWineSample(BaseModel):
    volatile_acidity: float = Field(0.3, ge=0)
    residual_sugar: float = Field(6.0, ge=0)
    free_sulfur_dioxide: float = Field(30.0, ge=0)
    density: float = Field(1.0, ge=0)
    alcohol: float = Field(10.0, gt=0)

def _predict_for_model(model_name: str, sample: BaseModel):
    if model_name not in _models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    features = _model_features[model_name]
    row = {f: getattr(sample, f) for f in features}
    X = pd.DataFrame([row], columns=features)

    model = _models[model_name]
    y_pred = model.predict(X)
    pred = int(y_pred[0])

    proba = None
    class_probs = None

    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", list(range(len(proba_arr))))
        class_probs = [
            {"class": int(c), "probability": float(p)}
            for c, p in zip(classes, proba_arr)
        ]
        proba = float(max(proba_arr))

    return {
        "model": model_name,
        "predicted_class": pred,
        "confidence": proba,
        "class_probabilities": class_probs,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(_models.keys()),
        "features_per_model": {
            name: len(feats) for name, feats in _model_features.items()
        },
    }

@app.post("/red/predict")
def predict_red(sample: RedWineSample):
    return _predict_for_model("red", sample)


@app.post("/white/predict")
def predict_white(sample: WhiteWineSample):
    return _predict_for_model("white", sample)
