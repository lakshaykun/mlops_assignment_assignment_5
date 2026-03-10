from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from modelConfig import modelConfig

app = FastAPI(
    title="Iris Classification API",
    description="API for predicting Iris flower species using Random Forest",
    version="1.0.0"
)

# Load model
modelName = "RandomForest"
model = joblib.load(modelConfig[modelName]["path"])

EXPECTED_FEATURES = 4


@app.get("/")
def health_check():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "current_model": modelName
    }


@app.get("/info")
def model_info():
    return modelConfig[modelName]["info"]


class IrisData(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(data: IrisData):
    """
    features = [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]
    """

    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features"
        )

    X = np.array(data.features).reshape(1, -1)

    prediction = model.predict(X)

    return {"prediction": int(prediction[0])}