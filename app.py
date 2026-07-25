from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("loan_risk_model.pkl")

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    probability = float(
        max(model.predict_proba(df)[0])
    )

    return {
        "prediction": int(prediction),
        "score": probability
    }