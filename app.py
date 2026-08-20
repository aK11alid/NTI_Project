from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("loan_risk_model.pkl")


# 1. تعريف الـ Schema بنفس أسماء وترتيب الـ Features اللي الموديل اتدرب عليها
class LoanApplicant(BaseModel):
  income: float
  age: int
  employment_status: int
  credit_score: int
  # كمل بقية الـ Features الخاصة بموديلك بنفس الترتيب بالضبط


@app.post("/predict")
def predict(data: LoanApplicant):
  try:
    # تحويل البيانات مع ضمان ترتيب الأعمدة بنفس طريقة الـ Model
    input_data = data.model_dump()  # أو data.dict() لو شغال بـ Pydantic v1
    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]

    # احتمالية الاستحقاق (الفئة 1)
    if hasattr(model, "predict_proba"):
      # الأفضل تحديد احتمالية القبول (Class 1) بدلاً من max()
      probability = float(model.predict_proba(df)[0][1])
    else:
      probability = None

    return {"prediction": int(prediction), "score": probability}
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
