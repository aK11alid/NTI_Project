import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# تحميل الموديل عند تشغيل السيرفر
model = joblib.load("loan_risk_model.pkl")


# 1. تحديد الـ Schema للبيانات المبعوثة من Power Automate
# (غير الأسماء والأنواع اللي هنا للأسماء الحقيقية للـ Features بنفس ترتيب تدريب الموديل)
class LoanApplicant(BaseModel):
  age: int
  sex: str
  job: str
  housing: str
  saving_accounts: str
  checking_account: str
  credit_amount: int
  duration: int
  # أضف باقي الـ Features بنفس الطريقة هنا...


@app.post("/predict")
def predict(data: LoanApplicant):
  try:
    # تحويل البيانات إلى DataFrame مع الحفاظ على الترتيب والأنواع الصحيحة
    input_dict = (
        data.model_dump() if hasattr(data, "model_dump") else data.dict()
    )
    df = pd.DataFrame([input_dict])

    # التنبؤ بالقرار (0 أو 1)
    prediction = model.predict(df)[0]

    # حساب احتمالية القبول (Class 1 - استحقاق القرض)
    if hasattr(model, "predict_proba"):
      probability = float(model.predict_proba(df)[0][1])
    else:
      probability = None

    return {"prediction": int(prediction), "score": probability}

  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
