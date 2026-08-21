import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# อนุญาตให้ Frontend ดึงข้อมูลได้ (กันติด CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล SARIMA
model = joblib.load("sarima_model.pkl")


@app.get("/predict")
def predict_pm25():
    # สั่งพยากรณ์ล่วงหน้า 7 วัน (ได้ค่าในรูปแบบ Log Scale)
    forecast_log = model.forecast(steps=7)

    # แปลงค่า Log กลับมาเป็นค่าฝุ่นจริงด้วย np.expm1()
    forecast_actual = np.expm1(forecast_log)

    # ปัดเศษทศนิยม 2 ตำแหน่ง และแปลงเป็นประเภท float
    predictions = [max(0.0, round(float(val), 2)) for val in forecast_actual]

    return {
        "status": "success",
        "location": "Phayao Provincial Stadium",
        "model_used": "SARIMA (Log Scale)",
        "forecast_7_days": predictions,
    }