import joblib
import numpy as np
import pandas as pd
from arch import arch_model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = FastAPI()

# ตั้งค่า CORS อนุญาตให้ Frontend ดึงข้อมูลได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล SARIMA ที่เทรนไว้
model = joblib.load("sarima_model.pkl")

# ป้องกัน Error 404 จากหน้าเว็บ
@app.get("/api/gee-tile")
def get_gee_tile():
    return {"status": "success", "tile_url": ""}

@app.get("/predict")
def predict_pm25():
    # 1. พยากรณ์ล่วงหน้า 7 วัน ( Log Scale) ด้วย SARIMA
    forecast_log = model.forecast(steps=7)
    
    # 2. ปรับแต่งความผันผวนด้วย GARCH Model
    try:
        residuals = model.resid
        garch = arch_model(residuals, vol='Garch', p=1, q=1, dist='normal')
        garch_fit = garch.fit(disp='off')
        
        garch_forecast = garch_fit.forecast(horizon=7)
        volatility = np.sqrt(garch_forecast.variance.values[-1, :])
        
        # ปรับแก้ค่าทำนายตาม Volatility
        adj_factor = np.where(volatility > np.median(volatility), 0.98, 1.01)
        adjusted_log = forecast_log * adj_factor
    except Exception as e:
        # หากเกิดข้อผิดพลาดในการคำนวณ GARCH ให้ใช้ค่า SARIMA ปกติ
        adjusted_log = forecast_log

    # 3. แปลงค่า Log กลับเป็นค่าฝุ่นจริงด้วย np.expm1()
    forecast_actual = np.expm1(adjusted_log)

    # 4. ปัดเศษทศนิยม 2 ตำแหน่ง และป้องกันค่าติดลบ
    predictions = [max(0.0, round(float(val), 2)) for val in forecast_actual]

    return {
        "status": "success",
        "location": "Phayao Provincial Stadium",
        "model_used": "ARIMA-GARCH (Log Scale)",
        "forecast_7_days": predictions,
    }