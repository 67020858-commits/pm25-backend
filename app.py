from contextlib import asynccontextmanager
import json
import os
import glob
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

models = {}
feature_names = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, feature_names
    try:
        # โหลดรายชื่อ Feature Sequence
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
            
        # โหลดโมเดลทั้ง 21 ตัว (t1..t7 x low/mid/high)
        for h in range(1, 8):
            for q_name in ['low', 'mid', 'high']:
                key = f"t{h}_{q_name}"
                model = XGBRegressor()
                model.load_model(f"models/xgb_{key}.json")
                models[key] = model
        print("✅ โหลดโมเดล XGBoost Multi-ring ทั้ง 21 ตัวสำเร็จ!")
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    yield
    models.clear()

app = FastAPI(title="PM2.5 Forecasting API - Phayao Stadium", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
async def predict_pm25():
    if not models:
        raise HTTPException(status_code=500, detail="โมเดลยังไม่ได้ถูกโหลดเข้าระบบ")
        
    try:
        # 1. โหลดข้อมูลแถวล่าสุดจากไฟล์เตรียมข้อมูล
        df_pm25 = pd.read_csv('merged_pm25_gee_data.csv')[['date', 'pm25']]
        df_gee = pd.read_csv('gee_advanced_features.csv')
        df_pm25['date'] = pd.to_datetime(df_pm25['date'])
        df_gee['date'] = pd.to_datetime(df_gee['date'])
        
        df = pd.merge(df_pm25, df_gee, on='date', how='inner').sort_values('date').reset_index(drop=True)
        
        # คำนวณ Features ล่าสุด
        df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
        df['wind_dir_rad'] = np.arctan2(df['v_wind'], df['u_wind'])
        df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
        df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
        df['dewpoint_spread'] = df['temp_k'] - df['dewpoint_k']
        
        day_of_year = df['date'].dt.dayofyear
        df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)
        
        df['hotspot_outer_100k'] = df['hotspot_100km'] - df['hotspot_50km']
        df['hotspot_outer_200k'] = df['hotspot_200km'] - df['hotspot_100km']
        df['hotspot_wind_50k_inter'] = df['hotspot_50km'] * df['wind_speed']
        df['hotspot_wind_200k_inter'] = df['hotspot_200km'] * df['wind_speed']
        df['ventilation_index'] = df['wind_speed'] * df['temp_k']
        
        for lag in [1, 2, 3, 7]:
            df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
            df[f'hs_50k_lag_{lag}'] = df['hotspot_50km'].shift(lag)
            df[f'hs_200k_lag_{lag}'] = df['hotspot_200km'].shift(lag)
            
        df['pm25_roll_3'] = df['pm25'].shift(1).rolling(3).mean()
        df['pm25_roll_7'] = df['pm25'].shift(1).rolling(7).mean()
        
        latest_X = df[feature_names].dropna().iloc[[-1]].astype(float)
        
        # 2. พยากรณ์ล่วงหน้า 7 วัน
        forecast_data = []
        for h in range(1, 8):
            p_low = max(0.0, float(models[f"t{h}_low"].predict(latest_X)[0]))
            p_mid = max(0.0, float(models[f"t{h}_mid"].predict(latest_X)[0]))
            p_high = max(0.0, float(models[f"t{h}_high"].predict(latest_X)[0]))
            
            forecast_data.append({
                "day": f"Day {h}",
                "pm25": round(p_mid, 2),
                "confidence_interval": {
                    "lower": round(p_low, 2),
                    "upper": round(p_high, 2)
                }
            })
            
        return {
            "status": "success",
            "location": "Phayao Provincial Stadium",
            "forecast_7_days": forecast_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    