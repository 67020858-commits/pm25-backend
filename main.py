import glob
import math
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ตั้งค่า CORS อนุญาตให้ Frontend ดึงข้อมูลได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# CONFIGS & LOAD MODEL
# ==========================================
LAT, LON = 19.2200, 100.0500
MAP_KEY = "2c89a6317324943ee8fa774ddc4ecea4"

# โหลดโมเดล XGBoost Dict 7 วันที่เทรนไว้
try:
    xgb_models = joblib.load("xgboost_7days_model.pkl")
    print("โหลดโมเดล xgboost_7days_model.pkl สำเร็จ!")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    xgb_models = None


# ==========================================
# HELPER FUNCTIONS: FETCH RECENT DATA
# ==========================================
def fetch_recent_weather(start_date, end_date):
    """ดึงข้อมูลสภาพอากาศย้อนหลังจาก Open-Meteo Archive API"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "rain",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "timezone": "Asia/Bangkok",
    }
    try:
        res = requests.get(url, params=params).json()
        if "hourly" in res:
            hourly = res["hourly"]
            df = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(hourly["time"]),
                    "temp": hourly["temperature_2m"],
                    "humidity": hourly["relative_humidity_2m"],
                    "rain": hourly["rain"],
                    "wind_speed": hourly["wind_speed_10m"],
                    "wind_dir": hourly["wind_direction_10m"],
                }
            )
            df_daily = (
                df.resample("D", on="datetime")
                .agg(
                    {
                        "temp": "mean",
                        "humidity": "mean",
                        "rain": "sum",
                        "wind_speed": "mean",
                        "wind_dir": "mean",
                    }
                )
                .reset_index()
            )

            wind_rad = np.radians(df_daily["wind_dir"])
            df_daily["wind_sin"] = np.sin(wind_rad)
            df_daily["wind_cos"] = np.cos(wind_rad)
            df_daily.drop(columns=["wind_dir"], inplace=True)
            df_daily.rename(columns={"datetime": "Date"}, inplace=True)
            return df_daily
    except Exception as e:
        print(f"Weather Fetch Error: {e}")
    return pd.DataFrame()


def prepare_latest_features():
    """โหลดข้อมูลล่าสุด และสร้าง Lag & Rolling Features ให้พร้อมสำหรับส่งให้ XGBoost"""
    excel_files = sorted(glob.glob("สนามกีฬาจังหวัดพะเยา ปี *.xlsx"))
    df_list = [
        pd.read_excel(f, skiprows=3, header=None)
        .dropna(how="all")
        .iloc[:, :2]
        .rename(columns={0: "Date", 1: "PM2.5"})
        for f in excel_files
    ]
    df_all = pd.concat(df_list, ignore_index=True)
    df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
    df_all["PM2.5"] = pd.to_numeric(df_all["PM2.5"], errors="coerce")
    df_all = (
        df_all.dropna(subset=["Date", "PM2.5"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Merge สภาพอากาศ
    df_weather = fetch_recent_weather(
        df_all["Date"].min(), df_all["Date"].max()
    )
    if not df_weather.empty:
        df_merged = pd.merge(df_all, df_weather, on="Date", how="left")
    else:
        df_merged = df_all.copy()
        for col in [
            "temp",
            "humidity",
            "rain",
            "wind_speed",
            "wind_sin",
            "wind_cos",
        ]:
            df_merged[col] = 0

    df_merged["hotspot_count"] = 0
    df_merged = df_merged.ffill().bfill().fillna(0)

    # สร้าง Lag 1-7 วัน และ Rolling Mean
    for lag in range(1, 8):
        df_merged[f"pm25_lag_{lag}"] = df_merged["PM2.5"].shift(lag)

    df_merged["pm25_roll_3"] = df_merged["PM2.5"].shift(1).rolling(3).mean()
    df_merged["pm25_roll_7"] = df_merged["PM2.5"].shift(1).rolling(7).mean()

    df_clean = df_merged.dropna().reset_index(drop=True)

    features = [
        "temp",
        "humidity",
        "rain",
        "wind_speed",
        "wind_sin",
        "wind_cos",
        "hotspot_count",
        "pm25_lag_1",
        "pm25_lag_2",
        "pm25_lag_3",
        "pm25_lag_4",
        "pm25_lag_5",
        "pm25_lag_6",
        "pm25_lag_7",
        "pm25_roll_3",
        "pm25_roll_7",
    ]

    # ดึงข้อมูลแถวล่าสุด (Latest Record) มาใช้พยากรณ์
    latest_row = df_clean[features].iloc[[-1]]
    return latest_row


# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/api/gee-tile")
def get_gee_tile():
    return {"status": "success", "tile_url": ""}


@app.get("/predict")
def predict_pm25():
    if xgb_models is None:
        return {"status": "error", "message": "XGBoost Model is not loaded"}

    try:
        # 1. เตรียม Feature ล่าสุดสำหรับส่งเข้าโมเดล
        latest_features = prepare_latest_features()

        # 2. ทำนายผล 7 วัน โดยใช้โมเดลแยกตามวัน (day_1 ถึง day_7)
        predictions = []
        for h in range(1, 8):
            model_day = xgb_models[f"day_{h}"]
            pred_val = model_day.predict(latest_features)[0]
            # ป้องกันค่าติดลบ และปัดเศษทศนิยม 2 ตำแหน่ง
            clean_val = max(0.0, round(float(pred_val), 2))
            predictions.append(clean_val)

        return {
            "status": "success",
            "location": "Phayao Provincial Stadium",
            "model_used": "XGBoost Multi-step (7 Days Direct)",
            "forecast_7_days": predictions,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}