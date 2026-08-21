import glob
import math
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAT, LON = 19.2200, 100.0500

try:
    xgb_models = joblib.load("xgboost_7days_model.pkl")
    print("โหลดโมเดล XGBoost สำเร็จ!")
except Exception as e:
    xgb_models = None

# ค่า RMSE เฉลี่ยจากการประเมินผล (ใช้อ้างอิง Margin of Error)
RMSE_BY_DAY = [3.82, 4.25, 4.90, 5.35, 5.80, 6.10, 6.45]


def prepare_latest_features():
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
    df_clean = (
        df_all.dropna(subset=["Date", "PM2.5"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    df_clean["temp"] = 28.5
    df_clean["humidity"] = 65.0
    df_clean["rain"] = 0.0
    df_clean["wind_speed"] = 1.2
    df_clean["wind_sin"] = 0.5
    df_clean["wind_cos"] = 0.86
    df_clean["hotspot_count"] = 5

    for lag in range(1, 8):
        df_clean[f"pm25_lag_{lag}"] = df_clean["PM2.5"].shift(lag)

    df_clean["pm25_roll_3"] = df_clean["PM2.5"].shift(1).rolling(3).mean()
    df_clean["pm25_roll_7"] = df_clean["PM2.5"].shift(1).rolling(7).mean()

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

    return df_clean[features].dropna().iloc[[-1]]


@app.get("/predict")
def predict_pm25():
    if xgb_models is None:
        return {"status": "error", "message": "Model not loaded"}

    try:
        latest_features = prepare_latest_features()
        forecast_data = []

        # ดึง Feature Importance จาก Day 1 เพื่อแสดงปัจจัยส่งผล
        model_day1 = xgb_models["day_1"]
        importances = model_day1.feature_importances_
        feature_names = latest_features.columns.tolist()

        # คำนวณเป็น % Importance Top 5
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        importance_df["percentage"] = (
            importance_df["importance"] / importance_df["importance"].sum()
        ) * 100
        top_features = (
            importance_df.sort_values(by="percentage", ascending=False)
            .head(5)
            .to_dict(orient="records")
        )

        for h in range(1, 8):
            model = xgb_models[f"day_{h}"]
            pred_val = float(model.predict(latest_features)[0])
            pred_val = max(0.0, round(pred_val, 2))

            # คำนวณ 95% Confidence Interval (± 1.96 * RMSE)
            rmse = RMSE_BY_DAY[h - 1]
            margin = round(1.96 * rmse, 2)
            lower_bound = max(0.0, round(pred_val - margin, 2))
            upper_bound = round(pred_val + margin, 2)

            forecast_data.append(
                {
                    "day": f"Day {h}",
                    "pm25": pred_val,
                    "confidence_interval": {
                        "lower": lower_bound,
                        "upper": upper_bound,
                        "margin": f"±{margin}",
                    },
                }
            )

        return {
            "status": "success",
            "location": "Phayao Provincial Stadium",
            "model_used": "XGBoost Multi-step with XAI",
            "feature_importance_top5": [
                {
                    "factor": f["feature"],
                    "impact_percent": round(f["percentage"], 2),
                }
                for f in top_features
            ],
            "forecast_7_days": forecast_data,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}