import glob
import math
import joblib
import numpy as np
import pandas as pd
import requests
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. FETCH EXTERNAL DATA
# ==========================================
LAT, LON = 19.2200, 100.0500

def fetch_weather(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m", "wind_direction_10m"],
        "timezone": "Asia/Bangkok",
    }
    try:
        res = requests.get(url, params=params).json()
        if "hourly" in res:
            hourly = res["hourly"]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(hourly["time"]),
                "temp": hourly["temperature_2m"],
                "humidity": hourly["relative_humidity_2m"],
                "rain": hourly["rain"],
                "wind_speed": hourly["wind_speed_10m"],
                "wind_dir": hourly["wind_direction_10m"],
            })
            df_daily = df.resample("D", on="datetime").agg({
                "temp": "mean", "humidity": "mean", "rain": "sum", "wind_speed": "mean", "wind_dir": "mean"
            }).reset_index()
            
            wind_rad = np.radians(df_daily["wind_dir"])
            df_daily["wind_sin"] = np.sin(wind_rad)
            df_daily["wind_cos"] = np.cos(wind_rad)
            df_daily.drop(columns=["wind_dir"], inplace=True)
            df_daily.rename(columns={"datetime": "Date"}, inplace=True)
            return df_daily
    except Exception as e:
        print(f"Weather Fetch Error: {e}")
    return pd.DataFrame()

# โหลดข้อมูล Excel
excel_files = sorted(glob.glob("สนามกีฬาจังหวัดพะเยา ปี *.xlsx"))
df_list = [pd.read_excel(f, skiprows=3, header=None).dropna(how="all").iloc[:, :2].rename(columns={0: "Date", 1: "PM2.5"}) for f in excel_files]
df_all = pd.concat(df_list, ignore_index=True)
df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
df_all["PM2.5"] = pd.to_numeric(df_all["PM2.5"], errors="coerce")
df_all = df_all.dropna(subset=["Date", "PM2.5"]).sort_values("Date").reset_index(drop=True)

# Merge ข้อมูลสภาพอากาศ
df_weather = fetch_weather(df_all["Date"].min(), df_all["Date"].max())
if not df_weather.empty:
    df_merged = pd.merge(df_all, df_weather, on="Date", how="left")
else:
    df_merged = df_all.copy()
    for col in ["temp", "humidity", "rain", "wind_speed", "wind_sin", "wind_cos"]: df_merged[col] = 0

df_merged["hotspot_count"] = 0
df_merged = df_merged.ffill().bfill().fillna(0)

# ==========================================
# 2. FEATURE ENGINEERING (Lag & Rolling)
# ==========================================
# สร้าง Lag 1-7 วัน และ Rolling Mean
for lag in range(1, 8):
    df_merged[f"pm25_lag_{lag}"] = df_merged["PM2.5"].shift(lag)

df_merged["pm25_roll_3"] = df_merged["PM2.5"].shift(1).rolling(3).mean()
df_merged["pm25_roll_7"] = df_merged["PM2.5"].shift(1).rolling(7).mean()

# ลบค่า NaN ที่เกิดจากการทำ Lag
df_clean = df_merged.dropna().reset_index(drop=True)

features = [
    "temp", "humidity", "rain", "wind_speed", "wind_sin", "wind_cos", "hotspot_count",
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_4", "pm25_lag_5", "pm25_lag_6", "pm25_lag_7",
    "pm25_roll_3", "pm25_roll_7"
]

# สร้าง Target ล่วงหน้า 7 วัน (t+1 ถึง t+7)
for h in range(1, 8):
    df_clean[f"target_t_{h}"] = df_clean["PM2.5"].shift(-h)

df_final = df_clean.dropna().reset_index(drop=True)

X = df_final[features]
Y = df_final[[f"target_t_{h}" for h in range(1, 8)]]

# แบ่ง Train 80% / Test 20%
train_size = int(len(df_final) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
Y_train, Y_test = Y.iloc[:train_size], Y.iloc[train_size:]

# ==========================================
# 3. TRAIN MULTI-STEP XGBOOST MODELS
# ==========================================
print("กำลังเทรนโมเดล XGBoost 7 วัน...")
models = {}
predictions = pd.DataFrame()

for h in range(1, 8):
    target_col = f"target_t_{h}"
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, Y_train[target_col])
    
    # พยากรณ์บน Test Set
    preds = model.predict(X_test)
    predictions[target_col] = preds
    models[f"day_{h}"] = model

# ==========================================
# 4. EVALUATION
# ==========================================
mae = mean_absolute_error(Y_test, predictions)
rmse = np.sqrt(mean_squared_error(Y_test, predictions))

print("\n==========================================")
print("    ผลการวัดประสิทธิภาพ XGBoost (7 วัน)   ")
print("==========================================")
print(f"Average 7-Day MAE  : {mae:.2f} ug/m3")
print(f"Average 7-Day RMSE : {rmse:.2f} ug/m3")
print("==========================================\n")

# บันทึก Dict รวมโมเดล 7 วัน
joblib.dump(models, "xgboost_7days_model.pkl")
print("บันทึกโมเดลเรียบร้อยแล้วที่ไฟล์: xgboost_7days_model.pkl")