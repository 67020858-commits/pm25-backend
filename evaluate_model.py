import glob
import math
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# ==========================================
# 1. LOAD & PREPARE DATA
# ==========================================
LAT, LON = 19.2200, 100.0500


def fetch_weather(start_date, end_date):
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
    except Exception:
        pass
    return pd.DataFrame()


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

df_weather = fetch_weather(df_all["Date"].min(), df_all["Date"].max())
df_merged = (
    pd.merge(df_all, df_weather, on="Date", how="left")
    if not df_weather.empty
    else df_all.copy()
)
if df_weather.empty:
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

# Feature Engineering
for lag in range(1, 8):
    df_merged[f"pm25_lag_{lag}"] = df_merged["PM2.5"].shift(lag)

df_merged["pm25_roll_3"] = df_merged["PM2.5"].shift(1).rolling(3).mean()
df_merged["pm25_roll_7"] = df_merged["PM2.5"].shift(1).rolling(7).mean()

for h in range(1, 8):
    df_merged[f"target_t_{h}"] = df_merged["PM2.5"].shift(-h)

df_final = df_merged.dropna().reset_index(drop=True)

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

X = df_final[features].values
Y = df_final[[f"target_t_{h}" for h in range(1, 8)]].values

# ==========================================
# 2. TIME-SERIES CROSS-VALIDATION (5-Folds)
# ==========================================
tscv = TimeSeriesSplit(n_splits=5)
horizon_metrics = {
    h: {"mae": [], "rmse": [], "r2": []} for h in range(1, 8)
}

print("กำลังประเมินผลด้วย Time-Series Cross-Validation...")

for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    for h in range(1, 8):
        model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, Y_train[:, h - 1])
        preds = model.predict(X_test)

        mae = mean_absolute_error(Y_test[:, h - 1], preds)
        rmse = np.sqrt(mean_squared_error(Y_test[:, h - 1], preds))
        r2 = r2_score(Y_test[:, h - 1], preds)

        horizon_metrics[h]["mae"].append(mae)
        horizon_metrics[h]["rmse"].append(rmse)
        horizon_metrics[h]["r2"].append(r2)

# ==========================================
# 3. DISPLAY EVALUATION RESULTS
# ==========================================
print("\n=======================================================")
print("      สรุปผลการประเมิน (Time-Series Cross-Validation)  ")
print("=======================================================")
print(
    f"{'Day Forecast':<15} | {'MAE (ug/m3)':<12} | {'RMSE (ug/m3)':<12} | {'R2 Score':<10}"
)
print("-" * 60)

all_mae, all_rmse, all_r2 = [], [], []

for h in range(1, 8):
    mean_mae = np.mean(horizon_metrics[h]["mae"])
    mean_rmse = np.mean(horizon_metrics[h]["rmse"])
    mean_r2 = np.mean(horizon_metrics[h]["r2"])

    all_mae.append(mean_mae)
    all_rmse.append(mean_rmse)
    all_r2.append(mean_r2)

    print(
        f"Day {h:<11} | {mean_mae:<12.2f} | {mean_rmse:<12.2f} | {mean_r2:<10.4f}"
    )

print("-" * 60)
print(
    f"{'Overall Average':<15} | {np.mean(all_mae):<12.2f} | {np.mean(all_rmse):<12.2f} | {np.mean(all_r2):<10.4f}"
)
print("=======================================================\n")