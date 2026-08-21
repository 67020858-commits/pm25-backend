import glob
import math
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ==========================================
# CONSTANTS & CONFIGS
# ==========================================
MAP_KEY = "2c89a6317324943ee8fa774ddc4ecea4"
LAT, LON = 19.2200, 100.0500  # พิกัดสนามกีฬาจังหวัดพะเยา


# ==========================================
# FUNCTIONS: FETCH EXTERNAL FEATURES
# ==========================================
def fetch_weather(start_date, end_date):
    """ดึงข้อมูลสภาพอากาศย้อนหลังจาก Open-Meteo API"""
    print("กำลังดึงข้อมูลสภาพอากาศจาก Open-Meteo...")
    url = "https://api.open-meteo.com/v1/forecast"
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
        print(f"เกิดข้อผิดพลาดในการดึงข้อมูล Weather: {e}")
        return pd.DataFrame()


def fetch_hotspots(radius_km=100, days=10):
    """ดึงข้อมูลจุดความร้อนจาก NASA FIRMS API"""
    print("กำลังดึงข้อมูลจุดความร้อนจาก NASA FIRMS...")
    lat_change = radius_km / 111.0
    lon_change = radius_km / (111.0 * math.cos(math.radians(LAT)))
    bbox = f"{round(LON-lon_change,2)},{round(LAT-lat_change,2)},{round(LON+lon_change,2)},{round(LAT+lat_change,2)}"

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP/{bbox}/{days}"
    try:
        df = pd.read_csv(url)
        if not df.empty and "acq_date" in df.columns:
            df_daily = (
                df.groupby("acq_date").size().reset_index(name="hotspot_count")
            )
            df_daily["Date"] = pd.to_datetime(df_daily["acq_date"])
            return df_daily[["Date", "hotspot_count"]]
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการดึงข้อมูล Hotspot: {e}")
    return pd.DataFrame(columns=["Date", "hotspot_count"])


# ==========================================
# MAIN PIPELINE
# ==========================================
# 1. โหลดและรวมข้อมูล Excel ทุกปี (ปรับปรุงป้องกัน KeyError)
excel_files = sorted(glob.glob("สนามกีฬาจังหวัดพะเยา ปี *.xlsx"))
df_list = []

print("กำลังโหลดและรวมข้อมูลจากไฟล์ Excel...")
for file in excel_files:
    df_temp = pd.read_excel(file, skiprows=3, header=None)
    df_temp = df_temp.dropna(how="all").iloc[:, :2]
    df_temp.columns = ["Date", "PM2.5"]  # บังคับตั้งชื่อคอลัมน์โดยตรง
    df_list.append(df_temp)

df_all = pd.concat(df_list, ignore_index=True)
df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
df_all["PM2.5"] = pd.to_numeric(df_all["PM2.5"], errors="coerce")
df_all = df_all.dropna(subset=["Date", "PM2.5"]).sort_values("Date").reset_index(drop=True)

# 2. ดึง Features สภาพอากาศ & จุดความร้อนมารวม (Merge)
min_date = df_all["Date"].min()
max_date = df_all["Date"].max()

df_weather = fetch_weather(min_date, max_date)
df_hotspot = fetch_hotspots(radius_km=100, days=10)

# Merge ข้อมูลรวมกัน
df_merged = pd.merge(df_all, df_weather, on="Date", how="left")
df_merged = pd.merge(df_merged, df_hotspot, on="Date", how="left")

# Clean ข้อมูลที่สูญหาย (Imputation)
df_merged["hotspot_count"] = df_merged["hotspot_count"].fillna(0)
df_merged = df_merged.ffill().bfill()

feature_cols = ["temp", "humidity", "rain", "wind_speed", "wind_sin", "wind_cos", "hotspot_count"]
data_y = df_merged["PM2.5"].values
data_exog = df_merged[feature_cols].values

# 3. ทำ Log Transformation ให้ตัวแปร Target
data_log = np.log1p(data_y)

# 4. แบ่งข้อมูล Train 80% / Test 20%
train_size = int(len(data_y) * 0.8)

train_log = data_log[:train_size]
train_exog = data_exog[:train_size]

test_data = data_y[train_size:]
test_exog = data_exog[train_size:]

# 5. เทรนโมเดล SARIMAX
print("กำลังเทรนโมเดล SARIMAX (Log Scale + Weather & Hotspots)...")
model = SARIMAX(
    train_log,
    exog=train_exog,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarima_result = model.fit(disp=False)

# 6. ทำนายผลและแปลงค่ากลับ
predictions_log = sarima_result.predict(
    start=len(train_log),
    end=len(train_log) + len(test_data) - 1,
    exog=test_exog,
)
predictions = np.expm1(predictions_log)

# 7. ประเมินผลประสิทธิภาพ
mae = mean_absolute_error(test_data, predictions)
rmse = np.sqrt(mean_squared_error(test_data, predictions))
r2 = r2_score(test_data, predictions)

print("\n==========================================")
print("   ผลการวัดประสิทธิภาพ SARIMAX + Features   ")
print("==========================================")
print(f"MAE  : {mae:.2f} ug/m3")
print(f"RMSE : {rmse:.2f} ug/m3")
print(f"R2   : {r2:.2f}")
print("==========================================\n")

# 8. บันทึกไฟล์ .pkl
print("กำลังสร้างและบันทึกโมเดลฉบับสมบูรณ์...")
final_model = SARIMAX(
    data_log,
    exog=data_exog,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

joblib.dump(final_model, "sarima_model.pkl")
print("บันทึกโมเดลเรียบร้อยแล้วที่ไฟล์: sarima_model.pkl")