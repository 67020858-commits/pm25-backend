import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime

print("=== STARTING DAILY UPDATE SCRIPT ===")

PARQUET_PATH = "สนามกีฬาจังหวัดพะเยา_processed.parquet"
STADIUM_LAT = 19.1672
STADIUM_LON = 99.8973
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

def fetch_current_weather_and_pm25():
    if not OPENWEATHER_API_KEY:
        print("[INFO] ไม่พบ OPENWEATHER_API_KEY ใช้ข้อมูลจำลองสำหรับการอัปเดต")
        return {
            "temp": 28.0,
            "humidity": 65.0,
            "rain": 0.0,
            "wind_speed": 3.0,
            "wind_deg": 180,
            "pm25": 25.0,
            "hotspot_count": 0
        }
    
    try:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={STADIUM_LAT}&lon={STADIUM_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
        res_w = requests.get(weather_url, timeout=10).json()
        
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={STADIUM_LAT}&lon={STADIUM_LON}&appid={OPENWEATHER_API_KEY}"
        res_a = requests.get(air_url, timeout=10).json()
        
        temp = res_w.get("main", {}).get("temp", 28.0)
        humidity = res_w.get("main", {}).get("humidity", 65.0)
        rain = res_w.get("rain", {}).get("1h", 0.0)
        wind_speed = res_w.get("wind", {}).get("speed", 3.0)
        wind_deg = res_w.get("wind", {}).get("deg", 180)
        pm25 = res_a.get("list", [{}])[0].get("components", {}).get("pm2_5", 25.0)
        
        return {
            "temp": temp,
            "humidity": humidity,
            "rain": rain,
            "wind_speed": wind_speed,
            "wind_deg": wind_deg,
            "pm25": pm25,
            "hotspot_count": 0
        }
    except Exception as e:
        print(f"[ERROR] เกิดข้อผิดพลาดในการดึงข้อมูล API: {e}")
        return None

def update_dataset():
    print("[INFO] เริ่มต้นตรวจสอบการอัปเดตข้อมูลรายวัน...")
    
    if not os.path.exists(PARQUET_PATH):
        print(f"[ERROR] ไม่พบไฟล์ {PARQUET_PATH} ในโฟลเดอร์ปัจจุบัน")
        return

    df = pd.read_parquet(PARQUET_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    today_str = datetime.now().strftime("%Y-%m-%d")
    latest_date_str = df["datetime"].iloc[-1].strftime("%Y-%m-%d")

    print(f"[INFO] วันที่ล่าสุดในไฟล์: {latest_date_str}")
    print(f"[INFO] วันที่ปัจจุบัน: {today_str}")

    if today_str == latest_date_str:
        print("[SUCCESS] ข้อมูลของวันนี้มีอยู่ในระบบเรียบร้อยแล้ว (ไม่ต้องอัปเดตซ้ำ)")
        return

    new_data = fetch_current_weather_and_pm25()
    if not new_data:
        return

    rad = np.radians(new_data["wind_deg"])
    wind_sin = float(np.sin(rad))
    wind_cos = float(np.cos(rad))

    new_row = {
        "datetime": pd.to_datetime(today_str),
        "pm25": new_data["pm25"],
        "temp": new_data["temp"],
        "humidity": new_data["humidity"],
        "rain": new_data["rain"],
        "wind_speed": new_data["wind_speed"],
        "wind_sin": wind_sin,
        "wind_cos": wind_cos,
        "hotspot_count": new_data["hotspot_count"]
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    for i in range(1, 8):
        df[f"pm25_lag_{i}"] = df["pm25"].shift(i)
    df["pm25_roll_3"] = df["pm25"].shift(1).rolling(window=3).mean()
    df["pm25_roll_7"] = df["pm25"].shift(1).rolling(window=7).mean()

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"[SUCCESS] บันทึกการอัปเดตข้อมูลวันที่ {today_str} สำเร็จเรียบร้อยแล้ว!")

# เรียกทำงานฟังก์ชันทันทีที่รันไฟล์
update_dataset()