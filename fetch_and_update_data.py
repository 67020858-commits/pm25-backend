import os
import glob
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# พิกัดและรหัสสถานีตรวจวัดสนามกีฬาจังหวัดพะเยา (สถานี 35t)
LAT = 19.1672
LON = 99.8973
STATION_ID = "35t"

def get_data_filepath() -> str:
    """ตรวจสอบไฟล์ข้อมูลหลักระหว่าง .parquet หรือ .csv"""
    if os.path.exists("สนามกีฬาจังหวัดพะเยา_processed.parquet"):
        return "สนามกีฬาจังหวัดพะเยา_processed.parquet"
    elif os.path.exists("สนามกีฬาจังหวัดพะเยา_processed.csv"):
        return "สนามกีฬาจังหวัดพะเยา_processed.csv"
    else:
        raise FileNotFoundError("ไม่พบไฟล์ข้อมูล สนามกีฬาจังหวัดพะเยา_processed (.parquet หรือ .csv)")

def fetch_air4thai_pm25() -> float:
    """ดึงค่าฝุ่น PM2.5 ล่าสุดจาก API กรมควบคุมมลพิษ (Air4Thai)"""
    url = "http://air4thai.pcd.go.th/for_web/getAQI_json.php"
    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        
        # ค้นหาสถานีจังหวัดพะเยา
        for station in data.get("stations", []):
            area = station.get("areaTH", "")
            st_id = station.get("stationID", "")
            
            if "พะเยา" in area or st_id == STATION_ID:
                pm25_val = station.get("LastUpdate", {}).get("PM25", {}).get("value")
                if pm25_val and pm25_val != "-":
                    print(f"[Air4Thai] ดึงข้อมูลสำเร็จจากสถานี {station.get('nameTH')}: {pm25_val} µg/m³")
                    return float(pm25_val)
                    
        print("[Warning] ไม่พบข้อมูลสถานีพะเยาใน Air4Thai ใช้ค่าล่าสุดก่อนหน้าเป็นตัวแทน")
        return None
    except Exception as e:
        print(f"[Error] ไม่สามารถเชื่อมต่อ Air4Thai API ได้: {e}")
        return None

def fetch_openweather_data():
    """ดึงสภาพอากาศจาก OpenWeatherMap API (หากไม่มี API Key จะใช้ค่าเฉลี่ยมาตรฐานสำรอง)"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        # ค่าสำรองมาตรฐานสำหรับจังหวัดพะเยา
        return 28.5, 65.0, 0.0, 3.2, 0.5, 0.86

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={api_key}&units=metric"
    try:
        res = requests.get(url, timeout=10).json()
        temp = float(res["main"]["temp"])
        humidity = float(res["main"]["humidity"])
        wind_speed = float(res["wind"]["speed"])
        wind_deg = float(res.get("wind", {}).get("deg", 0))
        
        # แปลงทิศทางลมเป็น Trigonometric Vector (Sin/Cos)
        rad = np.radians(wind_deg)
        wind_sin = float(np.sin(rad))
        wind_cos = float(np.cos(rad))
        rain = float(res.get("rain", {}).get("1h", 0.0))

        return temp, humidity, rain, wind_speed, wind_sin, wind_cos
    except Exception as e:
        print(f"[Warning] ไม่สามารถดึง OpenWeatherMap ได้: {e}")
        return 28.5, 65.0, 0.0, 3.2, 0.5, 0.86

def update_daily_dataset():
    """ดึงข้อมูลฝุ่นสด นำมาคำนวณ Feature Lags/Rolling และบันทึกต่อท้ายไฟล์ข้อมูลหลัก"""
    data_path = get_data_filepath()
    print(f"กำลังอัปเดตไฟล์ข้อมูลหลัก: {data_path}")

    # 1. โหลดข้อมูลเดิม
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    today_date = pd.Timestamp.now().normalize()
    latest_date_in_df = df["datetime"].max().normalize()

    # ตรวจสอบว่าข้อมูลของวันนี้ถูกอัปเดตไปแล้วหรือยัง
    if latest_date_in_df >= today_date:
        print(f"[Info] ข้อมูลประจำวันที่ {today_date.strftime('%Y-%m-%d')} ถูกอัปเดตเรียบร้อยแล้ว")
        return

    # 2. ดึงค่าฝุ่นสดประจำวัน
    pm25_val = fetch_air4thai_pm25()
    if pm25_val is None:
        # กรณี API ล่ม ให้ใช้ค่า PM2.5 ของวันล่าสุดแทนชั่วคราว
        pm25_val = float(df["pm25"].iloc[-1])

    temp, humidity, rain, wind_speed, wind_sin, wind_cos = fetch_openweather_data()

    # 3. สร้างแถวข้อมูลใหม่
    new_row = {
        "datetime": today_date,
        "pm25": pm25_val,
        "temp": temp,
        "humidity": humidity,
        "rain": rain,
        "wind_speed": wind_speed,
        "wind_sin": wind_sin,
        "wind_cos": wind_cos,
        "hotspot_count": 1.0
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 4. คำนวณ Lag 1-7 และ Rolling Means 3, 7 วันใหม่ทั้งหมด
    for i in range(1, 8):
        df[f"pm25_lag_{i}"] = df["pm25"].shift(i)
        
    df["pm25_roll_3"] = df["pm25"].shift(1).rolling(window=3).mean()
    df["pm25_roll_7"] = df["pm25"].shift(1).rolling(window=7).mean()

    # ตัดแถวที่มี NaN ช่วงเริ่มต้นออก
    df_updated = df.dropna(subset=["pm25_lag_7"]).reset_index(drop=True)

    # 5. บันทึกทับไฟล์เดิม
    if data_path.endswith(".parquet"):
        try:
            df_updated.to_parquet(data_path, index=False, compression="snappy")
        except Exception:
            df_updated.to_csv("สนามกีฬาจังหวัดพะเยา_processed.csv", index=False)
    else:
        df_updated.to_csv(data_path, index=False)

    print(f"[Success] อัปเดตข้อมูลสำเร็จสำหรับวันที่ {today_date.strftime('%Y-%m-%d')} (PM2.5 สด: {pm25_val} µg/m³)")

if __name__ == "__main__":
    update_daily_dataset()