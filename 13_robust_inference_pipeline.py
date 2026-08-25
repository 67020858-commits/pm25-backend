import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from xgboost import XGBRegressor

print("⏳ กำลังรัน Robust Data Pipeline & Auto Export...")

# 1. โหลดข้อมูลรวมล่าสุด
try:
    df = pd.read_csv('merged_pm25_gee_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการดึงไฟล์ข้อมูล: {e}")
    exit(1)

# 2. Robust Handling: จัดการค่าว่าง (Missing Values Guard)
raw_cols = ['pm25', 'temp_k', 'dewpoint_k', 'u_wind', 'v_wind', 'precip_m', 'hotspot_count_50km']
df[raw_cols] = df[raw_cols].ffill().bfill()
for col in raw_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

# 3. Feature Engineering Pipeline
df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
df['dewpoint_spread'] = df['temp_k'] - df['dewpoint_k']

day_of_year = df['date'].dt.dayofyear
df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)
df['month'] = df['date'].dt.month

df['hotspot_wind_inter'] = df['hotspot_count_50km'] * df['wind_speed']
df['ventilation_index'] = df['wind_speed'] * df['temp_k']
df['hotspot_ema_3'] = df['hotspot_count_50km'].ewm(span=3, adjust=False).mean()

for lag in [1, 2, 3, 7]:
    df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag).ffill()
    df[f'hotspot_lag_{lag}'] = df['hotspot_count_50km'].shift(lag).ffill()

df['pm25_roll_3'] = df['pm25'].shift(1).rolling(3, min_periods=1).mean()
df['pm25_roll_7'] = df['pm25'].shift(1).rolling(7, min_periods=1).mean()

# 4. โหลด Feature Names
with open('models/feature_names.json', 'r') as f:
    feature_cols = json.load(f)

# 5. ดึงวันที่ปัจจุบันของระบบคอมพิวเตอร์จริง (Real-time Date)
today_date = datetime.now().strftime('%Y-%m-%d')
latest_row = df.iloc[-1:]
latest_pm25 = float(latest_row['pm25'].values[0])
X_input = latest_row[feature_cols].astype(float)

# 6. Fast Inference และเจนวันที่พยากรณ์ล่วงหน้า
daily_forecasts = []

for h in range(1, 8):
    model_low = XGBRegressor()
    model_low.load_model(f"models/xgb_t{h}_low.json")
    
    model_mid = XGBRegressor()
    model_mid.load_model(f"models/xgb_t{h}_mid.json")
    
    model_high = XGBRegressor()
    model_high.load_model(f"models/xgb_t{h}_high.json")
    
    p_low = max(0.0, float(round(model_low.predict(X_input)[0], 1)))
    p_mid = max(0.0, float(round(model_mid.predict(X_input)[0], 1)))
    p_high = max(0.0, float(round(model_high.predict(X_input)[0], 1)))
    
    # คำนวณวันที่บวกเพิ่มจากวันนี้จริง
    target_date = (datetime.now() + timedelta(days=h)).strftime('%Y-%m-%d')
    
    status = "Good" if p_mid < 37.5 else "Moderate" if p_mid < 75.0 else "Unhealthy"
    status_th = "ดีมาก" if p_mid < 37.5 else "เริ่มมีผลกระทบ" if p_mid < 75.0 else "มีผลกระทบต่อสุขภาพ"
    
    daily_forecasts.append({
        "horizon": f"t+{h}",
        "forecast_date": target_date,
        "pm25_pred": p_mid,
        "ci_lower_90": p_low,
        "ci_upper_90": p_high,
        "status_en": status,
        "status_th": status_th
    })

# 7. Export JSON สำหรับหน้าเว็บ
json_output = {
    "metadata": {
        "location": "Phayao Stadium",
        "latitude": 19.166,
        "longitude": 99.902,
        "last_updated": today_date,
        "current_pm25": round(latest_pm25, 1),
        "unit": "µg/m³",
        "system_status": "Operational (Real-time Date Mode)"
    },
    "forecast": daily_forecasts
}

with open('forecast_output.json', 'w', encoding='utf-8') as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

print(f"✅ อัปเดตไฟล์ 'forecast_output.json' เป็นวันที่ปัจจุบัน ({today_date}) เรียบร้อยแล้ว!")