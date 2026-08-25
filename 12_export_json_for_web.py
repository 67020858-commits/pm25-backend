import json
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

print("⏳ กำลังโหลดโมเดลและประมวลผลสำหรับ Web Dashboard...")

# 1. โหลดข้อมูลและ Feature List
df = pd.read_csv('merged_pm25_gee_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

with open('models/feature_names.json', 'r') as f:
    feature_cols = json.load(f)

# 2. Compute Features
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
    df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
    df[f'hotspot_lag_{lag}'] = df['hotspot_count_50km'].shift(lag)

df['pm25_roll_3'] = df['pm25'].shift(1).rolling(3).mean()
df['pm25_roll_7'] = df['pm25'].shift(1).rolling(7).mean()

# ดึงข้อมูลล่าสุด 1 แถวเพื่อใช้ทำนาย
df_model = df.dropna(subset=feature_cols + ['pm25']).reset_index(drop=True)
latest_row = df_model.iloc[-1:]
latest_date = latest_row['date'].dt.strftime('%Y-%m-%d').values[0]
latest_pm25 = float(latest_row['pm25'].values[0])
X_input = latest_row[feature_cols].astype(float)

# 3. โหลดโมเดลและทำนาย 7 วันล่วงหน้า
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
    
    target_date = (pd.to_datetime(latest_date) + pd.Timedelta(days=h)).strftime('%Y-%m-%d')
    
    # กำหนดสถานะระดับความเสี่ยงฝุ่น
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

# 4. จัดรูปแบบเป็น JSON Structure มาตรฐาน
json_output = {
    "metadata": {
        "location": "Phayao Stadium",
        "latitude": 19.166,
        "longitude": 99.902,
        "last_updated": latest_date,
        "current_pm25": round(latest_pm25, 1),
        "unit": "µg/m³",
        "model_version": "Quantile XGBoost v1.0 (Optuna Tuned)"
    },
    "forecast": daily_forecasts
}

# บันทึกลงไฟล์ forecast_output.json
with open('forecast_output.json', 'w', encoding='utf-8') as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

print("\n" + "="*50)
print("✅ ส่งออกไฟล์ 'forecast_output.json' เรียบร้อยแล้ว!")
print("="*50)

# แสดงตัวอย่าง JSON
print(json.dumps(json_output, ensure_ascii=False, indent=2))