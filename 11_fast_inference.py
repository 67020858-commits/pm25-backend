import json
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

start_time = time.time()
print("⏳ กำลังโหลดโมเดลที่เซฟไว้เพื่อทำนาย (Fast Inference Mode)...")

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
X_input = latest_row[feature_cols].astype(float)

# 3. โหลดโมเดลที่เซฟไว้แล้วทำนายทันที (ไม่ต้องสั่ง .fit() ใหม่)
forecast_results = []
for h in range(1, 8):
    model_low = XGBRegressor()
    model_low.load_model(f"models/xgb_t{h}_low.json")
    
    model_mid = XGBRegressor()
    model_mid.load_model(f"models/xgb_t{h}_mid.json")
    
    model_high = XGBRegressor()
    model_high.load_model(f"models/xgb_t{h}_high.json")
    
    p_low = max(0, round(model_low.predict(X_input)[0], 1))
    p_mid = max(0, round(model_mid.predict(X_input)[0], 1))
    p_high = max(0, round(model_high.predict(X_input)[0], 1))
    
    forecast_results.append({
        'day': f't+{h}',
        'pm25_pred': p_mid,
        'ci_low': p_low,
        'ci_high': p_high
    })

elapsed = time.time() - start_time

print("\n" + "="*50)
print(f"⚡ ทำนายผลสำเร็จภายในเวลาเพียง {elapsed:.3f} วินาที!")
print("="*50)
print(pd.DataFrame(forecast_results).to_string(index=False))
print("="*50)