import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# สร้างโฟลเดอร์สำหรับเก็บไฟล์โมเดล
os.makedirs('models', exist_ok=True)

print("⏳ กำลังเตรียมข้อมูลและเทรนโมเดลฉบับสมบูรณ์...")

# 1. โหลดข้อมูลรวมล่าสุด
df = pd.read_csv('merged_pm25_gee_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 2. Feature Engineering Pipeline (R² = 0.733)
df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
df['dewpoint_spread'] = df['temp_k'] - df['dewpoint_k']

day_of_year = df['date'].dt.dayofyear
df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)
df['month'] = df['date'].dt.month

df['hotspot_wind_inter'] = df['hotspot_count_50km'] * df['wind_speed']
df['ventilation_index'] = df['wind_speed'] * df['temp_k']
df['hotspot_ema_3'] = df['hotspot_count_50km'].ewm(span=3, adjust=False).mean()

feature_cols = [
    'sin_day', 'cos_day', 'month', 'dewpoint_spread', 'temp_k', 
    'wind_speed', 'precip_m', 'hotspot_wind_inter', 'ventilation_index', 'hotspot_ema_3'
]

for lag in [1, 2, 3, 7]:
    df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
    df[f'hotspot_lag_{lag}'] = df['hotspot_count_50km'].shift(lag)
    feature_cols.extend([f'pm25_lag_{lag}', f'hotspot_lag_{lag}'])

df['pm25_roll_3'] = df['pm25'].shift(1).rolling(3).mean()
df['pm25_roll_7'] = df['pm25'].shift(1).rolling(7).mean()
feature_cols.extend(['pm25_roll_3', 'pm25_roll_7'])

df_model = df.dropna(subset=feature_cols + ['pm25']).reset_index(drop=True)

# 3. Best Hyperparameters จาก Optuna
best_params = {
    'objective': 'reg:quantileerror',
    'random_state': 42,
    'n_estimators': 238,
    'max_depth': 3,
    'learning_rate': 0.02538,
    'subsample': 0.67587,
    'colsample_bytree': 0.72022,
    'reg_alpha': 0.00115,
    'reg_lambda': 0.01617
}

quantiles = {'low': 0.05, 'mid': 0.50, 'high': 0.95}

# 4. เทรนและบันทึกโมเดล t+1 ถึง t+7 (รวม 21 โมเดล)
for h in range(1, 8):
    df_model[f'target_t{h}'] = df_model['pm25'].shift(-h)
    valid_df = df_model.dropna(subset=[f'target_t{h}'] + feature_cols).reset_index(drop=True)
    
    X = valid_df[feature_cols].astype(float)
    y = valid_df[f'target_t{h}'].astype(float)
    
    for q_name, q_val in quantiles.items():
        model = XGBRegressor(**best_params, quantile_alpha=q_val)
        model.fit(X, y)
        
        # เซฟเป็นไฟล์ JSON ของ XGBoost
        model_filename = f"models/xgb_t{h}_{q_name}.json"
        model.save_model(model_filename)

# บันทึกรายชื่อ Feature Sequence เก็บไว้ด้วยเพื่อป้องกันสลับลำดับคอลัมน์ตอนนำไปใช้
import json
with open('models/feature_names.json', 'w') as f:
    json.dump(feature_cols, f)

print("\n" + "="*50)
print("✅ บันทึกโมเดลทั้งหมด 21 ตัวลงโฟลเดอร์ 'models/' เรียบร้อยแล้ว!")
print("="*50)