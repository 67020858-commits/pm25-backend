import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

os.makedirs('models', exist_ok=True)
print("⏳ กำลังเตรียมข้อมูล Multi-ring และเทรนโมเดลฉบับสมบูรณ์ที่สุด...")

# 1. โหลดข้อมูล PM2.5 และ GEE Advanced Features (Multi-ring)
df_pm25 = pd.read_csv('merged_pm25_gee_data.csv')[['date', 'pm25']]
df_gee = pd.read_csv('gee_advanced_features.csv')

df_pm25['date'] = pd.to_datetime(df_pm25['date'])
df_gee['date'] = pd.to_datetime(df_gee['date'])

df = pd.merge(df_pm25, df_gee, on='date', how='inner').sort_values('date').reset_index(drop=True)

# 2. Feature Engineering ขั้นสูง (Multi-ring Buffers + Wind Dynamics)
df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
df['wind_dir_rad'] = np.arctan2(df['v_wind'], df['u_wind'])
df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
df['dewpoint_spread'] = df['temp_k'] - df['dewpoint_k']

day_of_year = df['date'].dt.dayofyear
df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)

# Multi-ring Hotspots Calculation
df['hotspot_outer_100k'] = df['hotspot_100km'] - df['hotspot_50km']
df['hotspot_outer_200k'] = df['hotspot_200km'] - df['hotspot_100km']

# Interactions
df['hotspot_wind_50k_inter'] = df['hotspot_50km'] * df['wind_speed']
df['hotspot_wind_200k_inter'] = df['hotspot_200km'] * df['wind_speed']
df['ventilation_index'] = df['wind_speed'] * df['temp_k']

feature_cols = [
    'sin_day', 'cos_day', 'dewpoint_spread', 'temp_k', 'wind_speed', 
    'wind_dir_sin', 'wind_dir_cos', 'precip_m',
    'hotspot_50km', 'hotspot_outer_100k', 'hotspot_outer_200k',
    'hotspot_wind_50k_inter', 'hotspot_wind_200k_inter', 'ventilation_index'
]

# Lags & Rolling Features
for lag in [1, 2, 3, 7]:
    df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
    df[f'hs_50k_lag_{lag}'] = df['hotspot_50km'].shift(lag)
    df[f'hs_200k_lag_{lag}'] = df['hotspot_200km'].shift(lag)
    feature_cols.extend([f'pm25_lag_{lag}', f'hs_50k_lag_{lag}', f'hs_200k_lag_{lag}'])

df['pm25_roll_3'] = df['pm25'].shift(1).rolling(3).mean()
df['pm25_roll_7'] = df['pm25'].shift(1).rolling(7).mean()
feature_cols.extend(['pm25_roll_3', 'pm25_roll_7'])

df_model = df.dropna(subset=feature_cols + ['pm25']).reset_index(drop=True)

# 3. Best Parameters
best_params = {
    'objective': 'reg:quantileerror',
    'random_state': 42,
    'n_estimators': 250,
    'max_depth': 4,
    'learning_rate': 0.02,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.01,
    'reg_lambda': 0.1
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
        
        model_filename = f"models/xgb_t{h}_{q_name}.json"
        model.save_model(model_filename)

with open('models/feature_names.json', 'w') as f:
    json.dump(feature_cols, f)

print("\n" + "="*60)
print("✅ บันทึกโมเดลขั้นสูง Multi-ring ทั้งหมด 21 ตัวลงโฟลเดอร์ 'models/' เรียบร้อยแล้ว!")
print("="*60)