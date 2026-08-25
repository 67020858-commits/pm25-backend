import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

print("⏳ กำลังประมวลผล Advanced Feature Pipeline (Multi-ring Hotspots + Wind Direction)...")

# 1. โหลดข้อมูล PM2.5 เดิมและ GEE ตัวใหม่
df_pm25 = pd.read_csv('merged_pm25_gee_data.csv')[['date', 'pm25']]
df_gee = pd.read_csv('gee_advanced_features.csv')

df_pm25['date'] = pd.to_datetime(df_pm25['date'])
df_gee['date'] = pd.to_datetime(df_gee['date'])

# Merge ข้อมูล
df = pd.merge(df_pm25, df_gee, on='date', how='inner').sort_values('date').reset_index(drop=True)

# ---------------------------------------------------------
# Feature Engineering ขั้นสูง (Advanced Multi-ring + Wind Physics)
# ---------------------------------------------------------
# 1. Wind Speed & Wind Direction (Sine/Cosine Breakdown)
df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
df['wind_dir_rad'] = np.arctan2(df['v_wind'], df['u_wind'])
df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])

# 2. Dewpoint Spread
df['dewpoint_spread'] = df['temp_k'] - df['dewpoint_k']

# 3. Seasonal Cyclical Features
day_of_year = df['date'].dt.dayofyear
df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)

# 4. Multi-ring Hotspots & Transboundary Ratios
# คำนวณจุดความร้อนส่วนต่างภายนอกวง ( Regional Fire Intensity)
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

# 5. Lags & Rolling Features
for lag in [1, 2, 3, 7]:
    df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
    df[f'hs_50k_lag_{lag}'] = df['hotspot_50km'].shift(lag)
    df[f'hs_200k_lag_{lag}'] = df['hotspot_200km'].shift(lag)
    feature_cols.extend([f'pm25_lag_{lag}', f'hs_50k_lag_{lag}', f'hs_200k_lag_{lag}'])

df['pm25_roll_3'] = df['pm25'].shift(1).rolling(3).mean()
df['pm25_roll_7'] = df['pm25'].shift(1).rolling(7).mean()
feature_cols.extend(['pm25_roll_3', 'pm25_roll_7'])

df_model = df.dropna(subset=feature_cols + ['pm25']).reset_index(drop=True)
df_model['target_t1'] = df_model['pm25'].shift(-1)
valid_df = df_model.dropna(subset=['target_t1'] + feature_cols).reset_index(drop=True)

X = valid_df[feature_cols].astype(float)
y = valid_df['target_t1'].astype(float)

# ---------------------------------------------------------
# Evaluation (Optuna Tuned Parameters)
# ---------------------------------------------------------
best_params = {
    'objective': 'reg:quantileerror',
    'quantile_alpha': 0.5,
    'random_state': 42,
    'n_estimators': 238,
    'max_depth': 3,
    'learning_rate': 0.02538,
    'subsample': 0.67587,
    'colsample_bytree': 0.72022,
    'reg_alpha': 0.00115,
    'reg_lambda': 0.01617
}

tscv = TimeSeriesSplit(n_splits=5)
mae_l, rmse_l, r2_l = [], [], []

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    
    model = XGBRegressor(**best_params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    
    mae_l.append(mean_absolute_error(y_te, preds))
    rmse_l.append(np.sqrt(mean_squared_error(y_te, preds)))
    r2_l.append(r2_score(y_te, preds))

print("\n" + "="*60)
print("🚀 ผลการประเมินโมเดลขั้นสูง (Multi-ring Buffers + Wind Direction)")
print("="*60)
print(f"MAE  (t+1): {np.mean(mae_l):.3f}")
print(f"RMSE (t+1): {np.mean(rmse_l):.3f}")
print(f"R²   (t+1): {np.mean(r2_l):.3f}")
print("="*60)