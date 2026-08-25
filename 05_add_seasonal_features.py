import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

print("⏳ กำลังโหลดและสร้าง Seasonal & Meteorological Features ใหม่...")

# 1. โหลดข้อมูล
df = pd.read_csv('merged_pm25_gee_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ---------------------------------------------------------
# Feature Engineering ขั้นสูง (สเต็ปที่ 2)
# ---------------------------------------------------------
# 1. Physical & Meteorological Features
df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
df['dewpoint_spread'] = df['temp_k'] - df['dewpoint_k']  # ยิ่งค่าน้อย ความชื้นสัมพัทธ์ยิ่งสูง

# 2. Cyclical Time Features (Seasonal Pattern 365 วัน)
day_of_year = df['date'].dt.dayofyear
df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)
df['month'] = df['date'].dt.month

# 3. Historical Lags & Rolling Features
feature_cols = ['sin_day', 'cos_day', 'month', 'dewpoint_spread', 'temp_k', 'wind_speed', 'precip_m']

for lag in [1, 2, 3, 7]:
    df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
    df[f'hotspot_lag_{lag}'] = df['hotspot_count_50km'].shift(lag)
    feature_cols.extend([f'pm25_lag_{lag}', f'hotspot_lag_{lag}'])

df['pm25_roll_3'] = df['pm25'].shift(1).rolling(3).mean()
df['pm25_roll_7'] = df['pm25'].shift(1).rolling(7).mean()
feature_cols.extend(['pm25_roll_3', 'pm25_roll_7'])

df_model = df.dropna(subset=feature_cols + ['pm25']).reset_index(drop=True)
df_model['target_t1'] = df_model['pm25'].shift(-1)
valid_df = df_model.dropna(subset=['target_t1'] + feature_cols).reset_index(drop=True)

X = valid_df[feature_cols].astype(float)
y = valid_df['target_t1'].astype(float)

# ---------------------------------------------------------
# ประเมินผลด้วย Best Parameters จาก Optuna
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

print("\n" + "="*50)
print("📊 ผลการประเมินสเต็ปที่ 2 (เพิ่ม Seasonal & Dewpoint Features)")
print("="*50)
print(f"MAE  (t+1): {np.mean(mae_l):.3f}")
print(f"RMSE (t+1): {np.mean(rmse_l):.3f}")
print(f"R²   (t+1): {np.mean(r2_l):.3f}")
print("="*50)