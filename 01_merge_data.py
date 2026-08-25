import glob
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

print("⏳ กำลังอ่านและรวมไฟล์ Excel PM2.5 ย้อนหลัง (ปี 2022 - 2026)...")

# 1. รวมไฟล์ Excel
excel_files = glob.glob('สนามกีฬาจังหวัดพะเยา*.xlsx')
df_list = []

for file in excel_files:
    df_temp = pd.read_excel(file, skiprows=2)
    df_temp.columns = ['date', 'pm25']
    df_list.append(df_temp)

df_pm25 = pd.concat(df_list, ignore_index=True)

# แปลงวันที่
df_pm25['date'] = pd.to_datetime(df_pm25['date'], format='%d/%m/%Y', errors='coerce').dt.strftime('%Y-%m-%d')

# บังคับแปลงค่า PM2.5 เป็นตัวเลข (ถ้ามีตัวหนังสือปนจะเปลี่ยนเป็น NaN)
df_pm25['pm25'] = pd.to_numeric(df_pm25['pm25'], errors='coerce')
df_pm25 = df_pm25.dropna(subset=['date', 'pm25']).sort_values('date').reset_index(drop=True)

# 2. อ่านไฟล์ GEE และ Merge
FILE_GEE = 'phayao_gee_environmental_features.csv'
df_gee = pd.read_csv(FILE_GEE)
df_gee['date'] = pd.to_datetime(df_gee['date']).dt.strftime('%Y-%m-%d')

df = pd.merge(df_pm25, df_gee, on='date', how='inner')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# บังคับแปลงคอลัมน์ตัวแปรสิ่งแวดล้อมให้เป็น Float ทั้งหมด
gee_cols = ['hotspot_count_50km', 'temp_k', 'dewpoint_k', 'u_wind', 'v_wind', 'precip_m']
for col in gee_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.to_csv('merged_pm25_gee_data.csv', index=False)
print("✅ รวมข้อมูลสำเร็จ! บันทึกไฟล์ 'merged_pm25_gee_data.csv' เรียบร้อย")

# 3. สร้าง Feature Engineering
print("⏳ กำลังสร้าง Features และเตรียมรัน Quantile XGBoost...")

df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)

feature_cols = []
for lag in [1, 2, 3, 7]:
    df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
    df[f'hotspot_lag_{lag}'] = df['hotspot_count_50km'].shift(lag)
    feature_cols.extend([f'pm25_lag_{lag}', f'hotspot_lag_{lag}'])

df['pm25_roll_3'] = df['pm25'].shift(1).rolling(3).mean()
df['pm25_roll_7'] = df['pm25'].shift(1).rolling(7).mean()
feature_cols.extend(['pm25_roll_3', 'pm25_roll_7', 'temp_k', 'wind_speed', 'precip_m'])

df_model = df.dropna(subset=feature_cols + ['pm25']).reset_index(drop=True)

# 4. Time-Series CV & Quantile XGBoost
print("⏳ กำลังประเมินผลโมเดลย้อนหลังด้วย Time-Series CV...")

tscv = TimeSeriesSplit(n_splits=5)
metrics_summary = []

for horizon in range(1, 8):
    df_model[f'target_t{horizon}'] = df_model['pm25'].shift(-horizon)
    valid_df = df_model.dropna(subset=[f'target_t{horizon}'] + feature_cols).reset_index(drop=True)
    
    X = valid_df[feature_cols].astype(float)
    y = valid_df[f'target_t{horizon}'].astype(float)
    
    mae_list, rmse_list, r2_list = [], [], []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = XGBRegressor(
            objective='reg:quantileerror', 
            quantile_alpha=0.5, 
            n_estimators=100, 
            learning_rate=0.05, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        mae_list.append(mean_absolute_error(y_test, preds))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
        r2_list.append(r2_score(y_test, preds))
        
    metrics_summary.append({
        'Horizon': f't+{horizon}',
        'MAE': round(np.mean(mae_list), 3),
        'RMSE': round(np.mean(rmse_list), 3),
        'R2': round(np.mean(r2_list), 3)
    })

# 5. แสดงผลลัพธ์
results_df = pd.DataFrame(metrics_summary)
print("\n" + "="*50)
print("📊 ตารางสรุปประสิทธิภาพโมเดล (นำไปใช้ใน Paper ได้ทันที)")
print("="*50)
print(results_df.to_string(index=False))
print("="*50)