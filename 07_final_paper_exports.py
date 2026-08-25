import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

print("⏳ กำลังประมวลผล Final Model Exports สำหรับ Paper...")

# 1. โหลดข้อมูลและสร้าง Features ทั้งหมด
df = pd.read_csv('merged_pm25_gee_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

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

# 2. คำนวณ t+1 ถึง t+7 แบบสะสม
tscv = TimeSeriesSplit(n_splits=5)
metrics_summary = []

for h in range(1, 8):
    df_model[f'target_t{h}'] = df_model['pm25'].shift(-h)
    valid_df = df_model.dropna(subset=[f'target_t{h}'] + feature_cols).reset_index(drop=True)
    
    X = valid_df[feature_cols].astype(float)
    y = valid_df[f'target_t{h}'].astype(float)
    
    m_l, rm_l, r_l = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model = XGBRegressor(**best_params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        m_l.append(mean_absolute_error(y_te, preds))
        rm_l.append(np.sqrt(mean_squared_error(y_te, preds)))
        r_l.append(r2_score(y_te, preds))
        
    metrics_summary.append({
        'Horizon': f't+{h}',
        'MAE': round(np.mean(m_l), 3),
        'RMSE': round(np.mean(rm_l), 3),
        'R2': round(np.mean(r_l), 3)
    })

# บันทึกตารางสรุปผลลง CSV
summary_df = pd.DataFrame(metrics_summary)
summary_df.to_csv('final_metrics_summary.csv', index=False)

print("\n" + "="*50)
print("📊 สรุปผลการประเมินโมเดลฉบับสมบูรณ์ (บันทึกใส่ 'final_metrics_summary.csv')")
print("="*50)
print(summary_df.to_string(index=False))
print("="*50)

# 3. อัปเดตรูป SHAP Summary Plot ล่าสุด
valid_t1 = df_model.dropna(subset=['target_t1'] + feature_cols).reset_index(drop=True)
X_final = valid_t1[feature_cols].astype(float)
y_final = valid_t1['target_t1'].astype(float)

final_model = XGBRegressor(**best_params)
final_model.fit(X_final, y_final)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_final)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_final, show=False)
plt.title('Final Optimized Model: SHAP Feature Importance (Phayao PM2.5)', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('shap_summary_plot_final.png', dpi=300)
plt.close()

print("✅ อัปเดตรูปภาพ 'shap_summary_plot_final.png' สำเร็จ!")