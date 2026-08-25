import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# ตั้งค่าสไตล์กราฟสำหรับ Paper วิชาการ
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# 1. โหลดข้อมูล
df = pd.read_csv('merged_pm25_gee_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

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

# ---------------------------------------------------------
# GRAPH 1: Performance Metrics across Lead Times (t+1 to t+7)
# ---------------------------------------------------------
print("⏳ กำลังประเมินผล $t+1$ ถึง $t+7$ และวาด Graph 1...")

tscv = TimeSeriesSplit(n_splits=5)
horizons = list(range(1, 8))
mae_list, rmse_list, r2_list = [], [], []

for h in horizons:
    df_model[f'target_t{h}'] = df_model['pm25'].shift(-h)
    valid_df = df_model.dropna(subset=[f'target_t{h}'] + feature_cols).reset_index(drop=True)
    
    X = valid_df[feature_cols].astype(float)
    y = valid_df[f'target_t{h}'].astype(float)
    
    m_list, rm_list, r_list = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=100, learning_rate=0.05, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        m_list.append(mean_absolute_error(y_te, preds))
        rm_list.append(np.sqrt(mean_squared_error(y_te, preds)))
        r_list.append(r2_score(y_te, preds))
        
    mae_list.append(np.mean(m_list))
    rmse_list.append(np.mean(rm_list))
    r2_list.append(np.mean(r_list))

# พลอต Graph 1
fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:red'
ax1.set_xlabel('Forecast Horizon (Days)', fontweight='bold')
ax1.set_ylabel('MAE / RMSE (µg/m³)', color=color, fontweight='bold')
l1 = ax1.plot(horizons, mae_list, marker='o', color='tab:red', linewidth=2, label='MAE')
l2 = ax1.plot(horizons, rmse_list, marker='s', color='tab:orange', linewidth=2, label='RMSE')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('R² Score', color=color, fontweight='bold')
l3 = ax2.plot(horizons, r2_list, marker='^', color='tab:blue', linewidth=2, linestyle='--', label='R²')
ax2.tick_params(axis='y', labelcolor=color)

lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Model Performance Decay Across 7-Day Forecast Horizon', fontsize=12, pad=12, fontweight='bold')
plt.xticks(horizons, [f't+{h}' for h in horizons])
plt.tight_layout()
plt.savefig('fig1_performance_decay.png', dpi=300)
plt.close()
print("✅ บันทึกรูป 'fig1_performance_decay.png' เรียบร้อย!")

# ---------------------------------------------------------
# GRAPH 2: Quantile Regression & 95% Confidence Interval (t+1)
# ---------------------------------------------------------
print("⏳ กำลังเทรน Quantile Models (2.5%, 50%, 97.5%) และวาด Graph 2...")

df_model['target_t1'] = df_model['pm25'].shift(-1)
valid_df = df_model.dropna(subset=['target_t1'] + feature_cols).reset_index(drop=True)

X = valid_df[feature_cols].astype(float)
y = valid_df['target_t1'].astype(float)
dates = valid_df['date']

# เทรน 3 Quantile Models
model_low = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.025, n_estimators=100, learning_rate=0.05, random_state=42)
model_mid = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.500, n_estimators=100, learning_rate=0.05, random_state=42)
model_high = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.975, n_estimators=100, learning_rate=0.05, random_state=42)

model_low.fit(X, y)
model_mid.fit(X, y)
model_high.fit(X, y)

pred_low = model_low.predict(X)
pred_mid = model_mid.predict(X)
pred_high = model_high.predict(X)

# สุ่มเจาะลึกเฉพาะช่วงฤดูฝุ่นวิกฤต 100 วันล่าสุดเพื่อความชัดเจนของภาพ
plot_range = slice(-100, None)

plt.figure(figsize=(12, 5))
plt.plot(dates.iloc[plot_range], y.iloc[plot_range], label='Actual PM2.5', color='black', linewidth=1.5)
plt.plot(dates.iloc[plot_range], pred_mid[plot_range], label='Median Forecast (q0.5)', color='blue', linestyle='--', linewidth=1.5)
plt.fill_between(dates.iloc[plot_range], pred_low[plot_range], pred_high[plot_range], color='skyblue', alpha=0.4, label='95% Prediction Interval')

plt.title('PM2.5 Daily Forecast with 95% Prediction Interval (Phayao Stadium)', fontsize=12, pad=12, fontweight='bold')
plt.xlabel('Date', fontweight='bold')
plt.ylabel('PM2.5 (µg/m³)', fontweight='bold')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('fig2_prediction_interval.png', dpi=300)
plt.close()
print("✅ บันทึกรูป 'fig2_prediction_interval.png' เรียบร้อย!")