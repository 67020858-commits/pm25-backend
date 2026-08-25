import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# 1. โหลดข้อมูลที่ Merge ไว้แล้ว
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
# ส่วนที่ 1: Baseline Comparison (ประเมินที่ t+1)
# ---------------------------------------------------------
print("⏳ กำลังเทรนและเปรียบเทียบ Baseline Models (t+1)...")

df_model['target_t1'] = df_model['pm25'].shift(-1)
valid_df = df_model.dropna(subset=['target_t1'] + feature_cols).reset_index(drop=True)

X = valid_df[feature_cols].astype(float)
y = valid_df['target_t1'].astype(float)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    'XGBoost (Proposed)': XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=100, learning_rate=0.05, random_state=42)
}

tscv = TimeSeriesSplit(n_splits=5)
comparison_results = []

for name, model in models.items():
    mae_l, rmse_l, r2_l = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        mae_l.append(mean_absolute_error(y_te, preds))
        rmse_l.append(np.sqrt(mean_squared_error(y_te, preds)))
        r2_l.append(r2_score(y_te, preds))
        
    comparison_results.append({
        'Model': name,
        'MAE': round(np.mean(mae_l), 3),
        'RMSE': round(np.mean(rmse_l), 3),
        'R2': round(np.mean(r2_l), 3)
    })

print("\n" + "="*50)
print("📊 ตารางเปรียบเทียบ Baseline Models (สำหรับ Paper)")
print("="*50)
print(pd.DataFrame(comparison_results).to_string(index=False))

# ---------------------------------------------------------
# ส่วนที่ 2: SHAP Analysis & Visualization
# ---------------------------------------------------------
print("\n⏳ กำลังคำนวณ SHAP Values และบันทึกรูปภาพ...")

xgb_model = models['XGBoost (Proposed)']
xgb_model.fit(X, y)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X)

# เซฟรูป SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title('SHAP Feature Importance (Phayao PM2.5 Forecast)', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300)
plt.close()

print("✅ บันทึกรูป 'shap_summary_plot.png' เรียบร้อยแล้ว!")