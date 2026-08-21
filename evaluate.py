import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    return mae, rmse, mape

# 1. โหลดข้อมูล
df = pd.read_csv('phayao_dataset.csv', index_col='time', parse_dates=True)
y = df['pm25']

# 2. ประเมินผล Cross-Validation (5 Folds)
tscv = TimeSeriesSplit(n_splits=5)
results_base, results_garch = [], []

print("⏳ กำลังทดสอบโมเดล ARIMA-GARCH...")

for fold, (train_idx, test_idx) in enumerate(tscv.split(y), start=1):
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # --- 1. SARIMA (เดิม) ---
    m1 = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 0, 0, 7)).fit(disp=False)
    pred_base = m1.forecast(len(y_test))
    results_base.append(calculate_metrics(y_test, pred_base))
    
    # --- 2. ARIMA-GARCH (ใหม่) ---
    # ขั้นตอนที่ A: เทรน ARIMA เพื่อหา Residuals (ค่าความคลาดเคลื่อน)
    arima_res = m1.resid
    
    # ขั้นตอนที่ B: นำ Residuals ไปเทรน GARCH(1,1) โมเดลเพื่อจับ Volatility
    garch = arch_model(arima_res, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch.fit(disp='off')
    
    # พยากรณ์ Variance
    garch_forecast = garch_fit.forecast(horizon=len(y_test))
    volatility = np.sqrt(garch_forecast.variance.values[-1, :])
    
    # ปรับแต่งค่าทำนายด้วยการถ่วงน้ำหนักความผันผวน
    # หากความผันผวนสูง ให้ปรับแต่งค่าทำนายเข้าหาแนวโน้มหลัก
    adj_factor = np.where(volatility > np.median(volatility), 0.98, 1.01)
    pred_garch = pred_base * adj_factor
    
    results_garch.append(calculate_metrics(y_test, pred_garch))

# 3. แสดงตารางเปรียบเทียบ
df_base = pd.DataFrame(results_base, columns=['MAE', 'RMSE', 'MAPE'])
df_garch = pd.DataFrame(results_garch, columns=['MAE', 'RMSE', 'MAPE'])

summary_df = pd.DataFrame({
    'Metric': ['MAE (µg/m³)', 'RMSE (µg/m³)', 'MAPE (%)'],
    'SARIMA (เดิม)': [df_base['MAE'].mean(), df_base['RMSE'].mean(), df_base['MAPE'].mean()],
    'ARIMA-GARCH (ใหม่)': [df_garch['MAE'].mean(), df_garch['RMSE'].mean(), df_garch['MAPE'].mean()]
})

print("\n" + "="*70)
print("📊 ผลการเปรียบเทียบประสิทธิภาพ SARIMA vs ARIMA-GARCH")
print("="*70)
print(summary_df.to_string(index=False))
print("="*70)