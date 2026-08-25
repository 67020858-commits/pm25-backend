import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# ปิด Warning ของ Optuna ให้ Console สะอาด
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("⏳ กำลังโหลดข้อมูลสำหรับทำ Hyperparameter Tuning...")

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
df_model['target_t1'] = df_model['pm25'].shift(-1)
valid_df = df_model.dropna(subset=['target_t1'] + feature_cols).reset_index(drop=True)

X = valid_df[feature_cols].astype(float)
y = valid_df['target_t1'].astype(float)

tscv = TimeSeriesSplit(n_splits=5)

# 2. ฟังก์ชัน Objective สำหรับ Optuna
def objective(trial):
    params = {
        'objective': 'reg:quantileerror',
        'quantile_alpha': 0.5,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    
    mae_list = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        mae_list.append(mean_absolute_error(y_te, preds))
        
    return np.mean(mae_list)

print("⏳ กำลังเริ่มกระบวนการ Optuna Tuning (50 Trials)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n" + "="*50)
print("🎉 ค้นพบ Best Hyperparameters สำเร็จ!")
print("="*50)
print(f"Best MAE (t+1): {study.best_value:.3f}")
print("Best Parameters:")
for key, value in study.best_params.items():
    print(f"  '{key}': {value if isinstance(value, int) else round(value, 5)},")
print("="*50)