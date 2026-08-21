import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# 1. PREPARE MULTI-STEP DATASET (Input 30 Days -> Output 7 Days)
# ==========================================
def create_multistep_dataset(df, feature_cols, target_col, input_window=30, output_window=7):
    X, y = [], []
    for i in range(len(df) - input_window - output_window + 1):
        # Input features รวมทั้ง Lag และ Weather/Hotspots
        X.append(df[feature_cols].iloc[i : i + input_window].values)
        # Target PM2.5 ล่วงหน้า 7 วัน
        y.append(df[target_col].iloc[i + input_window : i + input_window + output_window].values)
    return np.array(X), np.array(y)

# สมมติ df_merged คือ DataFrame ที่รวม Weather/Hotspot และ Impute ค่าเรียบร้อยแล้ว
# feature_cols = ['PM2.5', 'temp', 'humidity', 'rain', 'wind_speed', 'hotspot_count']
# X shape: (Samples, 30, Features), y shape: (Samples, 7)

# ==========================================
# 2. XGBOOST (Direct Multi-step Regressor)
# ==========================================
def train_xgboost_multistep(X_train, y_train, X_test, y_test):
    # แปลง 3D input (Samples, 30, Features) เป็น 2D Flat สำหรับ XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    xgb_preds = np.zeros(y_test.shape)
    
    # เทรน 7 โมเดลย่อย สำหรับวันพยากรณ์ t+1 ถึง t+7
    for step in range(7):
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train_flat, y_train[:, step])
        xgb_preds[:, step] = model.predict(X_test_flat)
        
    return xgb_preds

# ==========================================
# 3. LSTM (Deep Learning Sequence-to-Sequence)
# ==========================================
def train_lstm_multistep(X_train, y_train, X_test, y_test):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(7)  # Output 7 วันล่วงหน้า
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
    
    lstm_preds = model.predict(X_test)
    return lstm_preds

# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================
def evaluate_multistep(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{model_name}] Average 7-Day MAE  : {mae:.2f} ug/m3")
    print(f"[{model_name}] Average 7-Day RMSE : {rmse:.2f} ug/m3")