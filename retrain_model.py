import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from sklearn.metrics import mean_squared_error

# 1. กำหนดค่าต่างๆ
DATA_PATH = "pm25_data.parquet"
# สำคัญ: ต้องเรียง Feature ให้ตรงกับใน main.py ของคุณเป๊ะๆ
FEATURE_NAMES = [
    "pm25", "temp", "humidity", "rain", "wind_speed", "wind_deg", "hotspot_count",
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_4", "pm25_lag_5", "pm25_lag_6", "pm25_lag_7",
    "rolling_mean_3", "rolling_mean_7", "wind_sin", "wind_cos"
]

def retrain_models():
    print("🚀 เริ่มกระบวนการ Retrain โมเดล...")
    
    # 2. โหลดข้อมูลล่าสุด
    if not os.path.exists(DATA_PATH):
        print(f"❌ ไม่พบไฟล์ข้อมูล {DATA_PATH}")
        return
        
    df = pd.read_parquet(DATA_PATH)
    print(f"📊 โหลดข้อมูลสำเร็จ: {len(df)} แถว")
    
    rmse_metrics = {}
    
    # 3. วนลูปสร้างโมเดล 7 วัน
    for day in range(1, 8):
        print(f"กำลังเทรนโมเดลสำหรับล่วงหน้า {day} วัน (Day {day})...")
        
        # สร้าง Target Column (ค่า PM2.5 ในอนาคต day วัน)
        df_train = df.copy()
        df_train['target'] = df_train['pm25'].shift(-day)
        
        # ตัดแถวที่มีค่า NaN (แถวท้ายๆ ที่ยังไม่มีอนาคตให้ทำนาย)
        df_train = df_train.dropna(subset=['target'] + FEATURE_NAMES)
        
        X = df_train[FEATURE_NAMES]
        y = df_train['target']
        
        if len(X) < 30: # เช็กว่ามีข้อมูลพอเทรนไหม
            print("⚠️ ข้อมูลน้อยเกินไป ข้ามการเทรน")
            continue
            
        # สร้าง DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # ตั้งค่า Hyperparameters ของ XGBoost
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # ฝึกสอนโมเดล
        booster = xgb.train(params, dtrain, num_boost_round=100)
        
        # บันทึกโมเดล
        model_filename = f"xgboost_day_{day}.json"
        booster.save_model(model_filename)
        
        # คำนวณ RMSE เพื่อนำไปใช้อัปเดต Confidence Interval บนเว็บ
        predictions = booster.predict(dtrain)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        rmse_metrics[f"day_{day}"] = rmse
        print(f"✅ บันทึก {model_filename} เรียบร้อย (RMSE: {rmse:.2f})")
        
    # 4. บันทึกไฟล์ RMSE สำหรับ Frontend
    with open("rmse_metrics.json", "w") as f:
        json.dump(rmse_metrics, f, indent=4)
    print("🎯 บันทึกไฟล์ rmse_metrics.json สำเร็จ!")
    print("🎉 กระบวนการ Retrain เสร็จสมบูรณ์!")

if __name__ == "__main__":
    retrain_models()