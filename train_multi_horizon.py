import os
import glob
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------
# 1. Logging Configuration
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_pipeline")

FEATURE_NAMES = [
    "temp", "humidity", "rain", "wind_speed", "wind_sin", "wind_cos", "hotspot_count",
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_4", "pm25_lag_5", "pm25_lag_6", "pm25_lag_7",
    "pm25_roll_3", "pm25_roll_7"
]

def load_and_clean_all_data() -> pd.DataFrame:
    """โหลดไฟล์ .parquet, .csv หรือไฟล์ .xlsx ทุกปี แล้วทำการ Clean / Rename คอลัมน์อัตโนมัติ"""
    # 1. เช็กไฟล์ Parquet หรือ CSV ที่เคยประมวลผลไว้แล้ว
    if os.path.exists("สนามกีฬาจังหวัดพะเยา_processed.parquet"):
        try:
            logger.info("พบไฟล์ Parquet ที่ประมวลผลแล้ว")
            return pd.read_parquet("สนามกีฬาจังหวัดพะเยา_processed.parquet")
        except Exception:
            pass

    if os.path.exists("สนามกีฬาจังหวัดพะเยา_processed.csv"):
        logger.info("พบไฟล์ CSV ที่ประมวลผลแล้ว")
        return pd.read_csv("สนามกีฬาจังหวัดพะเยา_processed.csv")

    # 2. อ่านไฟล์ Excel ทั้งหมดในโฟลเดอร์
    excel_files = sorted(glob.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError("ไม่พบไฟล์ข้อมูล .xlsx, .parquet หรือ .csv ในโฟลเดอร์ปัจจุบัน")

    logger.info(f"พบไฟล์ Excel ทั้งหมด {len(excel_files)} ไฟล์: {excel_files}")
    dfs = []

    for file_path in excel_files:
        try:
            df_raw = pd.read_excel(file_path, header=None)
            
            header_idx = None
            for idx, row in df_raw.iterrows():
                if row.astype(str).str.contains("วันที่").any():
                    header_idx = idx
                    break

            if header_idx is not None:
                df_year = pd.read_excel(file_path, skiprows=header_idx)
                
                date_col = [c for c in df_year.columns if "วันที่" in str(c) or "date" in str(c).lower()]
                pm25_col = [c for c in df_year.columns if "PM2.5" in str(c) or "pm25" in str(c).lower()]

                if date_col and pm25_col:
                    df_year = df_year.rename(columns={
                        date_col[0]: "datetime",
                        pm25_col[0]: "pm25"
                    })
                    dfs.append(df_year)
        except Exception as e:
            logger.warning(f"ข้ามไฟล์ {file_path}: {e}")

    if not dfs:
        raise ValueError("ไม่สามารถสกัดคอลัมน์วันที่และ PM2.5 จากไฟล์ Excel ได้")

    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined["pm25"] = pd.to_numeric(df_combined["pm25"], errors="coerce")
    df_combined["datetime"] = pd.to_datetime(df_combined["datetime"], dayfirst=True, errors="coerce")
    
    df_clean = df_combined.dropna(subset=["datetime", "pm25"]).sort_values("datetime").reset_index(drop=True)
    logger.info(f"รวมข้อมูลสำเร็จทั้งหมด {len(df_clean)} แถว (ช่วงวันที่ {df_clean['datetime'].min().strftime('%Y-%m-%d')} ถึง {df_clean['datetime'].max().strftime('%Y-%m-%d')})")
    
    return df_clean

def train_direct_multi_horizon_models():
    # 1. โหลดข้อมูล
    df = load_and_clean_all_data()

    # 2. เติมคอลัมน์ฟีเจอร์พื้นฐาน
    default_features = {
        "temp": 28.5,
        "humidity": 65.0,
        "rain": 0.0,
        "wind_speed": 3.2,
        "wind_sin": 0.5,
        "wind_cos": 0.86,
        "hotspot_count": 1.0
    }
    for col, default_val in default_features.items():
        if col not in df.columns:
            df[col] = default_val

    # 3. สร้าง Feature Engineering (Lags 1-7 & Rolling Means)
    for i in range(1, 8):
        if f"pm25_lag_{i}" not in df.columns:
            df[f"pm25_lag_{i}"] = df["pm25"].shift(i)
            
    if "pm25_roll_3" not in df.columns:
        df["pm25_roll_3"] = df["pm25"].shift(1).rolling(window=3).mean()
    if "pm25_roll_7" not in df.columns:
        df["pm25_roll_7"] = df["pm25"].shift(1).rolling(window=7).mean()

    # 4. สร้าง Target ($y_{t+h}$) สำหรับ Direct Forecasting Day 1 ถึง Day 7
    for h in range(1, 8):
        df[f"target_day_{h}"] = df["pm25"].shift(-h)

    df_clean = df.dropna().reset_index(drop=True)

    # 5. บันทึกไฟล์ข้อมูลแคช (พยายามเซฟเป็น Parquet ก่อน ถ้าไม่มี pyarrow ให้เซฟเป็น CSV)
    try:
        parquet_out = "สนามกีฬาจังหวัดพะเยา_processed.parquet"
        df_clean.to_parquet(parquet_out, index=False, compression="snappy")
        logger.info(f"บันทึกไฟล์แคชเรียบร้อย: {parquet_out}")
    except Exception as e:
        csv_out = "สนามกีฬาจังหวัดพะเยา_processed.csv"
        df_clean.to_csv(csv_out, index=False)
        logger.info(f"บันทึกเป็นไฟล์สำรองเรียบร้อย: {csv_out}")

    # 6. Split ข้อมูล (Time-Series Split: 80% Train, 20% Test)
    train_size = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:train_size]
    test_df = df_clean.iloc[train_size:]

    X_train = train_df[FEATURE_NAMES]
    X_test = test_df[FEATURE_NAMES]

    rmse_results = {}
    logger.info("==========================================")
    logger.info(" เริ่มต้นการฝึกสอน Direct Multi-Horizon Models (7 Models) ")
    logger.info("==========================================")

    # 7. เทรน XGBoost แยก 7 โมเดลรายวัน
    for day in range(1, 8):
        y_train = train_df[f"target_day_{day}"]
        y_test = test_df[f"target_day_{day}"]

        model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        rmse_results[f"day_{day}"] = round(rmse, 2)

        model_filename = f"xgboost_day_{day}.json"
        model.save_model(model_filename)
        logger.info(f"[Day {day}] RMSE: {rmse:.2f} µg/m³ -> บันทึกโมเดลสำเร็จ: {model_filename}")

    # 8. บันทึกไฟล์สรุปผล RMSE
    metrics_filename = "model_rmse_metrics.json"
    with open(metrics_filename, "w", encoding="utf-8") as f:
        json.dump(rmse_results, f, indent=4)
        
    logger.info(f"บันทึกไฟล์สรุปผลค่า RMSE ไว้ที่: {metrics_filename}")
    logger.info("การเทรน 7 โมเดลเสร็จสมบูรณ์!")

if __name__ == "__main__":
    train_direct_multi_horizon_models()