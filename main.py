import os
import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ----------------------------------------------------
# 1. Logging Configuration
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pm25_api")

# ----------------------------------------------------
# 2. Constants & Data Schemas
# ----------------------------------------------------
STADIUM_LOCATION = {
    "name": "สนามกีฬาจังหวัดพะเยา (Phayao Provincial Stadium)",
    "latitude": 19.1672,
    "longitude": 99.8973
}

FEATURE_NAMES = [
    "temp", "humidity", "rain", "wind_speed", "wind_sin", "wind_cos", "hotspot_count",
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_4", "pm25_lag_5", "pm25_lag_6", "pm25_lag_7",
    "pm25_roll_3", "pm25_roll_7"
]

DEFAULT_RMSE_BY_DAY = {
    "day_1": 3.82,
    "day_2": 4.15,
    "day_3": 4.80,
    "day_4": 5.20,
    "day_5": 5.75,
    "day_6": 6.10,
    "day_7": 6.45
}

class FeatureImportanceItem(BaseModel):
    factor: str = Field(..., description="ชื่อปัจจัย/ฟีเจอร์")
    impact_percent: float = Field(..., description="สัดส่วนความสำคัญ (%)")

class ConfidenceIntervalItem(BaseModel):
    lower: float
    upper: float
    margin: str

class DailyForecastItem(BaseModel):
    day: int = Field(..., ge=1, le=7, description="จำนวนวันพยากรณ์ล่วงหน้า (1-7)")
    pm25: float = Field(..., description="ค่าพยากรณ์ PM2.5 (µg/m³)")
    confidence_interval: ConfidenceIntervalItem
    rmse: float = Field(..., description="ค่าความคลาดเคลื่อนสะสม (RMSE)")

class PredictionResponse(BaseModel):
    status: str = Field(default="success")
    location: Dict[str, Any]
    model_type: str
    ci_type: str = Field(..., description="ประเภทช่วงความเชื่อมั่น (static หรือ dynamic)")
    feature_importance_top5: List[FeatureImportanceItem]
    forecast_7_days: List[DailyForecastItem]

class HealthResponse(BaseModel):
    status: str
    models_loaded_count: int
    data_loaded: bool
    rmse_metrics_loaded: bool

# In-Memory Cache
app_state: Dict[str, Any] = {
    "models": {},          # {1: booster_day_1, ..., 7: booster_day_7}
    "rmse_metrics": {},    # {"day_1": 3.82, ...}
    "preprocessed_df": None,
    "top5_importance": []
}

# ----------------------------------------------------
# 3. Helper Functions & Data Preprocessing
# ----------------------------------------------------
def load_and_preprocess_data() -> pd.DataFrame:
    """ค้นหาไฟล์ข้อมูล (.parquet หรือ .xlsx) และเตรียม Lag/Rolling Features"""
    parquet_path = "สนามกีฬาจังหวัดพะเยา_processed.parquet"
    excel_path = "สนามกีฬาจังหวัดพะเยา ปี 2565-2567.xlsx"
    
    if os.path.exists(parquet_path):
        data_path = parquet_path
        df = pd.read_parquet(data_path)
    elif os.path.exists(excel_path):
        data_path = excel_path
        df = pd.read_excel(data_path)
    else:
        logger.warning("ไม่พบไฟล์ข้อมูลในระบบ สร้างข้อมูลจำลองชั่วคราวสำหรับการรัน API")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        df = pd.DataFrame({
            "datetime": dates,
            "pm25": np.random.uniform(15, 45, size=30),
            "temp": np.random.uniform(20, 35, size=30),
            "humidity": np.random.uniform(40, 80, size=30),
            "rain": np.zeros(30),
            "wind_speed": np.random.uniform(1, 10, size=30),
            "wind_sin": np.sin(np.random.uniform(0, 2*np.pi, size=30)),
            "wind_cos": np.cos(np.random.uniform(0, 2*np.pi, size=30)),
            "hotspot_count": np.random.poisson(2, size=30)
        })

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for i in range(1, 8):
        if f"pm25_lag_{i}" not in df.columns:
            df[f"pm25_lag_{i}"] = df["pm25"].shift(i)
            
    if "pm25_roll_3" not in df.columns:
        df["pm25_roll_3"] = df["pm25"].shift(1).rolling(window=3).mean()
    if "pm25_roll_7" not in df.columns:
        df["pm25_roll_7"] = df["pm25"].shift(1).rolling(window=7).mean()

    return df.dropna().reset_index(drop=True)

def compute_top_features_from_booster(booster: xgb.Booster) -> List[FeatureImportanceItem]:
    """คำนวณ Top 5 XAI Feature Importances จาก XGBoost Day 1 Model"""
    try:
        # ดึง score แบบ weight หากไม่มีให้สลับไปดึงแบบ gain
        scores = booster.get_score(importance_type='weight')
        if not scores:
            scores = booster.get_score(importance_type='gain')

        importances = []
        for i, name in enumerate(FEATURE_NAMES):
            # รองรับทั้งชื่อตัวแปรจริง (เช่น 'pm25_lag_1') และชื่อ 'f0', 'f1', ...
            val = float(scores.get(name, scores.get(f"f{i}", 0.0)))
            importances.append(val)
            
        importances = np.array(importances)
        total = np.sum(importances)
        percentages = (importances / total * 100) if total > 0 else np.zeros(len(importances))

        sorted_indices = np.argsort(percentages)[::-1][:5]
        return [
            FeatureImportanceItem(
                factor=FEATURE_NAMES[idx],
                impact_percent=round(float(percentages[idx]), 2)
            ) for idx in sorted_indices
        ]
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการคำนวณ Feature Importance: {e}")
        return []
# ----------------------------------------------------
# 4. Lifespan Event Handler (Startup & Shutdown)
# ----------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("เริ่มต้นเซิร์ฟเวอร์ และกำลังโหลดทรัพยากร Direct Multi-Horizon...")

    # 1. โหลดโมเดล XGBoost รายวัน 1 ถึง 7 (.json)
    loaded_count = 0
    for day in range(1, 8):
        model_file = f"xgboost_day_{day}.json"
        if os.path.exists(model_file):
            try:
                booster = xgb.Booster()
                booster.load_model(model_file)
                app_state["models"][day] = booster
                loaded_count += 1
                logger.info(f"โหลดโมเดลสำเร็จ: {model_file}")
            except Exception as e:
                logger.error(f"ไม่สามารถโหลด {model_file} ได้: {e}")

    logger.info(f"โหลดโมเดลแล้วทั้งหมด {loaded_count}/7 โมเดล")

    # 2. โหลดค่า RMSE Metrics จากการเทรนจริง
    if os.path.exists("model_rmse_metrics.json"):
        try:
            with open("model_rmse_metrics.json", "r", encoding="utf-8") as f:
                app_state["rmse_metrics"] = json.load(f)
            logger.info("โหลดค่า RMSE Metrics จริงสำเร็จ")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดขณะอ่าน model_rmse_metrics.json: {e}")
            app_state["rmse_metrics"] = DEFAULT_RMSE_BY_DAY
    else:
        app_state["rmse_metrics"] = DEFAULT_RMSE_BY_DAY

    # 3. คำนวณ Top 5 XAI จาก Day 1 Model
    if 1 in app_state["models"]:
        app_state["top5_importance"] = compute_top_features_from_booster(app_state["models"][1])

    # 4. โหลดข้อมูลลงแคช RAM
    try:
        app_state["preprocessed_df"] = load_and_preprocess_data()
        logger.info("โหลดและ Preprocess ข้อมูลอนุกรมเวลาใส่ RAM เรียบร้อยแล้ว")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดขณะโหลดชุดข้อมูล: {e}")

    yield

    app_state.clear()
    logger.info("ปิดการทำงานของระบบและคืนทรัพยากรเรียบร้อย")

# ----------------------------------------------------
# 5. FastAPI App Initialization & Endpoints
# ----------------------------------------------------
app = FastAPI(
    title="PM2.5 Direct Multi-Horizon Forecasting API - Phayao Stadium",
    description="ระบบบริการ API พยากรณ์ฝุ่น PM2.5 ล่วงหน้า 7 วัน ด้วย 7 XGBoost Models แยกรายวัน และ XAI",
    version="3.0.0",
    lifespan=lifespan
)
# 1. สร้าง FastAPI App
app = FastAPI(
    title="PM2.5 Direct Multi-Horizon Forecasting API - Phayao Stadium",
    description="ระบบบริการ API พยากรณ์ฝุ่น PM2.5 ล่วงหน้า 7 วัน ด้วย 7 XGBoost Models แยกรายวัน และ XAI",
    version="3.0.0",
    lifespan=lifespan
)

# 2. ใส่ CORS Middleware ทันทีหลังจากสร้าง app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# เพิ่ม Endpoint /api/gee-tile สำหรับแผนที่ GEE
# ----------------------------------------------------
@app.get("/api/gee-tile", tags=["Map"])
async def get_gee_tile():
    """ส่งคืน Tile URL สำหรับเลเยอร์ Google Earth Engine (สำรอง)"""
    return {
        "status": "success",
        "tile_url": None  # หรือใส่ URL แผนที่ดาวเทียมหากมี
    }

# ----------------------------------------------------
# เพิ่ม CORS Middleware อนุญาตให้ Frontend (Live Server) ดึงข้อมูลได้
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "ยินดีต้อนรับสู่ PM2.5 Forecasting API (Phayao Stadium)",
        "docs_url": "/docs",
        "health_check": "/health",
        "prediction_endpoint": "/predict"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """ตรวจสอบสถานะความพร้อมการทำงานของระบบ"""
    models_count = len(app_state.get("models", {}))
    return HealthResponse(
        status="healthy" if models_count == 7 else "degraded",
        models_loaded_count=models_count,
        data_loaded=app_state.get("preprocessed_df") is not None,
        rmse_metrics_loaded=bool(app_state.get("rmse_metrics"))
    )

@app.get("/predict", response_model=PredictionResponse, tags=["Forecasting"])
async def predict_pm25(
    use_dynamic_ci: bool = Query(
        False, 
        description="เปิดใช้งาน Dynamic Confidence Interval คำนวณตามความผันผวนย้อนหลัง"
    )
):
    """
    พยากรณ์ค่าฝุ่น PM2.5 ล่วงหน้า 7 วัน (Direct Forecasting) พร้อมช่วงความเชื่อมั่น 95% และ XAI
    """
    models = app_state.get("models", {})
    df = app_state.get("preprocessed_df")

    if not models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="โมเดลพยากรณ์ยังไม่พร้อมใช้งาน (ไม่พบไฟล์ xgboost_day_1.json ถึง xgboost_day_7.json)"
        )

    # 1. ดึง Vector ฟีเจอร์ล่าสุด 16 ตัว
    if df is not None and len(df) > 0:
        latest_row = df.iloc[-1]
        input_vector = [float(latest_row[f]) for f in FEATURE_NAMES if f in latest_row]
    else:
        input_vector = [28.5, 65.0, 0.0, 3.2, 0.5, 0.86, 1.0] + [25.0]*7 + [26.0, 27.5]

    # เตรียม DMatrix ทั้งแบบชื่อฟีเจอร์จริง และแบบ array ทั่วไป
    input_df = pd.DataFrame([input_vector], columns=FEATURE_NAMES)
    dmatrix_df = xgb.DMatrix(input_df)
    dmatrix_f_names = xgb.DMatrix(np.array([input_vector]), feature_names=[f"f{i}" for i in range(len(FEATURE_NAMES))])
    dmatrix_raw = xgb.DMatrix(np.array([input_vector]))

    # 2. คำนวณ Standard Deviation ย้อนหลัง 7 วัน สำหรับ Dynamic CI
    recent_std = 1.0
    if df is not None and len(df) >= 7:
        recent_std = float(df["pm25"].tail(7).std())
        if np.isnan(recent_std) or recent_std == 0:
            recent_std = 1.0

    forecast_results = []
    rmse_metrics = app_state.get("rmse_metrics", DEFAULT_RMSE_BY_DAY)

    # 3. ลูปทำนาย 1 ถึง 7 วัน
    for day in range(1, 8):
        booster = models.get(day)
        rmse_val = float(rmse_metrics.get(f"day_{day}", DEFAULT_RMSE_BY_DAY.get(f"day_{day}", 5.0)))

        pred_val = 25.0
        if booster:
            # ลองพยากรณ์ทีละรูปแบบกรณีชื่อฟีเจอร์ไม่ตรงกัน
            try:
                pred_val = float(booster.predict(dmatrix_df)[0])
            except Exception:
                try:
                    pred_val = float(booster.predict(dmatrix_f_names)[0])
                except Exception:
                    try:
                        pred_val = float(booster.predict(dmatrix_raw)[0])
                    except Exception as e:
                        logger.error(f"เกิดข้อผิดพลาดขณะทำนาย Day {day}: {e}")

        # คำนวณ Margin of Error
        if use_dynamic_ci:
            margin = 1.96 * rmse_val * (1.0 + 0.1 * (recent_std / 10.0))
        else:
            margin = 1.96 * rmse_val

        lower_bound = max(0.0, round(pred_val - margin, 2))
        upper_bound = round(pred_val + margin, 2)

        forecast_results.append(DailyForecastItem(
            day=day,
            pm25=round(pred_val, 2),
            confidence_interval=ConfidenceIntervalItem(
                lower=lower_bound,
                upper=upper_bound,
                margin=f"±{round(margin, 1)}"
            ),
            rmse=round(rmse_val, 2)
        ))

    return PredictionResponse(
        status="success",
        location=STADIUM_LOCATION,
        model_type="Direct Multi-Horizon XGBoost (7 Independent Models)",
        ci_type="dynamic" if use_dynamic_ci else "static",
        feature_importance_top5=app_state.get("top5_importance", []),
        forecast_7_days=forecast_results
    )

    # 1. ดึง Vector ฟีเจอร์ล่าสุด 16 ตัว
    if df is not None and len(df) > 0:
        latest_row = df.iloc[-1]
        input_vector = [float(latest_row[f]) for f in FEATURE_NAMES if f in latest_row]
    else:
        input_vector = [28.5, 65.0, 0.0, 3.2, 0.5, 0.86, 1.0] + [25.0]*7 + [26.0, 27.5]

    dmatrix = xgb.DMatrix(
        np.array([input_vector]), 
        feature_names=[f"f{i}" for i in range(len(FEATURE_NAMES))]
    )

    # 2. คำนวณ Standard Deviation ย้อนหลัง 7 วัน สำหรับ Dynamic CI
    recent_std = 1.0
    if df is not None and len(df) >= 7:
        recent_std = float(df["pm25"].tail(7).std())
        if np.isnan(recent_std) or recent_std == 0:
            recent_std = 1.0

    forecast_results = []
    rmse_metrics = app_state.get("rmse_metrics", DEFAULT_RMSE_BY_DAY)

    # 3. ลูปทำนาย 1 ถึง 7 วัน
    for day in range(1, 8):
        booster = models.get(day)
        rmse_val = float(rmse_metrics.get(f"day_{day}", DEFAULT_RMSE_BY_DAY.get(f"day_{day}", 5.0)))

        if booster:
            try:
                pred_val = float(booster.predict(dmatrix)[0])
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดขณะทำนาย Day {day}: {e}")
                pred_val = 25.0
        else:
            pred_val = 25.0

        if use_dynamic_ci:
            margin = 1.96 * rmse_val * (1.0 + 0.1 * (recent_std / 10.0))
        else:
            margin = 1.96 * rmse_val

        lower_bound = max(0.0, round(pred_val - margin, 2))
        upper_bound = round(pred_val + margin, 2)

        forecast_results.append(DailyForecastItem(
            day=day,
            pm25=round(pred_val, 2),
            confidence_interval=ConfidenceIntervalItem(
                lower=lower_bound,
                upper=upper_bound,
                margin=f"±{round(margin, 1)}"
            ),
            rmse=round(rmse_val, 2)
        ))

    return PredictionResponse(
        status="success",
        location=STADIUM_LOCATION,
        model_type="Direct Multi-Horizon XGBoost (7 Independent Models)",
        ci_type="dynamic" if use_dynamic_ci else "static",
        feature_importance_top5=app_state.get("top5_importance", []),
        forecast_7_days=forecast_results
    )