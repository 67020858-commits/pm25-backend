import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1. โหลดและรวมข้อมูล Excel ทุกปี
excel_files = sorted(glob.glob("สนามกีฬาจังหวัดพะเยา ปี *.xlsx"))
df_list = []

print("กำลังโหลดและรวมข้อมูลจากไฟล์ Excel...")
for file in excel_files:
    df_temp = pd.read_excel(file, skiprows=3, header=None)
    df_temp = df_temp.dropna(how="all").iloc[:, :2]
    df_temp.columns = ["Date", "PM2.5"]
    df_list.append(df_temp)

df_all = pd.concat(df_list, ignore_index=True)
df_all["PM2.5"] = pd.to_numeric(df_all["PM2.5"], errors="coerce")
df_all = df_all.dropna(subset=["PM2.5"])

data = df_all["PM2.5"].values

# 2. ทำ Log Transformation เพื่อลดผลกระทบของ Outlier (ค่าฝุ่นพุ่งสูงช่วงวิกฤต)
data_log = np.log1p(data)

# 3. แบ่งข้อมูล Train 80% / Test 20%
train_size = int(len(data) * 0.8)
train_log = data_log[:train_size]
test_data = data[train_size:]

# 4. เทรนโมเดล SARIMA บนข้อมูล Log Scale
print("กำลังเทรนโมเดล SARIMA (Log Scale)...")
model = SARIMAX(
    train_log,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarima_result = model.fit(disp=False)

# 5. ทำนายผลและแปลงค่ากลับด้วย np.expm1()
predictions_log = sarima_result.predict(
    start=len(train_log), end=len(train_log) + len(test_data) - 1
)
predictions = np.expm1(predictions_log)

# 6. ประเมินผลประสิทธิภาพ
mae = mean_absolute_error(test_data, predictions)
rmse = np.sqrt(mean_squared_error(test_data, predictions))
r2 = r2_score(test_data, predictions)

print("\n==========================================")
print("     ผลการวัดประสิทธิภาพ SARIMA (Log Scale)   ")
print("==========================================")
print(f"MAE  : {mae:.2f} ug/m3")
print(f"RMSE : {rmse:.2f} ug/m3")
print(f"R2   : {r2:.2f}")
print("==========================================\n")

# 7. เทรนโมเดลด้วยข้อมูลทั้งหมดแล้วบันทึกไฟล์ .pkl
print("กำลังสร้างและบันทึกโมเดลฉบับสมบูรณ์...")
final_model = SARIMAX(
    data_log,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

joblib.dump(final_model, "sarima_model.pkl")
print("บันทึกโมเดลเรียบร้อยแล้วที่ไฟล์: sarima_model.pkl")