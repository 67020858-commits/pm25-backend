import requests
import pandas as pd
import numpy as np

# พิกัดสนามกีฬาจังหวัดพะเยา
LAT = 19.2200
LON = 100.0500

print("🚀 กำลังดึงข้อมูล PM2.5 และสภาพอากาศจาก Open-Meteo...")

# 1. ดึงข้อมูล PM2.5 ย้อนหลัง
air_quality_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&hourly=pm2_5&past_days=90"
res_aq = requests.get(air_quality_url).json()

df_aq = pd.DataFrame({
    'time': pd.to_datetime(res_aq['hourly']['time']),
    'pm25': res_aq['hourly']['pm2_5']
})

# 2. ดึงข้อมูลสภาพอากาศ (ความเร็วลม, อุณหภูมิ, ความชื้น, ปริมาณฝน) ย้อนหลัง
weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&past_days=90"
res_w = requests.get(weather_url).json()

df_weather = pd.DataFrame({
    'time': pd.to_datetime(res_w['hourly']['time']),
    'temperature': res_w['hourly']['temperature_2m'],
    'humidity': res_w['hourly']['relative_humidity_2m'],
    'wind_speed': res_w['hourly']['wind_speed_10m'],
    'precipitation': res_w['hourly']['precipitation']
})

# 3. รวมข้อมูลเข้าด้วยกันตามเวลา
df = pd.merge(df_aq, df_weather, on='time')

# 4. แปลงข้อมูลเป็นรายวัน (Daily Resample) เพื่อให้เหมาะกับโมเดลรายวัน
df.set_index('time', inplace=True)
df_daily = df.resample('D').agg({
    'pm25': 'mean',          # PM2.5 เฉลี่ยรายวัน
    'temperature': 'mean',   # อุณหภูมิเฉลี่ย
    'humidity': 'mean',      # ความชื้นเฉลี่ย
    'wind_speed': 'mean',    # ความเร็วลมเฉลี่ย
    'precipitation': 'sum'   # ปริมาณฝนสะสมรายวัน
}).dropna()

# บันทึกไฟล์ไว้ใช้งานในขั้นตอนประเมินผล
df_daily.to_csv('phayao_dataset.csv')
print("✅ ดึงข้อมูลสำเร็จ! บันทึกไฟล์เป็น 'phayao_dataset.csv' เรียบร้อยแล้ว")
print(df_daily.head())