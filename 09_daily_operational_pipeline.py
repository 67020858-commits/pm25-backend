import numpy as np
import pandas as pd
import requests
import json
from xgboost import XGBRegressor

# ==========================================
# Configuration (Discord Webhook URL)
# ==========================================
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1540441523559333929/cII_Kt2K52ubJUZuID_aclTjQJH1c_Gr-F5p4GkO8O4M6z7JOPoTb8sgQYt7AWYXl0cu"

def send_discord_alert(date_str, current_pm25, forecast_list):
    # กำหนดสี Embed ตามระดับฝุ่นปัจจุบัน
    # 0x00FF00 = เขียว, 0xFFFF00 = เหลือง, 0xFF0000 = แดง
    color = 0x00FF00 if current_pm25 < 37.5 else 0xFFFF00 if current_pm25 < 75.0 else 0xFF0000
    
    fields = []
    for h, mid, low, high in forecast_list:
        icon = "🟢" if mid < 37.5 else "🟡" if mid < 75.0 else "🔴"
        fields.append({
            "name": f"Day +{h}",
            "value": f"{icon} **{mid}** µg/m³\n_({low} - {high})_",
            "inline": True
        })

    payload = {
        "username": "Phayao PM2.5 Forecast Bot",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/1163/1163661.png",
        "embeds": [{
            "title": "🚨 รายงานผลพยากรณ์ PM2.5 ล่วงหน้า 7 วัน (สนามกีฬาพะเยา)",
            "description": f"📅 **ข้อมูลประมวลผลล่าสุด:** `{date_str}`\n💨 **ค่าฝุ่นปัจจุบัน:** **{current_pm25:.1f} µg/m³**\n\n📊 **คาดการณ์ล่วงหน้า (พร้อมช่วงความเชื่อมั่น 90%):**",
            "color": color,
            "fields": fields,
            "footer": {
                "text": "เกณฑ์: 🟢 ปลอดภัย (<37.5) | 🟡 เริ่มมีผลกระทบ (37.5-75) | 🔴 วิกฤต (>75)"
            }
        }]
    }

    try:
        res = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        return res.status_code == 204 or res.status_code == 200
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการส่ง Discord: {e}")
        return False

print("⏳ กำลังโหลดข้อมูลย้อนหลัง และเตรียมทำนายรายวัน...")

# 1. โหลดข้อมูลรวมล่าสุด
df = pd.read_csv('merged_pm25_gee_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 2. Feature Engineering Pipeline (R² = 0.733)
df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
df['dewpoint_spread'] = df['temp_k'] - df['dewpoint_k']

day_of_year = df['date'].dt.dayofyear
df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)
df['month'] = df['date'].dt.month

df['hotspot_wind_inter'] = df['hotspot_count_50km'] * df['wind_speed']
df['ventilation_index'] = df['wind_speed'] * df['temp_k']
df['hotspot_ema_3'] = df['hotspot_count_50km'].ewm(span=3, adjust=False).mean()

feature_cols = [
    'sin_day', 'cos_day', 'month', 'dewpoint_spread', 'temp_k', 
    'wind_speed', 'precip_m', 'hotspot_wind_inter', 'ventilation_index', 'hotspot_ema_3'
]

for lag in [1, 2, 3, 7]:
    df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
    df[f'hotspot_lag_{lag}'] = df['hotspot_count_50km'].shift(lag)
    feature_cols.extend([f'pm25_lag_{lag}', f'hotspot_lag_{lag}'])

df['pm25_roll_3'] = df['pm25'].shift(1).rolling(3).mean()
df['pm25_roll_7'] = df['pm25'].shift(1).rolling(7).mean()
feature_cols.extend(['pm25_roll_3', 'pm25_roll_7'])

df_model = df.dropna(subset=feature_cols + ['pm25']).reset_index(drop=True)

# ดึงข้อมูลแถวล่าสุดสำหรับทำนายอนาคต
latest_row = df_model.iloc[-1:]
latest_date = latest_row['date'].dt.strftime('%Y-%m-%d').values[0]
latest_pm25 = latest_row['pm25'].values[0]

# 3. Best Parameters จาก Optuna
best_params = {
    'objective': 'reg:quantileerror',
    'random_state': 42,
    'n_estimators': 238,
    'max_depth': 3,
    'learning_rate': 0.02538,
    'subsample': 0.67587,
    'colsample_bytree': 0.72022,
    'reg_alpha': 0.00115,
    'reg_lambda': 0.01617
}

# 4. พยากรณ์ t+1 ถึง t+7 พร้อม Quantile Interval (5%, 50%, 95%)
forecast_results = []

print("⏳ กำลังเทรน Quantile Models และคำนวณผลทำนาย t+1 ถึง t+7...")
for h in range(1, 8):
    df_model[f'target_t{h}'] = df_model['pm25'].shift(-h)
    valid_df = df_model.dropna(subset=[f'target_t{h}'] + feature_cols).reset_index(drop=True)
    
    X = valid_df[feature_cols].astype(float)
    y = valid_df[f'target_t{h}'].astype(float)
    
    X_input = latest_row[feature_cols].astype(float)
    
    # Train Quantile Models
    model_low = XGBRegressor(**best_params, quantile_alpha=0.05).fit(X, y)
    model_mid = XGBRegressor(**best_params, quantile_alpha=0.50).fit(X, y)
    model_high = XGBRegressor(**best_params, quantile_alpha=0.95).fit(X, y)
    
    p_low = max(0, round(model_low.predict(X_input)[0], 1))
    p_mid = max(0, round(model_mid.predict(X_input)[0], 1))
    p_high = max(0, round(model_high.predict(X_input)[0], 1))
    
    forecast_results.append((h, p_mid, p_low, p_high))

# 5. ส่งการแจ้งเตือนไปยัง Discord
print("⏳ กำลังส่งการแจ้งเตือนไปยัง Discord...")
if send_discord_alert(latest_date, latest_pm25, forecast_results):
    print("\n✅ ส่งการแจ้งเตือนผลพยากรณ์ PM2.5 เข้า Discord เรียบร้อยแล้ว!")
else:
    print("\n❌ ไม่สามารถส่งการแจ้งเตือนไปยัง Discord ได้")