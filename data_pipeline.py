# ==========================================
# ไฟล์: data_pipeline.py
# ==========================================
import numpy as np
import pandas as pd

# 1. โค้ดแปลงทิศทางลม (Wind Direction Encoding)
def encode_wind_components(df):
    wind_rad = np.radians(df['wind_direction'])
    df['wind_u'] = -df['wind_speed'] * np.sin(wind_rad)
    df['wind_v'] = -df['wind_speed'] * np.cos(wind_rad)
    return df

# 2. โค้ดคำนวณ Weighted Hotspot ตามระยะทางและทิศทางลม
def calculate_weighted_hotspots(station_lat, station_lon, hotspots_df, wind_deg, alpha=1.5):
    if hotspots_df.empty:
        return 0.0
    
    R = 6371.0 # รัศมีโลก (km)
    lat1, lon1 = np.radians(station_lat), np.radians(station_lon)
    lat2, lon2 = np.radians(hotspots_df['lat']), np.radians(hotspots_df['lon'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distances = 2 * R * np.arcsin(np.sqrt(a))
    
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360

    angle_diff = np.radians(bearing - wind_deg)
    upwind_factor = 1.0 + np.cos(angle_diff)

    epsilon = 1.0
    distance_weight = 1.0 / ((distances + epsilon) ** alpha)

    return np.sum(distance_weight * upwind_factor)

# ฟังก์ชันรวมสำหรับการเตรียมข้อมูล
def prepare_dataset(raw_df, hotspots_df):
    df = encode_wind_components(raw_df)
    # คำนวณ Ventilation Coefficient
    df['ventilation_coeff'] = df['pblh'] * df['wind_speed']
    # เรียกใช้คำนวณ weighted hotspot...
    return df