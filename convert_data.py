import pandas as pd
import time
import os

def convert_excel_to_parquet(input_excel_path: str, output_parquet_path: str):
    print(f"กำลังเริ่มแปลงไฟล์: {input_excel_path}")
    start_time = time.time()
    
    # 1. อ่านไฟล์ Excel
    if not os.path.exists(input_excel_path):
        print(f"Error: ไม่พบไฟล์ {input_excel_path}")
        return
        
    df = pd.read_excel(input_excel_path)
    
    # 2. จัดเรียงตามลำดับเวลา
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        
    # 3. คำนวณ Pre-computed Lag & Rolling Features
    for i in range(1, 8):
        df[f"pm25_lag_{i}"] = df["pm25"].shift(i)
        
    df["pm25_roll_3"] = df["pm25"].shift(1).rolling(window=3).mean()
    df["pm25_roll_7"] = df["pm25"].shift(1).rolling(window=7).mean()
    
    # ลบแถวแรกๆ ที่เป็น NaN จากการ Shift ข้อมูล
    df = df.dropna().reset_index(drop=True)
    
    # 4. บันทึกเป็น Apache Parquet (Snappy Compression)
    df.to_parquet(output_parquet_path, index=False, compression="snappy")
    
    elapsed_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_parquet_path) / (1024 * 1024)
    print(f"แปลงไฟล์สำเร็จ! บันทึกไว้ที่: {output_parquet_path}")
    print(f"ขนาดไฟล์ใหม่: {file_size_mb:.2f} MB (ใช้เวลาประมวลผล: {elapsed_time:.2f} วินาที)")

if __name__ == "__main__":
    convert_excel_to_parquet(
        input_excel_path="สนามกีฬาจังหวัดพะเยา ปี 2565-2567.xlsx",
        output_parquet_path="สนามกีฬาจังหวัดพะเยา_processed.parquet"
    )