import pickle
import xgboost as xgb
import os

def convert_xgb_pkl_to_json(input_pkl_path: str, output_json_path: str):
    print(f"กำลังโหลดโมเดล XGBoost จาก: {input_pkl_path}")
    if not os.path.exists(input_pkl_path):
        print(f"Error: ไม่พบไฟล์ {input_pkl_path}")
        return

    with open(input_pkl_path, "rb") as f:
        model = pickle.load(f)

    # ดึงออบเจกต์ Booster ออกมาจาก Scikit-Learn Wrapper หรือ Booster Object
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif isinstance(model, xgb.Booster):
        booster = model
    else:
        print("ไม่รองรับโครงสร้างโมเดลประเภทนี้")
        return

    # บันทึกเป็นไฟล์ JSON
    booster.save_model(output_json_path)
    print(f"แปลงโมเดลสำเร็จ! บันทึกไฟล์ JSON ปลอดภัยไว้ที่: {output_json_path}")

if __name__ == "__main__":
    convert_xgb_pkl_to_json(
        input_pkl_path="xgboost_7days_model.pkl",
        output_json_path="xgboost_7days_model.json"
    )