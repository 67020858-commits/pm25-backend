import pickle
import json
import os

def optimize_sarimax_model(input_pkl_path: str):
    if not os.path.exists(input_pkl_path):
        print(f"Error: ไม่พบไฟล์ {input_pkl_path}")
        return

    print(f"กำลังโหลดโมเดล SARIMAX จาก: {input_pkl_path}")
    with open(input_pkl_path, "rb") as f:
        sarimax_res = pickle.load(f)

    # วิธีที่ 1: ตัด Memory Buffer และ Training Data ออกจากตัวโมเดล (Pruning)
    sarimax_res.remove_data()
    pruned_pkl_path = "sarimax_phayao_pruned.pkl"
    with open(pruned_pkl_path, "wb") as f:
        pickle.dump(sarimax_res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"1. บันทึกไฟล์ Pruned Model ขนาดเล็กเรียบร้อย: {pruned_pkl_path}")

    # วิธีที่ 2: สกัดเฉพาะพารามิเตอร์ที่จำเป็นในการคำนวณบันทึกลง JSON
    try:
        params_data = {
            "model_type": "SARIMAX",
            "params": {k: float(v) for k, v in sarimax_res.params.items()},
            "aic": float(sarimax_res.aic) if hasattr(sarimax_res, "aic") else None,
            "bic": float(sarimax_res.bic) if hasattr(sarimax_res, "bic") else None,
            "rmse": float(sarimax_res.mse_model ** 0.5) if hasattr(sarimax_res, "mse_model") else None
        }
        
        json_path = "sarimax_params.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(params_data, f, indent=4, ensure_ascii=False)
        print(f"2. สกัดเฉพาะสัมพฤทธิ์พารามิเตอร์เรียบร้อย: {json_path}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการสกัดข้อมูล JSON: {e}")

if __name__ == "__main__":
    optimize_sarimax_model("sarimax_phayao.pkl")