import os
import pandas as pd

SRC = "data/clean_loan_v2.csv"   # ถ้าไฟล์ใหญ่อยู่ที่อื่น ปรับ path ให้ตรง
DST = "data/clean_loan_small.csv"
N   = 10000                      # ปรับจำนวนแถวได้ตามใจ

os.makedirs("data", exist_ok=True)
if not os.path.exists(SRC):
    raise FileNotFoundError(f"ไม่พบไฟล์ต้นฉบับ: {SRC}")

df = pd.read_csv(SRC)
small = df.sample(n=min(N, len(df)), random_state=42)
small.to_csv(DST, index=False)
print(f"✅ Wrote {DST} with shape {small.shape}")
