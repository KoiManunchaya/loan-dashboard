
import os
import numpy as np
import pandas as pd
from math import ceil

INPUT_CSV  = "data/clean_loan_small.csv"   # อินพุต (ไฟล์เล็ก)
OUTPUT_CSV = "data/clean_loan_small.csv"   # เอาท์พุต (บันทึกทับไฟล์เล็กเดิม)

# --------- Load ---------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"ไม่พบไฟล์: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print("📂 Loaded shape:", df.shape)
df.columns = df.columns.str.strip().str.lower()

# --------- Drop columns ไม่จำเป็น ---------
drop_cols = ["emp_title", "emp_length", "address"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

# --------- Clean/Convert ---------
# term: "36 months" -> 36.0
if "term" in df.columns:
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)

# int_rate, revol_util: "13.56%" -> 13.56
for col in ["int_rate", "revol_util"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")

# --------- Drop rows ที่ NA เยอะเกินไป ---------
required_non_na = ceil(0.9 * df.shape[1])  # ต้องมีข้อมูลอย่างน้อย 90% ของคอลัมน์
df.dropna(thresh=required_non_na, inplace=True)

# --------- Fill NA ที่เหลือ (เฉพาะตัวเลข) ---------
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# --------- Drop grade (ซ้ำกับ sub_grade) ---------
if "grade" in df.columns:
    df.drop("grade", axis=1, inplace=True)

# --------- One-hot encode ---------
cat_candidates = ["sub_grade", "home_ownership", "initial_list_status", "application_type"]
cat_cols = [c for c in cat_candidates if c in df.columns]
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# --------- Save ---------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print("✅ Cleaned shape:", df.shape)
print(f"💾 Saved to: {OUTPUT_CSV}")
