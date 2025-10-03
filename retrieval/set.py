import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "retrieval"
OUT = ROOT / "splits"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW / "eligible.csv")

train_val, test = train_test_split(df, test_size=0.2, random_state=42)

train, val = train_test_split(train_val, test_size=0.25, random_state=42)

train.to_csv(OUT / "train.csv", index=False)
val.to_csv(OUT / "val.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)

print(f"数据集已拆分完成:")
print(f"训练集: {len(train)} 行")
print(f"验证集: {len(val)} 行")
print(f"测试集: {len(test)} 行")