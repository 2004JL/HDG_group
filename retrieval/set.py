import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "retrieval"
OUT = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW / "eligible.csv")
unique_students = df["student_id"].unique()

np.random.seed(42)
test_students = np.random.choice(unique_students, size=10, replace=False)

test = df[df["student_id"].isin(test_students)].copy()
test = test.sort_values(by=["student_id", "label_match"], ascending=[True, False])

train = df[~df["student_id"].isin(test_students)].copy()

train.to_csv(OUT / "train.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)

print(f"train set {len(train)} row")
print(f"test set {len(test)} row")