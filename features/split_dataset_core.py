import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "features"
OUT = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW / "eligible_core.csv")
df_sample = df.sample(frac=1, random_state=42).reset_index(drop=True)
unique_students = df_sample["student_id"].unique()

np.random.seed(42)
test_students = np.random.choice(unique_students, size=10, replace=False)

test = df_sample[df_sample["student_id"].isin(test_students)].copy()
test = test.sort_values(by=["student_id", "program_match"], ascending=[True, False])

train = df_sample[~df_sample["student_id"].isin(test_students)].copy()

train.to_csv(OUT / "train_core.csv", index=False)
test.to_csv(OUT / "test_core.csv", index=False)

print(f"train set {len(train)} row")
print(f"test set {len(test)} row")