import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "retrieval"
OUT = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW / "eligible.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv(OUT / "train.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)

print(f"8:2")
print(f"train_set {len(train)}")
print(f"test_set {len(test)}")