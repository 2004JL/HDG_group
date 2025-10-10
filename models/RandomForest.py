import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "models"
OUT  = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
test  = pd.read_csv(DATA / "test.csv")

print(f"Dataset sizes: train={len(train)}, test={len(test)}")

required = {"student_id", "program_id", "interests", "field_tags", "label_match"}
for name, df in [("train", train), ("test", test)]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}")
    df["student_id"]  = df["student_id"].astype(str)
    df["program_id"]  = df["program_id"].astype(str)
    df["interests"]   = df["interests"].fillna("")
    df["field_tags"]  = df["field_tags"].fillna("")
    df["label_match"] = pd.to_numeric(df["label_match"], errors="coerce").fillna(0.0)

# Feature engineering
def to_list(s: str):
    return [x.strip().lower() for x in str(s).split(";") if x.strip()]

mlb_int = MultiLabelBinarizer(sparse_output=True)
mlb_tag = MultiLabelBinarizer(sparse_output=True)

X_train = hstack([
    mlb_int.fit_transform(train["interests"].map(to_list)),
    mlb_tag.fit_transform(train["field_tags"].map(to_list))
], format="csr")
y_train = train["label_match"].values

X_test = hstack([
    mlb_int.transform(test["interests"].map(to_list)),
    mlb_tag.transform(test["field_tags"].map(to_list))
], format="csr")
y_test = test["label_match"].values

print("RF feature shapes:", X_train.shape, X_test.shape)

# Train RandomForestRegressor
reg = RandomForestRegressor(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

print("Training RandomForestRegressor ...")
Xtr = X_train.toarray()
Xte = X_test.toarray()
reg.fit(Xtr, y_train)

# Evaluate
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

test_pred = reg.predict(Xte)
test_rmse = rmse(y_test, test_pred)
print(f"[TEST] RMSE = {test_rmse:.4f}")

# Save model
joblib.dump(reg, OUT / "rf_labelmatch_regressor.pkl")
print(f"Model saved: {OUT / 'rf_labelmatch_regressor.pkl'}")

# csv
for split_name, df_split, Xmat, pred in [
    ("test",  test,  Xte, test_pred)
]:
    out_df = df_split.copy()
    out_df["pred_label_match"] = np.round(pred, 4)
    out_df.to_csv(OUT / f"rf_{split_name}.csv", index=False)

print("Predicted results saved rf_test.csv")
