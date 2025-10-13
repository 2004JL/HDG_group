import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
from sklearn.metrics import mean_squared_error
import xgboost as xgb
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

X_test = hstack([
    mlb_int.transform(test["interests"].map(to_list)),
    mlb_tag.transform(test["field_tags"].map(to_list))
], format="csr")

y_train = train["label_match"].values
y_test  = test["label_match"].values

print("XGBoost feature shapes:", X_train.shape, X_test.shape)

# Train XGBoost Regressor
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 10,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": -1,
    "seed": 42
}

print("Training XGBoostRegressor ...")
evals = [(dtrain, "train"), (dtest, "test")]
model = xgb.train(params, dtrain, num_boost_round=300, evals=evals, verbose_eval=50)

# Evaluate
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

test_pred = model.predict(dtest)
test_rmse = rmse(y_test, test_pred)
print(f"[TEST] RMSE = {test_rmse:.4f}")

# outputs
joblib.dump(model, OUT / "xgb_labelmatch_regressor.pkl")
print(f"Model saved: {OUT / 'xgb_labelmatch_regressor.pkl'}")

out_df = test.copy()
out_df["pred_label_match"] = np.round(test_pred, 4)
out_df.to_csv(OUT / "xgb_test.csv", index=False)
print("Predicted results saved (xgb_test.csv).")
