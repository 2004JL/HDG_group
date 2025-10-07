#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#metrics : ndcg@3 + diversity
#focuses on ranking quality(ndcg@3)


# In[ ]:


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

#import xgboost as xgb

#ROOT = Path(__file__).resolve().parents[1]
#RAW = ROOT / "models"

#1)load
train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
test = pd.read_csv("test.csv")

#(KeyError)
needed_cols = {
    "student_id", "program_id", "judge", "label_match",
    "interests", "degree_goal", "english_test_type",
    "english_score_overall", "gpa_std_4"
}
missing = needed_cols - set(train.columns) - set(val.columns) - set(test.columns)

if not needed_cols.issubset(set(train.columns)) \
   or not needed_cols.issubset(set(val.columns)) \
   or not needed_cols.issubset(set(test.columns)):
    raise ValueError(f"Some required columns are missing in splits."
                     f"Expected at least: {sorted(needed_cols)}")

#2)filter
train = train[train["judge"] == True].copy()
val = val[val["judge"] == True].copy()
test = test[test["judge"] == True].copy()

for name, df in [("train", train), ("val", val), ("test",test)]:
    if len(df) == 0:
        raise ValueError(f"{name} has 0 rows after filtering judge==True. Check data pipeline.")
        
#3)cleanup
for c in ["gpa_std_4", "english_score_overall", "label_match"]:
    for df in (train, val, test):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

for c in ["degree_goal", "english_test_type"]:
    for df in (train, val, test):
        df[c] = df[c].astype(str).str.strip().str.upper()

#interests
for df in (train, val, test):
    df["interests"] = df["interests"].fillna("")

#4)labels
y_le = LabelEncoder()
y_train = y_le.fit_transform(train["program_id"].astype(str))
y_val = y_le.transform(val["program_id"].astype(str))
y_test = y_le.transform(test["program_id"].astype(str))

#5)features
num_features = ["gpa_std_4", "english_score_overall", "label_match"]
cat_features = ["degree_goal", "english_test_type"]
txt_features = "interests"

cv = CountVectorizer(
    lowercase = True,
    token_pattern = None,
    tokenizer = lambda s: [t.strip() for t in s.split(";") if t.strip()],
    min_df = 2,
    max_features = 5000
)

preprocessor = ColumnTransformer(
    transformers = [
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_features),
        ("txt", cv, txt_features)
    ],
    remainder = "drop"
)

#6) add xgbclassifier
clf = XGBClassifier(
    objective = "multi:softprob", #
    n_estimators = 300, #
    max_depth = 6, #
    n_jobs = -1,
    colsample_bytree=0.9, #
    tree_method="hist", #
    eval_metric="mlogloss" #
)

pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", clf)
])

X_train = train[num_features + cat_features + [txt_features]]
X_val = val[num_features + cat_features + [txt_features]]
X_test = test[num_features + cat_features + [txt_features]]

pipe.fit(X_train, y_train)
#=======================ndcg@3===========================
def dcg_k(r,k):
    r = np.asarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2,r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_k(r,k):
    idcg = dcg_k(sorted(r, reverse=True),k)
    if not idcg:
        return 0.
    return dcg_k(r,k) / idcg

def ndcg_at3_for_split(df_split, X_split):
    proba = pipe.predict_proba(X_split)
    cls_idx = y_le.transform(df_split["program_id"].astype(str))
    row_scores = proba[np.arange(len(df_split)), cls_idx]

    ndcgs = []
    for sid, g in df_split.assign(_score=row_scores).groupby("student_id"):
        g = g.sort_values("_score", ascending=False)
        r = g["label_match"].to_numpy(dtype=float)   # accurate(0/1)
        ndcgs.append(ndcg_k(r, 3))
    return float(np.mean(ndcgs)) if ndcgs else 0.0

print("nDCG@3 (VAL):", ndcg_at3_for_split(val, X_val))
print("nDCG@3 (TEST):", ndcg_at3_for_split(test, X_test))


mask_val  = val["program_id"].astype(str).isin(y_le.classes_)
val, X_val = val[mask_val].copy(), X_val[mask_val].copy()
mask_test  = test["program_id"].astype(str).isin(y_le.classes_)
test, X_test = test[mask_test].copy(), X_test[mask_test].copy()




