#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#based on basic Yonghan's code(RandomForest)
#accuracy
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from xgboost import XGBClassifier
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


#7)
X_train = train[num_features + cat_features + [txt_features]]
X_val = val[num_features + cat_features + [txt_features]]
X_test = test[num_features + cat_features + [txt_features]]

pipe.fit(X_train, y_train)

#ndcg@3


#8)
def eval_split(X, y, name):
    y_pred = pipe.predict(X)
    acc = accuracy_score(y,y_pred)
    proba = pipe.predict_proba(X)
    k_vals = [1, 3, 5]
    
    clf_classes = pipe.named_steps["clf"].classes_ #
    topk = {}
    for k in k_vals:
        k_eff = min(k, proba.shape[1])
        topk[f"top{k_eff}"] = top_k_accuracy_score(y, proba, k=k_eff, labels = pipe.named_steps["clf"].classes_)
        #
    print(f"[{name}] accuracy = {acc:.4f} |" +
           " | ".join([f"{k}={v:.4f}" for k, v in topk.items()]))
    
print("===Evaluation===")
eval_split(X_val, y_val, "VAL")
eval_split(X_test, y_test, "TEST")

#9)
def topk_recommendations(df_split, X_split, topk = 5):
    
    proba = pipe.predict_proba(X_split)
    class_index = y_le.transform(df_split["program_id"].astype(str))
    row_scores = proba[np.arange(len(df_split)), class_index]
    #missing count
    missing_count = (~df_split["program_id"].astype(str).isin(y_le.classes_)).sum()
    print(f"Missing rows (program_id not seen in train): {missing_count}")
    
    scored = df_split[["student_id", "program_id"]].copy()
    scored["score"] = row_scores
    
    topk_df = (
        scored.sort_values(["student_id", "score"], ascending = [True, False])
                          .groupby("student_id")
                          .head(topk)
                          .reset_index(drop=True)
    )
    return topk_df

test_top5 = topk_recommendations(test, X_test, topk = 5)
test_top5_path = "test_top5.csv"
test_top5.to_csv(test_top5_path, index=False)
print(f"Saved per-student Top-5 recommendations: {test_top5_path}") 
    
    


# In[ ]:





# In[ ]:




