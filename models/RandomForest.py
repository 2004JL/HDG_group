import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "models"

# === 1) 读取数据 ===
train = pd.read_csv(RAW / "train.csv")
val   = pd.read_csv(RAW / "val.csv")
test  = pd.read_csv(RAW / "test.csv")

# 断言关键列存在（防止 KeyError）
needed_cols = {
    "student_id", "program_id", "judge", "label_match",
    "interests", "degree_goal", "english_test_type",
    "english_score_overall", "gpa_std_4"
}
missing = needed_cols - set(train.columns) - set(val.columns) - set(test.columns)
# 不是严格校验三者同时含全部字段，这里仅提示常见缺列
if not needed_cols.issubset(set(train.columns)) \
   or not needed_cols.issubset(set(val.columns)) \
   or not needed_cols.issubset(set(test.columns)):
    raise ValueError(f"Some required columns are missing in splits. "
                     f"Expected at least: {sorted(needed_cols)}")

# === 2) 仅在 judge==True 的样本上训练与评估 ===
train = train[train["judge"] == True].copy()
val   = val[val["judge"] == True].copy()
test  = test[test["judge"] == True].copy()

# 若过滤后为空，提示
for name, df in [("train", train), ("val", val), ("test", test)]:
    if len(df) == 0:
        raise ValueError(f"{name} has 0 rows after filtering judge==True. Check data pipeline.")

# === 3) 规范化/类型修正 ===
# 数值
for c in ["gpa_std_4", "english_score_overall", "label_match"]:
    for df in (train, val, test):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# 类别统一大写去空格
for c in ["degree_goal", "english_test_type"]:
    for df in (train, val, test):
        df[c] = df[c].astype(str).str.strip().str.upper()

# interests: 保持原始分号分隔，如空设为 "" 以便向量化
for df in (train, val, test):
    df["interests"] = df["interests"].fillna("")

# === 4) 目标编码（program_id 做多分类）===
y_le = LabelEncoder()
y_train = y_le.fit_transform(train["program_id"].astype(str))
y_val   = y_le.transform(val["program_id"].astype(str),)
y_test  = y_le.transform(test["program_id"].astype(str),)

# === 5) 特征列定义 ===
num_features = ["gpa_std_4", "english_score_overall", "label_match"]
cat_features = ["degree_goal", "english_test_type"]
txt_feature  = "interests"

# 文本向量化：以分号 ';' 为分隔，忽略低频特征（min_df可按需要调整）
cv = CountVectorizer(
    lowercase=True,
    token_pattern=None,      # 禁用默认正则
    tokenizer=lambda s: [t.strip() for t in s.split(";") if t.strip()],
    min_df=2                 # 过滤极低频标签，防止维度爆炸（可调）
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("txt", cv, txt_feature)
    ],
    remainder="drop"
)

# === 6) 模型（RandomForest）===
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", clf)
])

# === 7) 训练 ===
X_train = train[num_features + cat_features + [txt_feature]]
X_val   = val[num_features + cat_features + [txt_feature]]
X_test  = test[num_features + cat_features + [txt_feature]]

pipe.fit(X_train, y_train)

# === 8) 评估（Accuracy & Top-K）===
def eval_split(X, y, name):
    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    # 保护：二分类也能做 top_k，但若类别数<3 则只做可行的 top-k
    proba = pipe.predict_proba(X)
    k_vals = [1, 3, 5]
    topk = {}
    for k in k_vals:
        k_eff = min(k, proba.shape[1])
        topk[f"top{k_eff}"] = top_k_accuracy_score(y, proba, k=k_eff)
    print(f"[{name}] accuracy={acc:.4f} | " +
          " | ".join([f"{k}={v:.4f}" for k, v in topk.items()]))

print("=== Evaluation ===")
eval_split(X_val, y_val, "VAL")
eval_split(X_test, y_test, "TEST")

# === 9) 按学生生成 Top-K 推荐（示例：Test集）===
# 思路：同一个学生在 test 集会有多行（不同候选 program）。我们用每一行的
# predict_proba 得到“该行 program 对应类别的概率”，据此对候选 program 排序。
def topk_recommendations(df_split, X_split, topk=5):
    proba = pipe.predict_proba(X_split)
    # 把每行“对应自身program_id类别”的概率取出来，作为该候选的评分
    # 先把每行的 program_id -> 类别索引
    class_index = y_le.transform(df_split["program_id"].astype(str))
    row_scores = proba[np.arange(len(df_split)), class_index]

    scored = df_split[["student_id", "program_id"]].copy()
    scored["score"] = row_scores

    # 对每个 student 取 topk
    topk_df = (
        scored.sort_values(["student_id", "score"], ascending=[True, False])
              .groupby("student_id")
              .head(topk)
              .reset_index(drop=True)
    )
    return topk_df

test_top5 = topk_recommendations(test, X_test, topk=5)
test_top5_path = RAW / "test_top5_reco.csv"
test_top5.to_csv(test_top5_path, index=False)
print(f"Saved per-student Top-5 recommendations: {test_top5_path}")
