import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

# === 配置 ===
K_PER_PROGRAM = 100      # 每个 program_id 最多抽取多少条
TRAIN_RATIO   = 0.6     # 训练占比（组内，除去“种子”之外在剩余中按比例）
VAL_RATIO     = 0.2     # 验证占比（组内），剩余给测试
RANDOM_STATE  = 42

# === 路径设置 ===
ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "retrieval"
OUT  = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)

# === 读取全量候选对（巨表） ===
df = pd.read_csv(RAW / "eligible.csv")

# 去重（如果整表存在完全重复行）
df = df.drop_duplicates().reset_index(drop=True)

# 为可复现，先全表洗牌
df = shuffle(df, random_state=RANDOM_STATE)

train_parts, val_parts, test_parts = [], [], []

for pid, g in df.groupby("program_id", sort=False):
    g = g.copy()

    # --- 1) 训练种子：优先抽取 judge==True 的 1 条；若没有则从全组取 1 条 ---
    g_true = g[g.get("judge", False) == True]
    if len(g_true) > 0:
        seed_train = g_true.sample(n=1, random_state=RANDOM_STATE)
    else:
        seed_train = g.sample(n=1, random_state=RANDOM_STATE)

    # 从组内剔除种子样本
    remaining = g.drop(seed_train.index)

    # --- 2) 从剩余中再取最多 (K_PER_PROGRAM - 1) 条（若不足则全取） ---
    n_rem_take = max(0, min(K_PER_PROGRAM - 1, len(remaining)))
    rem_take = remaining.head(n_rem_take)  # 因为上面已经洗牌，这里 head 即随机

    # --- 3) 组内切分（对 remaining 部分按比例分配；seed 固定进 train）---
    n_rem = len(rem_take)
    n_train_add = int(round(n_rem * TRAIN_RATIO))
    n_val       = int(round(n_rem * VAL_RATIO))
    n_test      = n_rem - n_train_add - n_val

    # 边界校正
    if n_test < 0:
        n_test = 0
        n_val = n_rem - n_train_add
        if n_val < 0:
            n_val = 0
            n_train_add = n_rem

    g_train = pd.concat([seed_train, rem_take.iloc[:n_train_add]], axis=0)
    g_val   = rem_take.iloc[n_train_add:n_train_add + n_val]
    g_test  = rem_take.iloc[n_train_add + n_val:]

    train_parts.append(g_train)
    val_parts.append(g_val)
    test_parts.append(g_test)

# --- 4) 拼接集合 ---
train = pd.concat(train_parts, axis=0, ignore_index=True)
val   = pd.concat(val_parts,   axis=0, ignore_index=True)
test  = pd.concat(test_parts,  axis=0, ignore_index=True)

# --- 5) 三者无交集（整行判等） ---
def no_intersection(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    inter = pd.merge(a, b, how="inner")
    return len(inter) == 0

assert no_intersection(train, val),  "train 与 val 有交集"
assert no_intersection(train, test), "train 与 test 有交集"
assert no_intersection(val, test),   "val 与 test 有交集"

# --- 6) 训练集覆盖所有 program_id ---
all_programs   = set(df["program_id"].astype(str).unique())
train_programs = set(train["program_id"].astype(str).unique())
missing = sorted(all_programs - train_programs)
if missing:
    raise RuntimeError(f"训练集缺少以下 program_id：{missing[:20]} ... 共 {len(missing)} 个")
else:
    print(f"训练集已覆盖所有 program_id，总类数：{len(train_programs)}")

# （可选）进一步校验：若某个 program 在整体中存在 judge==True，
# 则训练集中也应至少有一条 judge==True（防止后续 RandomForest 过滤后丢类）
has_true_overall = set(df.loc[df.get("judge", False) == True, "program_id"].astype(str).unique())
has_true_in_train = set(train.loc[train.get("judge", False) == True, "program_id"].astype(str).unique())
leak_risk = sorted(has_true_overall - has_true_in_train)
if leak_risk:
    # 理论上不应出现；若出现说明该 program 的 seed 被错误分配了（上面的逻辑已避免）
    print(f"风险提示：以下 program_id 在总体有 judge==True，但训练集中未包含其 judge==True：{leak_risk[:20]} ... 共 {len(leak_risk)} 个")

# --- 7) 保存 ---
train.to_csv(OUT / "train.csv", index=False)
val.to_csv(OUT / "val.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)

print("\n数据集已拆分完成（按 program 分组抽样 + 种子保证 judge==True + 组内 60/20/20 + 无交集）:")
print(f"训练集: {len(train):,} 行")
print(f"验证集: {len(val):,} 行")
print(f"测试集: {len(test):,} 行")