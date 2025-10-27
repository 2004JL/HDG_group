import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW1 = ROOT / "data_clean"
RAW2 = ROOT / "taxonomy"
OUT = ROOT / "features"
OUT.mkdir(parents=True, exist_ok=True)

students = pd.read_csv(RAW1 / "students.csv")
sim = pd.read_csv(RAW2 / "label_matrix_core.csv", index_col=0)

core_labels = list(sim.columns)

students_slim = students[["student_id"]].copy()
labels_df = pd.DataFrame({"core_program": core_labels})

students_slim["_tmp"] = 1
labels_df["_tmp"] = 1
cross = students_slim.merge(labels_df, on="_tmp").drop(columns="_tmp")

out = (
    cross[["student_id", "core_program"]]
    .merge(students[["student_id", "interests"]],on="student_id", how="left")
)

labels = list(sim.columns)

lc_to_canonical = {s.lower(): s for s in set(labels)}

def split_clean(cell):

    return [p.strip().lower() for p in str(cell).split(";") if p.strip()]

def row_match(row):
    
    interests = split_clean(row.get("interests", ""))
    core = split_clean(row.get("core_program", ""))

    m = len(interests)
    n = len(core)

    if m == 0 or n == 0:
        return float(0)

    total = 0.0

    for it in interests:
        for tg in core:
            if tg in sim.columns and it in sim.index:
                total += float(sim.at[it, tg])

    denom = m * n
    avg = (total / denom) if denom > 0 else float(0)

    if pd.notna(avg):
        avg = round(avg, 2)
    return avg

out["program_match"] = out.apply(row_match, axis=1)

out [["student_id", "interests", "core_program", "program_match"]].to_csv(OUT / "eligible_core.csv", index=False)