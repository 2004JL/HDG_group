import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW1 = ROOT / "data_clean"
RAW2 = ROOT / "taxonomy"
OUT = ROOT / "features"
OUT.mkdir(parents=True, exist_ok=True)

students = pd.read_csv(RAW1 / "students.csv")
programs = pd.read_csv(RAW1 / "programs.csv")
institutions = pd.read_csv(RAW1 / "institutions.csv")
reqs = pd.read_csv(RAW1 / "program_requirements.csv")
sim = pd.read_csv(RAW2 / "label_matrix_program.csv", index_col=0)

students_slim = students[["student_id", "interests"]].copy()
programs_ins = (programs[["program_id", "field_tags", "institution_id"]]
 .merge(institutions[["institution_id", "overall_ranking"]], on="institution_id", how="left"))

students_slim["_tmp"] = 1
programs_ins["_tmp"] = 1
cross = students_slim.merge(programs_ins, on="_tmp").drop(columns="_tmp")

eligible_out = cross[
    ["student_id", "interests", "program_id", "field_tags", "institution_id", "overall_ranking"]
].copy()

# Cache row and column labels from the matrix
labels = list(sim.index)

# Unity to the lowercased
lc_to_canonical = {s.lower(): s for s in set(labels)}

# Split a cell by ';' into word, strip whitespace and drop empty pieces
def split_clean(cell):

    if pd.isna(cell):
        return []
    else:
        return [p.strip().lower() for p in str(cell).split(";") if p.strip()]

# Compute the average similarity between student interests and programs field_tags for a row.
def row_match(row):
    
    interests = split_clean(row.get("interests", ""))
    tags = split_clean(row.get("field_tags", ""))

    m = len(interests)
    n = len(tags)

    # If either side is empty no comparison can be made
    if m == 0 or n == 0:
        return float(0)

    total = 0.0

    # Assume that interests is a(n) label [a(0)...a(n)], and field_tags is b(m) label [b(0)...b(m)]
    # Sum { set [a(n), b(m)] }
    for it in interests:
        for tg in tags:
            if tg in sim.index and it in sim.columns:
                total += float(sim.at[it, tg])

    # Clamp the result to between 0 and 1
    # The maximum value of set [a(n), b(m)] is 1, where n*m ​​is the count of paths
    denom = m * n
    avg = (total / denom) if denom > 0 else float(0)

    ## Round to 2 decimals number
    if pd.notna(avg):
        avg = round(avg, 2)
    return avg

eligible_out["label_match"] = eligible_out.apply(row_match, axis=1)

eligible_out[["student_id", "interests", "program_id", "field_tags", "label_match"]].to_csv(OUT / "eligible_program.csv", index=False)
print(f"Saved: {OUT / 'eligible_program.csv'}")