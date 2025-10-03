import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW1 = ROOT / "data_clean"
RAW2 = ROOT / "taxonomy"
OUT = ROOT / "retrieval"
OUT.mkdir(parents=True, exist_ok=True)

students = pd.read_csv(RAW1 / "students.csv")
programs = pd.read_csv(RAW1 / "programs.csv")
institutions = pd.read_csv(RAW1 / "institutions.csv")
reqs = pd.read_csv(RAW1 / "program_requirements.csv")
sim = pd.read_csv(RAW2 / "label_matrix.csv", index_col=0)

students_slim = students[["student_id", "major_intent", "degree_goal", "english_test_type", "english_score_overall", "gpa_std_4"]].copy()
programs_slim = programs[["program_id", "name", "institution_id", "degree_level"]].copy()
institutions_slim = institutions[["institution_id", "locations", "overall_ranking"]].copy()
reqs_slim = reqs[["program_id", "min_gpa_std_4", "english_required_type", "english_min_overall"]].copy()

prog_full = programs_slim.merge(reqs_slim, on="program_id", how="inner")
prog_ins = prog_full.merge(institutions_slim, on="institution_id", how="inner")

students_slim["_tmp"] = 1
prog_ins["_tmp"] = 1
cross = students_slim.merge(prog_ins, on="_tmp").drop(columns="_tmp")

eligible = cross[
    (cross["gpa_std_4"] * 1.15 >= cross["min_gpa_std_4"])&
    (cross["english_test_type"] == cross["english_required_type"])&
    (cross["english_score_overall"] * 1.15 >= cross["english_min_overall"])&
    (cross["degree_goal"] == cross["degree_level"])
]

eligible_out = (
    eligible[["student_id", "program_id"]]
    .drop_duplicates()
    .sort_values(["student_id", "program_id"])
    .reset_index(drop=True)

    .merge(students[["student_id", "degree_goal"]],
    on="student_id", how="left")
    .merge(programs[["program_id", "degree_level"]],
    on="program_id", how="left")

    .merge(students[["student_id", "interests"]],
    on="student_id", how="left")
    .merge(programs[["program_id", "field_tags"]],
    on="program_id", how="left")

    .merge(programs[["program_id", "institution_id"]],
    on="program_id", how="left")
    .merge(institutions[["institution_id", "overall_ranking"]],
    on="institution_id", how="left")
)

eligible_out = eligible_out[["student_id", "interests", "degree_goal", "program_id", "field_tags", "degree_level", "institution_id", "overall_ranking"]]

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
    tags      = split_clean(row.get("field_tags", ""))

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

# Load the length (range) of the 'rank'
max = pd.to_numeric(institutions["overall_ranking"], errors="coerce").max()

# Weight match score (based on rank)
def calc_c_match(row):
    r = row.get("overall_ranking", -1)
    m = row.get("label_match", float("nan"))
    n = (1 - (float(r)-1) / float(max))
    # The match num is multiply by "n" between 0 and 1
    # The value of "n" is linearly dependent to institutions rank
    return round(float(m) * n, 2)

eligible_out["wight"] = eligible_out.apply(calc_c_match, axis=1)

eligible_out.to_csv(OUT / "eligible.csv", index=False)
print(f"Saved: {OUT / 'eligible.csv'}")