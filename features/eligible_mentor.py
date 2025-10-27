import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW1 = ROOT / "data_clean"
RAW2 = ROOT / "taxonomy"
OUT = ROOT / "features"
OUT.mkdir(parents=True, exist_ok=True)

programs = pd.read_csv(RAW1 / "programs.csv")
mentors = pd.read_csv(RAW1 / "mentors.csv")
sim = pd.read_csv(RAW2 / "label_matrix_mentor.csv", index_col=0)

programs_slim = programs[["program_id", "field_tags"]].copy()
mentors_slim = mentors[["mentor_id", "languages", "expertise_tags","education_background","years_experience"]].copy()

programs_slim ["_tmp"] = 1
mentors_slim["_tmp"] = 1
cross = programs_slim .merge(mentors_slim, on="_tmp").drop(columns="_tmp")

eligible_out = cross[
    ["program_id", "field_tags", "mentor_id", "languages", "expertise_tags", "education_background", "years_experience"]
].copy()

labels = list(sim.index)

lc_to_canonical = {s.lower(): s for s in set(labels)}

def split_clean(cell):

    if pd.isna(cell):
        return []
    else:
        return [p.strip().lower() for p in str(cell).split(";") if p.strip()]

def row_match(row):
    
    interests = split_clean(row.get("field_tags", ""))
    tags = split_clean(row.get("expertise_tags", ""))

    m = len(interests)
    n = len(tags)

    if m == 0 or n == 0:
        return float(0)

    total = 0.0

    for it in interests:
        for tg in tags:
            if tg in sim.index and it in sim.columns:
                total += float(sim.at[it, tg])

    denom = m * n
    avg = (total / denom) if denom > 0 else float(0)

    if pd.notna(avg):
        avg = round(avg, 2)
    return avg

eligible_out["label_match"] = eligible_out.apply(row_match, axis=1)

eligible_out[["program_id", "field_tags", "mentor_id", "languages", "expertise_tags", "education_background", "years_experience", "label_match"]].to_csv(OUT / "eligible_mentor.csv", index=False)
print(f"Saved: {OUT / 'eligible_mentor.csv'}")