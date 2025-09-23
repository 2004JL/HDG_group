import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "data_clean"
OUT.mkdir(parents=True, exist_ok=True)

def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def bool_col(s: pd.Series) -> pd.Series:
    m = {"true": True, "1": True, "yes": True, "y": True, "t": True,
         "false": False, "0": False, "no": False, "n": False, "f": False}
    return (s.astype(str).str.strip().str.lower().map(m)).astype("boolean")

def english_test_col(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    synonyms = {"IELTS ACADEMIC": "IELTS", "TOEFL IBT": "TOEFL", "PTE ACADEMIC": "PTE"}
    return s.replace(synonyms)

def strip_obj(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

# institutions
institutions = pd.read_csv(RAW / "institutions.csv")
institutions = strip_obj(institutions)
if "institution_id" in institutions.columns:
    institutions["institution_id"] = to_int(institutions["institution_id"])

need_nonnull = [c for c in ["institution_id", "name"] if c in institutions.columns]
if need_nonnull:
    institutions = institutions.dropna(subset=need_nonnull)

if "institution_id" in institutions.columns:
    institutions = institutions.drop_duplicates(subset=["institution_id"])
institutions = institutions.reset_index(drop=True)
institutions.to_csv(OUT / "institutions.csv", index=False)

# programs
programs = pd.read_csv(RAW / "programs.csv")
programs = strip_obj(programs)

for col in ["program_id", "institution_id"]:
    if col in programs.columns:
        programs[col] = to_int(programs[col])
if "tuition_aud_per_year" in programs.columns:
    programs["tuition_aud_per_year"] = to_float(programs["tuition_aud_per_year"])
if "is_migration_aligned" in programs.columns:
    programs["is_migration_aligned"] = bool_col(programs["is_migration_aligned"])

if {"institution_id"}.issubset(programs.columns) and "institution_id" in institutions.columns:
    programs = programs[programs["institution_id"].isin(institutions["institution_id"])]

need_nonnull = [c for c in ["program_id", "program_name", "degree_level"] if c in programs.columns]
if need_nonnull:
    programs = programs.dropna(subset=need_nonnull)

if "program_id" in programs.columns:
    programs = programs.drop_duplicates(subset=["program_id"])
programs = programs.reset_index(drop=True)
programs.to_csv(OUT / "programs.csv", index=False)

# program_requirements
reqs = pd.read_csv(RAW / "program_requirements.csv")
reqs = strip_obj(reqs)

if "program_id" in reqs.columns:
    reqs["program_id"] = to_int(reqs["program_id"])
if "min_gpa_std_4" in reqs.columns:
    reqs["min_gpa_std_4"] = to_float(reqs["min_gpa_std_4"]).clip(0, 4)
if "english_required_type" in reqs.columns:
    reqs["english_required_type"] = english_test_col(reqs["english_required_type"])
if "english_min_overall" in reqs.columns:
    reqs["english_min_overall"] = to_float(reqs["english_min_overall"])
    reqs = reqs[reqs["english_min_overall"].notna() & (reqs["english_min_overall"] >= 0)]

if "program_id" in reqs.columns and "program_id" in programs.columns:
    reqs = reqs[reqs["program_id"].isin(programs["program_id"])]

reqs = reqs.dropna(subset=["program_id"])
reqs = reqs.drop_duplicates(subset=["program_id"]).reset_index(drop=True)
reqs.to_csv(OUT / "program_requirements.csv", index=False)

# students
students = pd.read_csv(RAW / "students.csv")
students = strip_obj(students)

if "student_id" in students.columns:
    students["student_id"] = to_int(students["student_id"])
if "age" in students.columns:
    students["age"] = to_int(students["age"])
    students = students[(students["age"].isna()) | ((students["age"] >= 14) & (students["age"] <= 80))]
if "budget_aud_per_year" in students.columns:
    students["budget_aud_per_year"] = to_float(students["budget_aud_per_year"])
if "migration_interest" in students.columns:
    students["migration_interest"] =  bool_col(students["migration_interest"])
if "english_test_type" in students.columns:
    students["english_test_type"] = english_test_col(students["english_test_type"])
if "english_score_overall" in students.columns:
    students["english_score_overall"] = to_float(students["english_score_overall"])
if "gpa_std_4" in students.columns:
    students["gpa_std_4"] = to_float(students["gpa_std_4"]).clip(0, 4)

students = students.dropna(subset=["student_id"])
students = students.drop_duplicates(subset=["student_id"]).reset_index(drop=True)
students.to_csv(OUT / "students.csv", index=False)

# mentors
mentors = pd.read_csv(RAW / "mentors.csv")
mentors = strip_obj(mentors)

if "mentor_id" in mentors.columns:
    mentors["mentor_id"] = to_int(mentors["mentor_id"])
if "years_experience" in mentors.columns:
    mentors["years_experience"] = to_int(mentors["years_experience"]).fillna(0).clip(0, 80)

mentors = mentors.dropna(subset=["mentor_id"])
mentors = mentors.drop_duplicates(subset=["mentor_id"]).reset_index(drop=True)
mentors.to_csv(OUT / "mentors.csv", index=False)

print("✓ cleaned parquet written to", OUT.resolve())