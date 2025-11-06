import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "data_clean"
OUT.mkdir(parents=True, exist_ok=True)

# institutions
institutions = pd.read_csv(RAW / "institutions.csv")

institutions["institution_id"] = institutions["institution_id"].astype(str)
institutions["overall_ranking"] = pd.to_numeric(institutions["overall_ranking"], errors="coerce")
institutions["locations"] = institutions["locations"].str.title()
institutions["tuition_fee_low"] = institutions["tuition_fee_low"].astype(float)
institutions["tuition_fee_up"] = institutions["tuition_fee_up"].astype(float)

institutions = institutions.reset_index(drop=True)
institutions.to_csv(OUT / "institutions.csv", index=False)

# programs
programs = pd.read_csv(RAW / "programs.csv")

programs["program_id"] = programs["program_id"].astype(str)
programs["institution_id"] = programs["institution_id"].astype(str)
programs["degree_level"] = programs["degree_level"].astype(str)
programs["is_migration_aligned"] = programs["is_migration_aligned"].astype(bool)

programs = programs.reset_index(drop=True)
programs.to_csv(OUT / "programs.csv", index=False)

# program_requirements
reqs = pd.read_csv(RAW / "program_requirements.csv")

reqs["program_id"] = reqs["program_id"].astype(str)
reqs["min_gpa_std_4"] = reqs["min_gpa_std_4"].astype(float).clip(0, 4)
reqs["english_required_type"] = reqs["english_required_type"].astype(str).str.strip()
reqs["english_min_overall"] = reqs["english_min_overall"].astype(float)

reqs = reqs.reset_index(drop=True)
reqs.to_csv(OUT / "program_requirements.csv", index=False)

# students
students = pd.read_csv(RAW / "students.csv")

students["student_id"] = students["student_id"].astype(str)
students["age"] = students["age"].astype(int).clip(16, 40)
students["budget_aud_per_year"] = students["budget_aud_per_year"].astype(float)
students["migration_interest"] = students["migration_interest"].astype(bool)
students["gpa_std_4"] = (students["gpa_std_4"]).astype(float).clip(0, 4)
students["english_test_type"] = students["english_test_type"].astype(str).str.strip()
students["english_score_overall"] = students["english_score_overall"].astype(float)

students = students.reset_index(drop=True)
students.to_csv(OUT / "students.csv", index=False)

# mentors
mentors = pd.read_csv(RAW / "mentors.csv")

mentors["mentor_id"] = mentors["mentor_id"].astype(str)
mentors["years_experience"] = mentors["years_experience"].astype(int).clip(0, 50)

mentors = mentors.reset_index(drop=True)
mentors.to_csv(OUT / "mentors.csv", index=False)

# scholarship
scholarship = pd.read_csv(RAW / "scholarship.csv")

scholarship["institution_id"] = scholarship["institution_id"].astype(str)
scholarship["scholarship_percent"] = scholarship["scholarship_percent"].astype(float).clip(0, 1)
scholarship["scholarship_GPA_request"] = scholarship["scholarship_GPA_request"].astype(float).clip(0, 4)

scholarship = scholarship.reset_index(drop=True)
scholarship.to_csv(OUT / "scholarship.csv", index=False)

print("cleaned parquet written to", OUT.resolve())