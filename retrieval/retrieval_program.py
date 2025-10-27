# retrieval.py  —— 可导入类版本
import json
import argparse
import pandas as pd
from pathlib import Path

class Retrieval:

    def __init__(self):

        self.ROOT = Path(__file__).resolve().parents[1]
        self.RAW1 = self.ROOT / "data_clean"
        self.OUT  = self.ROOT / "retrieval"
        self.OUT.mkdir(parents=True, exist_ok=True)

        self.programs_df   = pd.read_csv(self.RAW1 / "programs.csv")
        self.institutions  = pd.read_csv(self.RAW1 / "institutions.csv")
        self.reqs_df       = pd.read_csv(self.RAW1 / "program_requirements.csv")

        self.prog_ins = self.programs_df[["program_id", "name", "field_tags", "institution_id", "degree_level"]].copy() \
            .merge(self.reqs_df[["program_id", "min_gpa_std_4", "english_required_type", "english_min_overall"]].copy(), on="program_id", how="inner") \
            .merge(self.institutions[["institution_id", "locations", "overall_ranking", "website"]].copy(), on="institution_id", how="inner")

    @staticmethod
    def load_student_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        return {
            "student_id": str(obj["student_id"]).strip(),
            "major_intent": str(obj["major_intent"]).strip(),
            "degree_goal": str(obj["degree_goal"]).strip(),
            "english_test_type": str(obj["english_test_type"]).strip(),
            "english_score_overall": float(obj["english_score_overall"]),
            "gpa_std_4": float(obj["gpa_std_4"]),
            "interests": (str(obj.get("interests", "")).strip() if obj.get("interests") is not None else "")
        }

    def eligible_programs(self, student: dict) -> pd.DataFrame:
        student_row = pd.DataFrame([{
            "student_id": student["student_id"],
            "major_intent": student["major_intent"],
            "degree_goal": student["degree_goal"],
            "english_test_type": student["english_test_type"],
            "english_score_overall": student["english_score_overall"],
            "gpa_std_4": student["gpa_std_4"],
            "interests": student["interests"],
        }])

        student_row["_tmp"] = 1
        tmp_prog = self.prog_ins.copy()
        tmp_prog["_tmp"] = 1
        cross = student_row.merge(tmp_prog, on="_tmp").drop(columns="_tmp")

        cross["english_test_type"] = cross["english_test_type"].astype(str).str.upper()
        cross["english_required_type"] = cross["english_required_type"].astype(str).str.upper()
        cross["degree_goal"] = cross["degree_goal"].astype(str).str.lower()
        cross["degree_level"] = cross["degree_level"].astype(str).str.lower()

        eligible = cross[
            (cross["gpa_std_4"] >= cross["min_gpa_std_4"]) &
            (cross["english_test_type"] == cross["english_required_type"]) &
            (cross["english_score_overall"] >= cross["english_min_overall"]) &
            (cross["degree_goal"] == cross["degree_level"])
        ].copy()

        out = eligible[["student_id", "program_id", "interests", "field_tags", "name", "institution_id", "website", "locations", "overall_ranking"]]
        return out

    def run(self, student_json: Path | None = None) -> pd.DataFrame:
        stu = self.load_student_json(Path(student_json))
        df  = self.eligible_programs(stu)
        ROOT = Path(__file__).resolve().parents[1]
        OUT  = ROOT / "retrieval"
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "student_program_retrieval.csv", index=False)
        return df