# retrieval.py  —— 可导入类版本
import json
import argparse
import pandas as pd
from pathlib import Path

class RetrievalCore:

    def __init__(self):

        self.ROOT = Path(__file__).resolve().parents[1]
        self.RAW1 = self.ROOT / "data_clean"
        self.RAW2 = self.ROOT / "taxonomy"
        self.OUT  = self.ROOT / "retrieval"
        self.OUT.mkdir(parents=True, exist_ok=True)

        self.programs_df = pd.read_csv(self.RAW1 / "programs.csv")
        self.institutions = pd.read_csv(self.RAW1 / "institutions.csv")
        self.reqs_df = pd.read_csv(self.RAW1 / "program_requirements.csv")
        self.sim = pd.read_csv(self.RAW2 / "label_matrix_core.csv", index_col=0)


        self.prog_ins = self.programs_df[["program_id", "program_name", "field_tags", "institution_id", "degree_level"]].copy() \
            .merge(self.reqs_df[["program_id", "min_gpa_std_4", "english_required_type", "english_min_overall"]].copy(), on="program_id", how="inner") \
            .merge(self.institutions[["institution_id", "institution_name", "locations", "overall_ranking", "website"]].copy(), on="institution_id", how="inner")

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

    def core_programs(self, student: dict) -> pd.DataFrame:
        student_row = pd.DataFrame([{
            "student_id": student["student_id"],
            "major_intent": student["major_intent"],
            "degree_goal": student["degree_goal"],
            "english_test_type": student["english_test_type"],
            "english_score_overall": student["english_score_overall"],
            "gpa_std_4": student["gpa_std_4"],
            "interests": student["interests"],
        }])

        core_labels = list(self.sim.columns)
        labels_df = pd.DataFrame({"core_program": core_labels})
        
        labels_df["_tmp"] = 1
        student_row["_tmp"] = 1
        cross = student_row.merge(labels_df, on="_tmp").drop(columns="_tmp")

        out = cross[["student_id", "interests", "core_program"]]
        return out

    def eligible_programs(self, student: dict, top_n: int = 3) -> pd.DataFrame:
        student_row = pd.DataFrame([{
            "student_id": student["student_id"],
            "major_intent": student["major_intent"],
            "degree_goal": student["degree_goal"],
            "english_test_type": student["english_test_type"],
            "english_score_overall": student["english_score_overall"],
            "gpa_std_4": student["gpa_std_4"],
            "interests": student["interests"],
        }])

        core = pd.read_csv(self.OUT / "core_program.csv")

        top_name = core["core_program"].astype(str).str.strip().replace("", pd.NA).dropna().head(top_n).str.lower().tolist()
        prog_name = self.prog_ins["program_name"].astype(str).str.strip().str.lower()
        mask = prog_name.isin(top_name)
        match = self.prog_ins.loc[mask].copy()
        match["_tmp"] = 1
        student_row["_tmp"] = 1
        cross = student_row.merge(match, on="_tmp").drop(columns="_tmp")

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

        out = eligible[["student_id", "interests", "program_id", "program_name", "field_tags", "institution_id", "institution_name", "website", "locations", "overall_ranking"]]
        return out

    def find(self, student_json: Path | None = None) -> pd.DataFrame:
        stu = self.load_student_json(Path(student_json))
        df  = self.core_programs(stu)
        return df
    
    def run(self, student_json: Path | None = None, top_n: int = 3) -> pd.DataFrame:
        stu = self.load_student_json(Path(student_json))
        df  = self.eligible_programs(stu, top_n)
        ROOT = Path(__file__).resolve().parents[1]
        OUT  = ROOT / "retrieval"
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "student_program_retrieval.csv", index=False)
        return df