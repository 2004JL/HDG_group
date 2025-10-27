# retrieval/retrieval_mentor.py
from pathlib import Path
import pandas as pd

class RetrievalMentor:

    def __init__(self):
        self.ROOT = Path(__file__).resolve().parents[1]
        self.RAW  = self.ROOT / "data_clean"
        self.OUT  = self.ROOT / "retrieval"
        self.OUT.mkdir(parents=True, exist_ok=True)

        self.mentors  = pd.read_csv(self.RAW / "mentors.csv")
        self.programs = pd.read_csv(self.RAW / "programs.csv")

    def eligible_mentors(self, top_n: int = 3) -> pd.DataFrame:

        sp = pd.read_csv(self.OUT / "student_program.csv")

        order_cols = [c for c in ["pred_label_match", "score", "label_match"] if c in sp.columns]
        if order_cols:
            sp_top = sp.sort_values(order_cols, ascending=[False]*len(order_cols)).head(top_n)
        else:
            sp_top = sp.head(top_n)

        sp_top = sp_top[["program_id", "field_tags"]].copy()
        sp_top["_tmp"] = 1

        mentors = self.mentors.copy()
        mentors["_tmp"] = 1

        cross = sp_top.merge(mentors, on="_tmp").drop(columns="_tmp")

        base_cols = ["program_id", "field_tags", "mentor_id", "expertise_tags"]
        extra_cols = [c for c in ["languages", "education_background", "years_experience"] if c in cross.columns]
        out = cross[base_cols + extra_cols].copy()
        return out

    def run(self, top_n: int = 3) -> pd.DataFrame:

        df = self.eligible_mentors(top_n=top_n)
        df.to_csv(self.OUT / "program_mentor_retrieval.csv", index=False)
        return df