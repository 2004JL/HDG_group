import gensim.downloader as api
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

model = api.load("glove-wiki-gigaword-300")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_clean"
OUT = ROOT / "taxonomy"
OUT.mkdir(parents=True, exist_ok=True)

students_df = pd.read_csv(RAW / "students.csv")
programs_df = pd.read_csv(RAW / "programs.csv")

interests = students_df["interests"].dropna()
field_tags = programs_df["field_tags"].dropna()

all_interests = interests.str.split(";").explode().str.strip().str.lower()
all_tags = field_tags.str.split(";").explode().str.strip().str.lower()

word_set = sorted(set(all_interests.unique()) | set(all_tags.unique()))

target_labels = [
    "Nursing", "Education", "Engineering", "Information Technology",
    "Cyber Security", "Construction", "Mining", "Trades",
    "Agriculture", "Logistics", "Health Sciences"
]
target_labels_lower = [t.lower() for t in target_labels]

def get_vector(text: str) -> np.ndarray:
    tokens = text.split()
    vecs = [model[w] for w in tokens if w in model.key_to_index]
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)

word_vecs = np.array([get_vector(w) for w in word_set])
target_vecs = np.array([get_vector(t) for t in target_labels_lower])


sim_matrix = cosine_similarity(word_vecs, target_vecs)
sim_matrix = np.round(np.clip(sim_matrix, 0, 1), 3)

df_sim = pd.DataFrame(sim_matrix, index=word_set, columns=target_labels_lower)
out_path = OUT / "label_matrix_core.csv"
df_sim.to_csv(out_path, encoding="utf-8-sig")