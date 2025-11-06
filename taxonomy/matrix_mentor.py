import gensim.downloader as api
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

model = api.load("glove-wiki-gigaword-300")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_clean"
OUT = ROOT / "taxonomy"

students_df = pd.read_csv(RAW / "programs.csv")
mentors_df = pd.read_csv(RAW / "mentors.csv")

field_tags = students_df["field_tags"].dropna()
expertise_tags = mentors_df["expertise_tags"].dropna()

all_field_tags = field_tags.str.split(";").explode().str.strip().str.lower()
all_expertise_tags = expertise_tags.str.split(";").explode().str.strip().str.lower()

word_set = sorted(set(all_field_tags.unique()) | set(all_expertise_tags.unique()))

def get_vector(phrase):

    words = phrase.split()
    vecs = []

    for w in words:
        if w in model.key_to_index:
            vecs.append(model[w])
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)

vectors = np.array([get_vector(i) for i in word_set])

similarity_matrix = cosine_similarity(vectors)

similarity_matrix = np.round(np.clip(similarity_matrix, 0, 1), 2)

df_sim = pd.DataFrame(similarity_matrix, index=word_set, columns=word_set)
df_sim.to_csv(OUT / "label_matrix_mentor.csv", encoding="utf-8-sig")