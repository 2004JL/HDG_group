import gensim.downloader as api
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Build a label similarity matrix (students interests, programs field_tags)
# Pretrained 300-d GloVe word vectors
model = api.load("glove-wiki-gigaword-300")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_clean"
OUT = ROOT / "taxonomy"

# Load source data
students_df = pd.read_csv(RAW / "students.csv")
programs_df = pd.read_csv(RAW / "programs.csv")

# Collect label text
interests = students_df["interests"].dropna()
field_tags = programs_df["field_tags"].dropna()

# Split on ';' trim spaces and flatten into one label per row
all_interests = interests.str.split(";").explode().str.strip().str.lower()
all_tags = field_tags.str.split(";").explode().str.strip().str.lower()

# Unique vocabulary across both sides (case sensitive canonical set)
word_set = sorted(set(all_interests.unique()) | set(all_tags.unique()))

def get_vector(phrase):

    words = phrase.split()
    vecs = []

    # Average the GloVe vectors of tokens in a phrase
    # If none of the tokens exist in the model return a zero vector

    for w in words:
        if w in model.key_to_index:
            vecs.append(model[w])
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)

# Embed every label into a vector
vectors = np.array([get_vector(i) for i in word_set])

# Cosine similarity between every pair of labels
similarity_matrix = cosine_similarity(vectors)

# Keep values within [0, 1] and round to 2 decimals
similarity_matrix = np.round(np.clip(similarity_matrix, 0, 1), 2)

# Setting and output
df_sim = pd.DataFrame(similarity_matrix, index=word_set, columns=word_set)
df_sim.to_csv(OUT / "label_matrix.csv", encoding="utf-8-sig")