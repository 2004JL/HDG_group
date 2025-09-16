import pandas as pd

# Continue with step_2 csv file by Junlong, based on student gpa and ielts score choosing the institution where the programs is offered
# This script has three goals

# Part 1, interesrt match based on label matrix (word cosine similarity)
# Part 2, base on australia institutions rank to calculate match value with wight
# Part 3, For students who want to migrate, recommend the institutions with address is in designated regional area category 2 city.

# Part 1 script
eligible = pd.read_csv("eligible_programs.csv")
sim = pd.read_csv("label_matrix.csv", index_col=0)

# Cache row and column labels from the matrix
labels = list(sim.index)

# Unity to the lowercased
lc_to_canonical = {s.lower(): s for s in set(labels)}

# Split a cell by ';' into word, strip whitespace and drop empty pieces
def split_clean(cell):

    if pd.isna(cell):
        return []
    else:
        return [p.strip() for p in str(cell).split(";") if p.strip()]

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

eligible["label_match"] = eligible.apply(row_match, axis=1)

# Part 2 script
programs_df = pd.read_csv("base/programs.csv")
institutions_df = pd.read_csv("base/institutions.csv")

# Merge csv and align rows base on feature column
pti = eligible.merge(programs_df, on="program_id", how="left")
pti = pti.merge(institutions_df, on="institution_id", how="left")

# Add the rank (easy for view)
eligible["institutions_rank"] = pd.to_numeric(pti["Rank"], errors="coerce")

# Load the length (range) of the 'rank'
max = pd.to_numeric(institutions_df["Rank"], errors="coerce").max()

# Weight match score (based on rank)
def calc_c_match(row):
    r = row.get("institutions_rank", -1)
    m = row.get("label_match", float("nan"))
    n = (1 - (float(r)-1) / float(max))
    # The match num is multiply by "n" between 0 and 1
    # The value of "n" is linearly dependent to institutions rank
    return round(float(m) * n, 2)

eligible["wight"] = eligible.apply(calc_c_match, axis=1)

# Part 3 script
ra = pd.read_csv("base/regional_area.csv")

# Get designated regional area category 2 city list
regional_city = set(ra["city"].str.strip().str.lower().unique())

def designated_regional(loc):
    word = str(loc).strip().lower()

    # Check the address is eligible or not
    if word in regional_city:
        return 1
    else:
        return 0

# append to csv
eligible["institution_id"] = pti["institution_id"].values
eligible["designated_regional"] = pti["locations"].apply(designated_regional).astype(int)


students_df = pd.read_csv("base/students.csv")

# Has migration intention is 1 else 0
migr = eligible.merge(students_df[["student_id", "migration_interest"]],on="student_id", how="left")
eligible["student_migration"] = (migr["migration_interest"] != 0).astype(int)

# Output
eligible.to_csv("programs_match.csv", index=False, encoding="utf-8-sig")
print("completed: programs_match.csv")
