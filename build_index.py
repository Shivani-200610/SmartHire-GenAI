import os
import pickle

import pandas as pd

from src.embeddings import model
from src.faiss_db import FAISSDatabase


print("Loading dataset...")

df = pd.read_excel("data/jobs/jobs_data.xlsx")

# ---------------------------------------------------
# TEST MODE
# Comment this line later to index the full dataset
# ---------------------------------------------------

df = df.head(1000)

# ---------------------------------------------------

df = df.fillna("")

print("Preparing searchable text...")

df["search_text"] = (
    "Title: " + df["title"].astype(str)
    + "\nCompany: " + df["companyName"].astype(str)
    + "\nSkills: " + df["tagsAndSkills"].astype(str)
    + "\nExperience: " + df["experience"].astype(str)
    + "\nLocation: " + df["location"].astype(str)
    + "\nDescription: " + df["jobDescription"].astype(str)
)

texts = df["search_text"].tolist()

print(f"Generating embeddings for {len(texts)} jobs...")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("Building FAISS index...")

db = FAISSDatabase()
db.add_embeddings(embeddings)

os.makedirs("vectorstore", exist_ok=True)

db.save("vectorstore/jobs.index")

with open("vectorstore/jobs_metadata.pkl", "wb") as f:
    pickle.dump(df.to_dict("records"), f)

print("\n✅ Index built successfully!")

print(f"Indexed jobs: {len(df)}")