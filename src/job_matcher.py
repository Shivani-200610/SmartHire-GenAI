import pickle

from src.embeddings import create_embedding
from src.faiss_db import FAISSDatabase


class JobMatcher:

    def __init__(self):
        self.db = FAISSDatabase.load("vectorstore/jobs.index")

        with open("vectorstore/jobs_metadata.pkl", "rb") as f:
            self.jobs = pickle.load(f)

    def find_matching_jobs(self, resume_text, top_k=10):

        embedding = create_embedding(resume_text)

        scores, indices = self.db.search(
            embedding,
            k=top_k
        )

        results = []

        for score, idx in zip(scores, indices):

            if idx == -1:
                continue

            job = self.jobs[idx].copy()

            job["match_score"] = round(float(score) * 100, 2)

            results.append(job)

        return results