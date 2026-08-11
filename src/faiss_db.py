import faiss
import numpy as np


class FAISSDatabase:

    def __init__(self, dimension=384):
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)

    def search(self, embedding, k=10):
        embedding = np.asarray([embedding], dtype="float32")
        scores, indices = self.index.search(embedding, k)
        return scores[0], indices[0]

    def save(self, path):
        faiss.write_index(self.index, path)

    @staticmethod
    def load(path):
        db = FAISSDatabase()
        db.index = faiss.read_index(path)
        return db