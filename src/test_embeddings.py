from embeddings import create_embedding

text = """
Python
TensorFlow
Deep Learning
Computer Vision
"""

embedding = create_embedding(text)

print("Embedding Shape:", embedding.shape)
print(embedding[:10])