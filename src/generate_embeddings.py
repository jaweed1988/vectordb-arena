import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

os.makedirs("embeddings", exist_ok=True)

df = pd.read_csv("../data/documents.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

np.save("../embeddings/document_embeddings.npy", embeddings)
df.to_csv("../data/documents_clean.csv", index=False)

print("Embeddings saved:", embeddings.shape)