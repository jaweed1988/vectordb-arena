from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

dataset = load_dataset("ag_news", split="train")

df = pd.DataFrame(dataset)

df = df.rename(columns={"label": "category"})
df["id"] = range(1, len(df) + 1)

df = df[["id", "text", "category"]]

# Start with 10,000 first. Later increase to 50,000 or 100,000.
df = df.head(10000)

df.to_csv("../data/documents.csv", index=False)

print("Saved dataset:", df.shape)
print(df.head())