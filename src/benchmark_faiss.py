import time
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("../data/documents_clean.csv")
embeddings = np.load("../embeddings/document_embeddings.npy").astype("float32")

model = SentenceTransformer("all-MiniLM-L6-v2")

queries = [
    {"query": "latest technology and artificial intelligence", "expected_category": 3},
    {"query": "global politics and international news", "expected_category": 0},
    {"query": "stock market and business news", "expected_category": 2},
    {"query": "football basketball and sports teams", "expected_category": 1},
]

top_k = 5

start_build = time.perf_counter()

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

end_build = time.perf_counter()
index_build_time_sec = end_build - start_build

results = []
latencies = []
category_matches = []

total_start = time.perf_counter()

for item in queries:
    query = item["query"]
    expected_category = item["expected_category"]

    query_vector = model.encode([query]).astype("float32")

    start = time.perf_counter()
    distances, indices = index.search(query_vector, top_k)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

    retrieved_categories = df.iloc[indices[0]]["category"].tolist()

    match_count = retrieved_categories.count(expected_category)
    category_match_rate = match_count / top_k
    category_matches.append(category_match_rate)

    results.append({
        "system": "FAISS",
        "query": query,
        "expected_category": expected_category,
        "latency_ms": latency_ms,
        "top_k": top_k,
        "category_match_rate": category_match_rate,
        "top_result": df.iloc[indices[0][0]]["text"]
    })

total_end = time.perf_counter()
total_time_sec = total_end - total_start

summary = {
    "system": "FAISS",
    "dataset_size": len(df),
    "top_k": top_k,
    "num_queries": len(queries),
    "index_build_time_sec": index_build_time_sec,
    "avg_latency_ms": np.mean(latencies),
    "p50_latency_ms": np.percentile(latencies, 50),
    "p95_latency_ms": np.percentile(latencies, 95),
    "p99_latency_ms": np.percentile(latencies, 99),
    "min_latency_ms": np.min(latencies),
    "max_latency_ms": np.max(latencies),
    "throughput_qps": len(queries) / total_time_sec,
    "avg_category_match_rate": np.mean(category_matches)
}

pd.DataFrame(results).to_csv("../results/faiss_results.csv", index=False)
pd.DataFrame([summary]).to_csv("../results/faiss_summary.csv", index=False)

print("FAISS detailed results:")
print(pd.DataFrame(results))

print("\nFAISS summary:")
print(pd.DataFrame([summary]))