import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

df = pd.read_csv("../data/documents_clean.csv")
embeddings = np.load("../embeddings/document_embeddings.npy").astype("float32")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(":memory:")

collection_name = "documents"
top_k = 5

start_build = time.perf_counter()

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embeddings.shape[1],
        distance=Distance.COSINE
    )
)

points = []

for i, row in df.iterrows():
    points.append(
        PointStruct(
            id=int(row["id"]),
            vector=embeddings[i].tolist(),
            payload={
                "text": row["text"],
                "category": int(row["category"])
            }
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points
)

end_build = time.perf_counter()
index_build_time_sec = end_build - start_build

queries = [
    {"query": "latest technology and artificial intelligence", "expected_category": 3},
    {"query": "global politics and international news", "expected_category": 0},
    {"query": "stock market and business news", "expected_category": 2},
    {"query": "football basketball and sports teams", "expected_category": 1},
]

results = []
latencies = []
category_matches = []

total_start = time.perf_counter()

for item in queries:
    query = item["query"]
    expected_category = item["expected_category"]

    query_vector = model.encode(query).tolist()

    start = time.perf_counter()

    hits = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    ).points

    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

    retrieved_categories = [hit.payload["category"] for hit in hits]

    match_count = retrieved_categories.count(expected_category)
    category_match_rate = match_count / len(hits) if hits else 0
    category_matches.append(category_match_rate)

    results.append({
        "system": "Qdrant",
        "query": query,
        "expected_category": expected_category,
        "latency_ms": latency_ms,
        "top_k": top_k,
        "category_match_rate": category_match_rate,
        "top_result": hits[0].payload["text"] if hits else None,
        "top_score": hits[0].score if hits else None
    })

total_end = time.perf_counter()
total_time_sec = total_end - total_start

summary = {
    "system": "Qdrant",
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
    "throughput_qps": len(queries) / total_time_sec if total_time_sec > 0 else 0,
    "avg_category_match_rate": np.mean(category_matches)
}

os.makedirs("../results", exist_ok=True)

pd.DataFrame(results).to_csv("../results/qdrant_results.csv", index=False)
pd.DataFrame([summary]).to_csv("../results/qdrant_summary.csv", index=False)

print("Qdrant detailed results:")
print(pd.DataFrame(results))

print("\nQdrant summary:")
print(pd.DataFrame([summary]))