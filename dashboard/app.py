import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="VectorDB Arena", layout="wide")

st.title("VectorDB Arena")
st.subheader("Semantic Similarity Search Benchmark")

# -----------------------------
# Load detailed query results
# -----------------------------

faiss_results = pd.read_csv("../results/faiss_results.csv")
qdrant_results = pd.read_csv("../results/qdrant_results.csv")

results_df = pd.concat(
    [faiss_results, qdrant_results],
    ignore_index=True
)

# -----------------------------
# Load benchmark summaries
# -----------------------------

faiss_summary = pd.read_csv("../results/faiss_summary.csv")
qdrant_summary = pd.read_csv("../results/qdrant_summary.csv")

summary_df = pd.concat(
    [faiss_summary, qdrant_summary],
    ignore_index=True
)

# -----------------------------
# Benchmark Summary Table
# -----------------------------

st.header("Benchmark Summary")

st.dataframe(summary_df)

# -----------------------------
# Latency Metrics
# -----------------------------

st.header("Latency Metrics")

latency_cols = [
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms"
]

fig, ax = plt.subplots(figsize=(10, 5))

x = range(len(summary_df))

bar_width = 0.2

for i, metric in enumerate(latency_cols):
    ax.bar(
        [p + i * bar_width for p in x],
        summary_df[metric],
        width=bar_width,
        label=metric
    )

ax.set_xticks([p + 1.5 * bar_width for p in x])
ax.set_xticklabels(summary_df["system"])

ax.set_ylabel("Latency (ms)")
ax.set_title("Latency Comparison")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Throughput Comparison
# -----------------------------

st.header("Throughput (QPS)")

fig2, ax2 = plt.subplots(figsize=(6, 4))

ax2.bar(
    summary_df["system"],
    summary_df["throughput_qps"]
)

ax2.set_ylabel("Queries Per Second")
ax2.set_title("Throughput Comparison")

st.pyplot(fig2)

# -----------------------------
# Index Build Time
# -----------------------------

st.header("Index Build Time")

fig3, ax3 = plt.subplots(figsize=(6, 4))

ax3.bar(
    summary_df["system"],
    summary_df["index_build_time_sec"]
)

ax3.set_ylabel("Seconds")
ax3.set_title("Index Build Time")

st.pyplot(fig3)

# -----------------------------
# Semantic Match Quality
# -----------------------------

st.header("Category Match Rate")

fig4, ax4 = plt.subplots(figsize=(6, 4))

ax4.bar(
    summary_df["system"],
    summary_df["avg_category_match_rate"]
)

ax4.set_ylabel("Average Match Rate")
ax4.set_ylim(0, 1)

ax4.set_title("Semantic Retrieval Quality")

st.pyplot(fig4)

# -----------------------------
# Query-Level Results
# -----------------------------

st.header("Detailed Query Results")

st.dataframe(results_df)

# -----------------------------
# Individual Query Cards
# -----------------------------

st.header("Per Query Analysis")

for _, row in results_df.iterrows():

    st.markdown(f"### {row['system']}")

    st.write(f"**Query:** {row['query']}")
    st.write(f"**Latency:** {row['latency_ms']:.4f} ms")
    st.write(f"**Category Match Rate:** {row['category_match_rate']:.2f}")
    st.write(f"**Top Result:** {row['top_result']}")

    st.divider()