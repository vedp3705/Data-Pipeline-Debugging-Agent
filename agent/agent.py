import streamlit as st
import pandas as pd
import psycopg2
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

load_dotenv()
API_KEY = os.getenv("API_KEY")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "debugdb")
DB_USER = os.getenv("DB_USER", "debug")
DB_PASSWORD = os.getenv("DB_PASSWORD", "debug")

DB_CONN = {
    "host": DB_HOST,
    "port": DB_PORT,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD
}

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def retrieve_logs(query_vector, top_k=5):
    vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"
    sql = """
        SELECT log_text, embedding <=> %s::vector AS distance
        FROM log_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    try:
        conn = psycopg2.connect(**DB_CONN)
        cur = conn.cursor()
        cur.execute(sql, (vector_str, vector_str, top_k))
        rows = cur.fetchall()
    except Exception as e:
        st.error(f"DB Error: {e}")
        rows = []
    finally:
        if cur: cur.close()
        if conn: conn.close()
    
    logs = [{"log_text": row[0], "distance": row[1]} for row in rows]

    unique = {}
    for item in logs:
        text = item["log_text"]
        if text not in unique or item["distance"] < unique[text]["distance"]:
            unique[text] = item

    return list(unique.values())
    # return [{"log_text": row[0], "distance": row[1]} for row in rows]

def ask_agent(user_query, top_k=5):
    try:
        emb = client.embeddings.create(
            model="text-embedding-004",
            input=user_query
        )
        query_vector = emb.data[0].embedding
    except Exception as e:
        return f"Error generating embedding: {e}", []

    results = retrieve_logs(query_vector, top_k=top_k)
    context_text = "\n".join([f"- {r['log_text']}" for r in results]) if results else "No relevant logs found."

    # prompt = f"""You are a senior assistant.

    # User Query: {user_query}

    # Relevant logs from previous debugging sessions:
    # {context_text}

    # Provide a clear, practical solution or explanation for the user's issue.
    # """

    prompt = f"""You are a senior debugging assistant.

    User Query: {user_query}

    Relevant logs from previous debugging sessions:
    {context_text}

    Respond with a well-formatted solution using:

    Summary:
    Causes:
    Fix:
    Verification:

    Do NOT use markdown bold syntax (**text**) or bullet asterisks.
    """


    try:
        response = client.chat.completions.create(
            model="models/gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error calling Gemini API: {e}"

        answer = response.choices[0].message.content

        answer = (
            answer.replace("**", "")
                .replace("*", "")
                .replace("•", "- ")
                .replace("–", "- ")
                .strip()
        )


    return answer, results

def classify_query(q: str) -> str:
    q_low = q.lower()

    if any(k in q_low for k in ["fail", "error", "crash", "exception", "stacktrace"]):
        return "failure"

    if any(k in q_low for k in ["latency", "slow", "timeout", "duration", "performance"]):
        return "performance"

    return "general"

def orchestrator(user_query, top_k=5):
    category = classify_query(user_query)

    if category == "failure":
        return failure_agent(user_query, top_k)

    if category == "performance":
        return performance_agent(user_query, top_k)

    return general_agent(user_query, top_k)

def failure_agent(user_query, top_k=5):
    answer, logs = ask_agent(user_query, top_k=top_k)
    logs = [r for r in logs if "fail" in r['log_text'].lower() or "error" in r['log_text'].lower()]
    return answer, logs

def performance_agent(user_query, top_k=5):
    answer, logs = ask_agent(user_query, top_k=top_k)
    logs = [r for r in logs if "latency" in r['log_text'].lower() or "timeout" in r['log_text'].lower()]
    return answer, logs

def general_agent(user_query, top_k=5):
    return ask_agent(user_query, top_k=top_k)

st.title("LLM Ops Pipeline Dashboard (RAG + Clusters + Gemini)")

DATA_FILE = Path(__file__).parent.parent / "pipeline/data/etl_output.csv"
RAG_FILE = Path(__file__).parent.parent / "pipeline/data/rag_interactions.parquet"

try:
    df = pd.read_csv(DATA_FILE)
    failures = df[df["is_failure"] == True]
    st.subheader("Failures Over Time")
    failure_counts = failures.groupby("date_utc").size()
    st.line_chart(failure_counts)
except Exception as e:
    st.warning(f"Could not load ETL CSV: {e}")

try:
    st.subheader("Latency per Model")
    latency_means = df.groupby("model_name")["latency_ms"].mean()
    st.bar_chart(latency_means)
except Exception as e:
    st.warning(f"Could not calculate latency stats: {e}")

try:
    interactions = pd.read_parquet(RAG_FILE)
    emb = np.vstack(interactions["embedding"].values)
    k = max(3, min(20, len(emb)//50))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    interactions["cluster_id"] = kmeans.fit_predict(emb)

    st.subheader("Top Clusters (Semantic + Failure Aware)")
    clusters = []
    for cid in range(k):
        subset = interactions[interactions["cluster_id"] == cid]
        if len(subset) == 0:
            continue
        centroid = kmeans.cluster_centers_[cid]
        sims = subset["embedding"].apply(lambda x: np.dot(x, centroid))
        rep_idx = sims.idxmax()
        rep_text = subset.loc[rep_idx, "prompt_text"] if rep_idx in subset.index else "(no representative)"
        clusters.append({
            "cluster_id": cid,
            "size": len(subset),
            "failure_rate": subset["is_failure"].mean(),
            "rep": rep_text
        })

    cluster_df = pd.DataFrame(clusters)
    cluster_df = cluster_df.sort_values(by=["size","failure_rate"], ascending=[False,False])

    st.dataframe(
        cluster_df.rename(columns={
            "cluster_id": "Cluster",
            "size": "Size",
            "failure_rate": "Fail %",
            "rep": "Representative Query"
        }),
        width='stretch'
    )

    st.subheader("Cluster Visualization (PCA)")
    pca = PCA(n_components=2)
    p2 = pca.fit_transform(emb)
    plt.figure(figsize=(6,4))
    plt.scatter(p2[:,0], p2[:,1], c=interactions["cluster_id"], s=18)
    plt.title("PCA Cluster Visualization")
    st.pyplot(plt)

    st.subheader("Cluster Visualization (UMAP)")
    u = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine").fit_transform(emb)
    plt.figure(figsize=(6,4))
    plt.scatter(u[:,0], u[:,1], c=interactions["cluster_id"], s=18)
    plt.title("UMAP Cluster Visualization")
    st.pyplot(plt)

except Exception as e:
    st.warning(f"Could not load or cluster RAG interactions: {e}")

st.subheader("Ask the Debugging Agent")
user_query = st.text_input("Enter your query here:")
top_k = st.slider("Number of relevant logs to retrieve:", min_value=1, max_value=10, value=5)

if st.button("Get Diagnosis") and user_query.strip():
    with st.spinner("Generating diagnosis..."):
        # answer, logs = ask_agent(user_query, top_k=top_k)
        answer, logs = orchestrator(user_query, top_k=top_k)

    if logs:
        st.markdown(f"### Top {len(logs)} Relevant Logs")
        for i, log in enumerate(logs, 1):
            st.text_area(f"Log {i} (distance={log['distance']:.4f})", log["log_text"], height=120)
    else:
        st.info("No relevant logs found.")

    st.markdown("### Gemini Solution")
    st.text_area("Solution", answer, height=300)