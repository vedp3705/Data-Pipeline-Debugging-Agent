# from prefect import flow, task
# import pandas as pd
# import random
# import json
# import time


# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# LOGS_DIR = os.path.join(BASE_DIR, "../logs")
# os.makedirs(LOGS_DIR, exist_ok=True)

# LOG_FILE = os.path.join(LOGS_DIR, "run_logs.jsonl")


# def write_log(event):
#     with open(LOG_FILE, "a") as f:
#         f.write(json.dumps(event) + "\n")

# @task
# def extract():
#     write_log({"stage": "extract", "msg": "Extracting data"})
#     data = pd.DataFrame({"age": [25, 30, "unknown", 22]})
#     return data

# @task
# def transform(df):
#     write_log({"stage": "transform", "msg": "Transforming data"})
#     if random.random() < 0.5:
#         raise ValueError("TransformError: could not convert age to int")
#     df['age'] = df['age'].astype(int)
#     return df

# @task
# def load(df):
#     write_log({"stage": "load", "msg": "Loading data"})
#     time.sleep(1)
#     return True

# @flow
# def run_pipeline():
#     write_log({"event": "start", "msg": "Pipeline started"})
#     try:
#         df = extract()
#         df = transform(df)
#         load(df)
#         write_log({"event": "success", "msg": "Pipeline completed"})
#     except Exception as e:
#         write_log({"event": "error", "error": str(e)})
#         write_log({"event": "failed", "msg": "Pipeline failed"})

# if __name__ == "__main__":
#     run_pipeline()

# app/pipeline/etl.py
# import pandas as pd
# from pathlib import Path
# import json
# # from datetime import datetime
# from datetime import datetime, UTC
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt




# # Logs
# LOG_FILE = Path(__file__).parent.parent / "logs/run_logs.jsonl"
# LOG_FILE.parent.mkdir(exist_ok=True)

# def write_log(entry):
#     # entry["timestamp"] = datetime.utcnow().isoformat()
#     entry["timestamp"] = datetime.now(UTC).isoformat()
#     with open(LOG_FILE, "a") as f:
#         f.write(json.dumps(entry) + "\n")

# # Paths
# DATA_DIR = Path(__file__).parent / "data"
# INTERACTIONS = DATA_DIR / "llm_system_interactions.csv"
# SESSIONS = DATA_DIR / "llm_system_sessions_summary.csv"
# USERS = DATA_DIR / "llm_system_users_summary.csv"
# PROMPTS = DATA_DIR / "llm_system_prompts_lookup.csv"




# def run_pipeline():
#     write_log({"event": "start", "msg": "ETL pipeline started"})
#     try:
#         # ===== Extract =====
#         # df_interactions = pd.read_csv(INTERACTIONS)
#         # df_sessions = pd.read_csv(SESSIONS)
#         # df_users = pd.read_csv(USERS)
#         # df_prompts = pd.read_csv(PROMPTS)

#         df_interactions = pd.read_csv("data/llm_system_interactions.csv")
#         df_sessions     = pd.read_csv("data/llm_system_sessions_summary.csv")
#         df_users        = pd.read_csv("data/llm_system_users_summary.csv")
#         df_prompts      = pd.read_csv("data/llm_system_prompts_lookup.csv")

#         interactions = df_interactions[["session_id", "user_id", "prompt_text", "completion_text", "timestamp", "is_failure"]]



#         write_log({"stage": "extract", "msg": "Extracted CSVs"})

#         # ===== Transform =====
#         # Join tables
#         # df = df_interactions.merge(df_sessions, on="session_id", how="left") \
#         #                     .merge(df_users, on="user_id", how="left") \
#         #                     .merge(df_prompts, on="prompt_id", how="left")

#         df = (df_interactions
#         .merge(df_sessions, on=["session_id", "user_id"], how="left")
#         .merge(df_users, on="user_id", how="left")
#         .merge(df_prompts, on="prompt_id", how="left")
# )


#         write_log({"stage": "transform", "msg": "Joined tables"})

#         # Derived fields
#         df["latency_per_token"] = df["latency_ms"] / df["total_tokens"].replace(0, 1)

#         # Clean numeric fields
#         for col in ["latency_ms", "total_tokens", "prompt_tokens", "completion_tokens", "cost_usd"]:
#             df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

#         # Identify failures
#         df_failures = df[df["is_failure"] == True]

    


#         write_log({"stage": "transform", "msg": f"Processed {len(df_failures)} failures"})



#         # interactions = load_llm_interactions()
#         # df_interactions = enrich_with_embeddings(df_interactions)
#         # store(df_interactions)

#         # df_interactions

#         # score =  α * semantic_similarity
#         # + β * word_overlap
#         # + γ * recency_factor
#         # + δ * failure_correlation


#         model = SentenceTransformer("models/text-embedding-004")

#         def embed_col(texts):
#             return np.vstack(model.encode(texts, normalize_embeddings=True))

#         interactions["embedding"] = list(embed_col(interactions["prompt_text"].fillna("").astype(str)))

#         # Compute keyword-based overlap score (cheap lexical signal)
#         def word_overlap(a, b):
#             a, b = set(a.lower().split()), set(b.lower().split())
#             return len(a & b) / max(len(a | b), 1)

#         # Failure correlation weighting (more weight for failed sessions)
#         interactions["failure_weight"] = interactions["is_failure"].apply(lambda x: 3 if x else 1)

#         # Save for downstream RAG agent
#         rag_file = DATA_DIR / "rag_interactions.parquet"
#         interactions.to_parquet(rag_file, index=False)
#         write_log({"stage": "rag", "msg": f"Stored RAG embeddings ({len(interactions)} rows)"})





#         # ===== Load =====
#         df.to_csv(DATA_DIR / "etl_output.csv", index=False)
#         write_log({"stage": "load", "msg": f"Saved cleaned ETL output ({len(df)} rows)"})



#     except Exception as e:
#         write_log({"event": "error", "error": str(e)})
#         raise

# if __name__ == "__main__":
#     run_pipeline()






import pandas as pd
from pathlib import Path
import json
from datetime import datetime, UTC
# from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import os
import numpy as np

LOG_FILE = Path(__file__).parent.parent / "logs/run_logs.jsonl"
LOG_FILE.parent.mkdir(exist_ok=True)
load_dotenv()
API_KEY = os.getenv("API_KEY")
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def write_log(entry):
    entry["timestamp"] = datetime.now(UTC).isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

DATA_DIR = Path(__file__).parent / "data"
INTERACTIONS = DATA_DIR / "llm_system_interactions.csv"
SESSIONS = DATA_DIR / "llm_system_sessions_summary.csv"
USERS = DATA_DIR / "llm_system_users_summary.csv"
PROMPTS = DATA_DIR / "llm_system_prompts_lookup.csv"

def embed_col(texts, batch_size=32):
    vectors = []

    texts = [str(t) for t in texts]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        resp = client.embeddings.create(
            model="models/text-embedding-004",
            input=batch
        )

        vectors.extend([d.embedding for d in resp.data])

    return np.vstack(vectors)

def gemini_embed_batch(texts, batch_size=50):
    """
    Generate embeddings for a list of texts in batches using Gemini embeddings.
    Returns a list of np.array embeddings.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(
                model="text-embedding-004",
                input=batch
            )
            batch_emb = [np.array(item.embedding) for item in resp.data]
            embeddings.extend(batch_emb)
        except Exception as e:
            write_log({"event": "error", "msg": f"Embedding batch failed: {e}"})
            # fallback: add zero embeddings if batch fails
            embeddings.extend([np.zeros(768)] * len(batch))
    return embeddings

def run_pipeline():
    write_log({"event": "start", "msg": "ETL pipeline started"})
    try:
        df_interactions = pd.read_csv(INTERACTIONS)
        df_sessions     = pd.read_csv(SESSIONS)
        df_users        = pd.read_csv(USERS)
        df_prompts      = pd.read_csv(PROMPTS)

        df_interactions = df_interactions.rename(columns={
        "request_text": "prompt_text",
        "response_text_snippet": "completion_text",
        "timestamp_utc": "timestamp",
        })


        interactions = df_interactions[[
            "session_id","user_id","prompt_text","completion_text","timestamp","is_failure"
        ]]

        write_log({"stage": "extract", "msg": "Extracted CSVs"})

        df = (df_interactions
            .merge(df_sessions, on=["session_id","user_id"], how="left")
            .merge(df_users, on="user_id", how="left")
            .merge(df_prompts, on="prompt_id", how="left")
        )

        write_log({"stage": "transform", "msg": "Joined tables"})

        df["latency_per_token"] = df["latency_ms"] / df["total_tokens"].replace(0,1)

        num_cols = ["latency_ms","total_tokens","prompt_tokens","completion_tokens","cost_usd"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df_failures = df[df["is_failure"] == True]
        write_log({"stage": "transform", "msg": f"Processed {len(df_failures)} failures"})

        # model = SentenceTransformer("models/text-embedding-004")
        # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # emb_resp = client.embeddings.create(
        #     model="models/text-embedding-004",
        #     input=text
        # )
        # embedding = emb_resp.data[0].embedding  # 768-d



        # def embed_col(texts):
        #     return np.vstack(model.encode(texts, normalize_embeddings=True))

        # interactions["embedding"] = list(embed_col(
        #     interactions["prompt_text"].fillna("").astype(str)
        # ))


        # interactions["failure_weight"] = interactions["is_failure"].apply(lambda x: 3 if x else 1)

        interactions = interactions.copy()
        interactions.loc[:, "embedding"] = list(embed_col(
            interactions["prompt_text"].fillna("").astype(str)
        ))
        interactions.loc[:, "failure_weight"] = interactions["is_failure"].apply(lambda x: 3 if x else 1)
        interactions = df_interactions[["session_id","user_id","prompt_text","completion_text","timestamp","is_failure"]]

        interactions = interactions.copy()
        interactions["embedding"] = gemini_embed_batch(interactions["prompt_text"].fillna("").astype(str).tolist())



        rag_file = DATA_DIR / "rag_interactions.parquet"
        interactions.to_parquet(rag_file, index=False)
        write_log({"stage": "rag", "msg": f"Stored RAG embeddings ({len(interactions)} rows)"})

        df.to_csv(DATA_DIR / "etl_output.csv", index=False)
        write_log({"stage": "load", "msg": f"Saved cleaned ETL output ({len(df)} rows)"})

    except Exception as e:
        write_log({"event": "error", "error": str(e)})
        raise

if __name__ == "__main__":
    run_pipeline()