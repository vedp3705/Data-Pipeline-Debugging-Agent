import os
import psycopg2
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

CSV_PATH = "pipeline/data/etl_output.csv"

BATCH_SIZE = 32

DB_CONN = {
    "host": "localhost", 
    "port": 5432,
    "dbname": "debugdb",
    "user": "debug",
    "password": "debug"
}

load_dotenv()
API_KEY = os.getenv("API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def embed_batch(texts):
    """
    Get embeddings for a batch of texts.
    """
    resp = client.embeddings.create(
        model="text-embedding-004", 
        input=texts
    )
    return [np.array(e.embedding, dtype=np.float32) for e in resp.data]

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded rows: {len(df)}")

    if "request_text" in df.columns:
        text_col = "request_text"
    else:
        text_col = df.columns[0] 

    conn = psycopg2.connect(**DB_CONN)
    cur = conn.cursor()

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i:i+BATCH_SIZE]
        texts = batch[text_col].fillna("").tolist()

        embeddings = embed_batch(texts)

        args_str = ",".join(
            cur.mogrify(
                "(%s,%s,%s,%s)",
                (
                    row.interaction_id,              
                    getattr(row, "session_id", ""), 
                    text,
                    emb.tolist(),
                ),
            ).decode("utf-8")
            for row, text, emb in zip(batch.itertuples(index=False), texts, embeddings)
        )

        cur.execute(f"INSERT INTO log_embeddings (interaction_id, session_id, log_text, embedding) VALUES {args_str}")
        conn.commit()

    cur.close()
    conn.close()
    print("done")