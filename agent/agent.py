import json
import os
from pathlib import Path
from openai import OpenAI  # We'll use Gemini via OpenAI-compatible API
from dotenv import load_dotenv
import chromadb

# ===== Paths =====
LOG_FILE = Path(__file__).parent.parent / "logs/run_logs.jsonl"

# ===== Initialize Vector Store =====
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("pipeline_errors")

# ===== Initialize Gemini Client =====
load_dotenv()
API_KEY = os.getenv("API_KEY")
client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ===== Ingest Logs into Vector Store =====

def clear_collection():
    # Get all IDs in collection
    all_docs = collection.get()
    all_ids = all_docs["ids"]
    if all_ids:
        collection.delete(ids=all_ids)


def ingest_logs():
    clear_collection()
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                if "error" in data:
                    collection.add(
                        documents=[data["error"]],
                        metadatas=[{
                            "stage": str(data.get("stage", "N/A")),
                            "msg": str(data.get("msg", "N/A"))
                        }],
                        ids=[f"err{i}"]
                    )


# ===== Query Gemini Agent =====
def diagnose():
    ingest_logs()
    
    if collection.count() == 0:
        return "No errors detected in the ETL pipeline."
    
    # Retrieve most recent error
    result = collection.query(query_texts=["Explain the error and suggest a fix."], n_results=1)
    error_text = result['documents'][0][0] if result['documents'][0] else "No details available."
    
    # Ask Gemini to analyze
    prompt = f"Pipeline error: {error_text}\nExplain the root cause and suggest a fix in simple terms."
    
    resp = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return resp.choices[0].message.content
