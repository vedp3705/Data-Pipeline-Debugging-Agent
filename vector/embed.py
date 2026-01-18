from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def embed_text(text: str):
    resp = client.embeddings.create(
        model="text-embedding-004",
        input=text
    )
    return resp.data[0].embedding
