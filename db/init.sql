CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS log_embeddings (
    id SERIAL PRIMARY KEY,
    log_text TEXT NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMP DEFAULT NOW()
);
