# app/vector/retrieve.py

# import psycopg2
# from psycopg2.extras import RealDictCursor
# import numpy as np

# # DB connection info
# DB_CONN = {
#     "host": "localhost",
#     "port": 5432,
#     "dbname": "debugdb",
#     "user": "debug",
#     "password": "debug"
# }

# def retrieve(query_vector, top_k=5):
#     """
#     Retrieve top-k most similar log entries from pgvector.
#     """
#     conn = psycopg2.connect(**DB_CONN)
#     cur = conn.cursor(cursor_factory=RealDictCursor)

#     # Use cosine distance for similarity search
#     sql = f"""
#         SELECT log_text, embedding <=> %s AS distance
#         FROM log_embeddings
#         ORDER BY distance ASC
#         LIMIT {top_k};
#     """
#     cur.execute(sql, (query_vector,))
#     results = cur.fetchall()
    
#     cur.close()
#     conn.close()
    
#     return [r["log_text"] for r in results]

# # Example usage for testing
# if __name__ == "__main__":
#     # Example: random vector (replace with real embedding)
#     dummy_vector = np.random.rand(1536).tolist()
#     matches = retrieve(dummy_vector)
#     print("Top matches:")
#     for m in matches:
#         print("-", m[:100], "...")  # print first 100 chars




# import os
# import psycopg2

# # Load DB connection from environment
# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_PORT = os.getenv("DB_PORT", "5432")
# DB_NAME = os.getenv("DB_NAME", "debugdb")
# DB_USER = os.getenv("DB_USER", "debug")
# DB_PASSWORD = os.getenv("DB_PASSWORD", "debug")

# DB_CONN = {
#     "host": DB_HOST,
#     "port": DB_PORT,
#     "dbname": DB_NAME,
#     "user": DB_USER,
#     "password": DB_PASSWORD
# }

# def retrieve(query_vector, top_k=5):
#     """
#     Retrieve top_k most similar log entries from pgvector DB.
#     query_vector: Python list of floats (768-d embedding)
#     Returns: list of dicts with 'log_text' and 'distance'
#     """
#     # Convert Python list to Postgres pgvector format
#     vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"

#     sql = """
#         SELECT log_text, embedding <=> %s::vector AS distance
#         FROM log_embeddings
#         ORDER BY embedding <=> %s::vector
#         LIMIT %s
#     """

#     conn = psycopg2.connect(**DB_CONN)
#     cur = conn.cursor()
#     cur.execute(sql, (vector_str, vector_str, top_k))
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()

#     return [{"log_text": row[0], "distance": row[1]} for row in rows]
