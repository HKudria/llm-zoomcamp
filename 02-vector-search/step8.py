from sentence_transformers import SentenceTransformer
from ragPostgress import RAGVector
from anthropic import Anthropic
from dotenv import load_dotenv
from db_postgres import pgvector_search
import psycopg

import os
load_dotenv()

client = Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

conn = psycopg.connect(os.getenv("POSTGRESS"))


vector_assistant = RAGVector(
    embedder=model,
    index=None,
    conn= conn,
    llm_client=client,
    model=os.getenv("ANTHROPIC_MODEL")
)


query = "the program has already begun, can I still sign up?"

print(vector_assistant.rag(query))
