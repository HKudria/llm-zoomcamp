from sentence_transformers import SentenceTransformer
from ragVector import RAGVector
from anthropic import Anthropic
from dotenv import load_dotenv
from sqlitesearch import VectorSearchIndex

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)

import os
load_dotenv()

client = Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)


vector_assistant = RAGVector(
    embedder=model,
    index=vs_index,
    llm_client=client,
    model=os.getenv("ANTHROPIC_MODEL")
)


query = "the program has already begun, can I still sign up?"
query_vector = model.encode(query)

print(vector_assistant.rag(query))
vs_index.close()