from dotenv import load_dotenv
import os
load_dotenv()

from rag_helper import RAGBase
from toyaikit.llm import AnthropicClient
from sqlitesearch import TextSearchIndex


index = TextSearchIndex(
    text_fields=["content"],
    keyword_fields=["filename"],
    db_path="faq.db"
)

client = AnthropicClient(
    model=os.getenv("ANTHROPIC_MODEL"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

assistant = RAGBase(
    index=index,
    llm_client=client,
    model=os.getenv("ANTHROPIC_MODEL")
)

index.close()



print(len(assistant.runner.loop("How does the agentic loop work, and how is it different from plain RAG?").all_messages))