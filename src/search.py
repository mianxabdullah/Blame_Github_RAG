import os
from src.vectorstore import SupabaseVectorStore
from src.embedding import EmbeddingManager
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

class RAGSearch:
    """One shared instance for the whole server. Individual users are kept apart
    by passing session_id into index_repo() / search_and_summarize(), not by
    creating a separate RAGSearch per user (that would reload the embedding model
    and LLM client repeatedly, which is expensive)."""

    def __init__(self, llm_model: str = "openai/gpt-oss-120b"):
        self.vectorstore = SupabaseVectorStore()
        self.embedding_manager = EmbeddingManager()

        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")