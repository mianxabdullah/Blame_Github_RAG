import os
from src.vectorstore import SupabaseVectorStore
from src.embedding import EmbeddingManager
from src.chunking import chunk_code
from src.github_loader import load_github_repo
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

    def index_repo(self, repo_url: str, session_id: str, clear_existing: bool = True) -> dict:
        if clear_existing:
            self.vectorstore.clear(session_id)

        docs = load_github_repo(repo_url)
        chunks = chunk_code(docs)
        chunk_vectors = self.embedding_manager.embed_documents(chunks)

        metadatas = [{"text": chunk.page_content, **chunk.metadata} for chunk in chunks]
        self.vectorstore.add_embeddings(chunk_vectors, metadatas, session_id, repo_url=repo_url)

        return {"repo_url": repo_url, "files_indexed": len(docs), "chunks_indexed": len(chunks)}

    def search_and_summarize(self, query: str, session_id: str, top_k: int = 5) -> dict:
        if self.vectorstore.get_count(session_id) == 0:
            return {"answer": "No repo has been indexed yet. Call /index-repo first.", "sources": []}

