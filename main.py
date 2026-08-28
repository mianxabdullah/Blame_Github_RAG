from fastapi import FastAPI
from src.search import RAGSearch

app = FastAPI(title="GitHub Codebase RAG API")

# One shared instance - loads the embedding model + LLM client once at startup.
# Different users stay isolated via the X-Session-Id header, not separate instances.
rag_search = RAGSearch()

