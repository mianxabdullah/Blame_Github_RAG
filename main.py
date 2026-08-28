from fastapi import FastAPI
from pydantic import BaseModel
from src.search import RAGSearch

app = FastAPI(title="GitHub Codebase RAG API")

# One shared instance - loads the embedding model + LLM client once at startup.
# Different users stay isolated via the X-Session-Id header, not separate instances.
rag_search = RAGSearch()

class IndexRequest(BaseModel):
    repo_url: str
    clear_existing: bool = True


class IndexResponse(BaseModel):
    repo_url: str
    files_indexed: int
    chunks_indexed: int
