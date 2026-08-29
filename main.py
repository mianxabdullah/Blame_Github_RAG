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

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class Source(BaseModel):
    file_path: str | None

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

@app.get("/")
def home():
    return {
        "message": "GitHub Codebase RAG API",
        "status": "active",
        "endpoints": "/health, /index-repo, /query",
    }

@app.get("/health")
def health():
    return {"status": "ok"}