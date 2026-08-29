from fastapi import FastAPI, Header, HTTPException
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

@app.post("/index-repo", response_model=IndexResponse)
def index_repo(request: IndexRequest, x_session_id: str = Header(...)):
    try:
        result = rag_search.index_repo(request.repo_url, session_id=x_session_id, clear_existing=request.clear_existing)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index repo: {e}")