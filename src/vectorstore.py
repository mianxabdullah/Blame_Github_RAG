import os
from supabase import create_client, Client
import numpy as np
from typing import List, Any

class SupabaseVectorStore:
    """Stores and searches chunk embeddings in Supabase (Postgres + pgvector).
    Every row is tagged with a session_id so different users' indexed repos stay isolated.
    """
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        self.client: Client = create_client(url, key)
        print("[INFO] Connected to Supabase")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any], session_id: str, repo_url: str = None):
        """Insert chunks + their vectors, tagged with session_id."""
        rows = []
        for meta, vector in zip(metadatas, embeddings):
            rows.append({
                "session_id": session_id,
                "repo_url": repo_url,
                "file_path": meta.get("file_path"),
                "text": meta.get("text", ""),
                "metadata": meta,
                "embedding": vector.tolist(),  # Supabase's client expects a plain list, not numpy array
            })

        # Insert in batches - Supabase/Postgres has payload size limits on a single request
        batch_size = 200
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            self.client.table("chunks").insert(batch).execute()