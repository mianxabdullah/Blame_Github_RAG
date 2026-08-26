import os
from supabase import create_client, Client

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