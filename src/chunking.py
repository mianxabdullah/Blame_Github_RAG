from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_core.documents import Document

EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".html": Language.HTML,
}

def _chunking(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""], 
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[INFO] Total chunks created: {len(chunks)} out of {len(docs)} documents.")
    return chunks
