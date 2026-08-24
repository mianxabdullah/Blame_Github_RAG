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

def chunk_code(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split code documents using language-aware rules (won't cut mid-function where possible).
    Docs whose file_type isn't a recognized language fall back to the plain chunking() splitter.
    """
    by_language: dict = {}  # groups docs by their Language enum, so we can split each group together
    fallback_docs = [] # docs whose file_type isn't recognized, so we can split them with the generic chunking() function
 
    for doc in docs:
        file_type = doc.metadata.get("file_type", "")
        language = EXTENSION_TO_LANGUAGE.get(file_type)  #get the Language enum for this file_type, or None if not recognized
        if language:
            by_language.setdefault(language, []).append(doc) # group docs by language so we can split each group together
            """
            by_language is a dictionary where the keys are Language enums (like Language.PYTHON, Language.JS, etc.) 
            and the values are lists of Document objects that have that language.
            """
        else:
            fallback_docs.append(doc) # if the file_type isn't recognized, we add it to fallback_docs so we can split it with the generic chunking() function later.

    all_chunks = []
 
    for language, lang_docs in by_language.items(): # iterate over each language group and split them with the language-aware splitter
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        lang_chunks = splitter.split_documents(lang_docs) 
        print(f"{language.value}: {len(lang_docs)} files -> {len(lang_chunks)} chunks")
        all_chunks.extend(lang_chunks)

    if fallback_docs:
        fallback_chunks = _chunking(fallback_docs, chunk_size, chunk_overlap) # use the generic chunking() function for docs whose file_type isn't recognized
        all_chunks.extend(fallback_chunks)

    print(f"Total code chunks created: {len(all_chunks)} out of {len(docs)} documents.")
    return all_chunks 
