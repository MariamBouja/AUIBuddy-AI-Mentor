"""Build and load the Chroma vector store for AUIBuddy."""

import os

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    PDF_PATH, CHROMA_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS,
)


def load_and_split_pdf(pdf_path: str = PDF_PATH):
    """Load the PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and persist them to a new Chroma collection."""
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
    )


def load_vectorstore():
    """Load an existing Chroma collection from disk."""
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
    )


def get_vectorstore_and_chunks():
    """Return a ready-to-use vectorstore, building it on first run.

    Also returns the raw chunks, since the BM25 retriever needs the
    documents directly rather than the vector store.
    """
    chunks = load_and_split_pdf()

    if os.path.exists(CHROMA_PATH):
        vectorstore = load_vectorstore()
        print(f"Loaded existing Chroma DB from {CHROMA_PATH}")
    else:
        vectorstore = build_vectorstore(chunks)
        print(f"Built new Chroma DB at {CHROMA_PATH}")

    return vectorstore, chunks
