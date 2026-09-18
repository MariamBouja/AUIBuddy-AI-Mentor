"""Hybrid retrieval (dense + keyword) and query expansion for AUIBuddy."""

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from config import (
    VECTOR_K, VECTOR_FETCH_K, VECTOR_LAMBDA_MULT,
    BM25_K, VECTOR_WEIGHT, BM25_WEIGHT,
)


def expand_query(query: str) -> str:
    """Add domain synonyms for common AUI-specific terms to improve recall."""
    if "major" in query.lower():
        query += " programs, undergraduate, Bachelor of Science, Bachelor of Arts, degrees"
    return query


def build_hybrid_retriever(vectorstore, chunks):
    """Combine dense (vector/MMR) and sparse (BM25) retrieval."""
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": VECTOR_K,
            "fetch_k": VECTOR_FETCH_K,
            "lambda_mult": VECTOR_LAMBDA_MULT,
        },
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = BM25_K

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[VECTOR_WEIGHT, BM25_WEIGHT],
    )


def format_docs(docs) -> str:
    """Format retrieved chunks into a single citable context string."""
    parts = []
    for doc in docs:
        page = doc.metadata.get("page")
        label = f"p.{page + 1}" if page is not None else "p.?"
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
