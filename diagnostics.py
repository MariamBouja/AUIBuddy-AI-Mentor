"""Standalone diagnostics for inspecting each stage of the RAG pipeline:
PDF extraction, chunking, hybrid retrieval, and embedding similarity.
"""

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL
from vectorstore import load_and_split_pdf, build_vectorstore
from retrieval import build_hybrid_retriever, expand_query


def check_pdf_extraction():
    chunks_source = load_and_split_pdf()
    print(f"✅ Loaded and split PDF into {len(chunks_source)} chunks")
    for i, c in enumerate(chunks_source[:3]):
        print(f"\n--- Chunk {i + 1} ---\n{c.page_content[:400]}\n---")
    return chunks_source


def check_retrieval(chunks, query="what are the majors at AUI?"):
    vectorstore = build_vectorstore(chunks)
    retriever = build_hybrid_retriever(vectorstore, chunks)

    expanded_query = expand_query(query)
    print(f"\n🔍 Expanded query: {expanded_query}\n")

    docs = retriever.invoke(expanded_query)
    print(f"✅ Retrieved {len(docs)} chunks for query: '{query}'\n")

    for i, d in enumerate(docs):
        page = d.metadata.get("page", "?")
        print(f"\n--- Retrieved Chunk {i + 1} (Page {page}) ---\n{d.page_content[:400]}\n---")

    return docs


def check_embedding_similarity():
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    q1 = "majors"
    q2 = "Bachelor of Science in Computer Science"
    emb1 = embedding_model.embed_query(q1)
    emb2 = embedding_model.embed_query(q2)
    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"✅ Cosine similarity between '{q1}' and '{q2}': {cosine_sim:.4f}")


if __name__ == "__main__":
    chunks = check_pdf_extraction()
    check_retrieval(chunks, "what are the majors at AUI?")
    check_embedding_similarity()
