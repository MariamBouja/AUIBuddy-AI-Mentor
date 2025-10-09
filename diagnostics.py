from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import numpy as np

PDF_PATH = "24-25.pdf"


# --- Query expansion helper ---
def expand_query(query: str) -> str:
    if "major" in query.lower():
        query += " programs, undergraduate, Bachelor of Science, Bachelor of Arts, degrees"
    return query


def check_pdf_extraction():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()[15:]  # start from page 15 to skip front matter
    print(f"✅ Loaded {len(docs)} pages (starting from page 15)")
    for i, d in enumerate(docs[:3]):  # just preview first 3
        print(f"\n--- Page {i+16} ---\n{d.page_content[:800]}\n---")
    return docs


def check_chunking(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):  # show first 3 chunks
        print(f"\n--- Chunk {i+1} ---\n{c.page_content[:800]}\n---")
    return chunks


def check_retrieval(chunks, query="what are the majors at AUI?"):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Vector store for semantic search
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db_diag"
    )

    # Create vector retriever
    retriever_vector = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.3}
    )

    # Create BM25 retriever (keyword-based)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5

    # Combine both (hybrid retriever)
    retriever = EnsembleRetriever(
        retrievers=[retriever_vector, bm25],
        weights=[0.5, 0.5]
    )

    # Expand query with synonyms
    expanded_query = expand_query(query)
    print(f"\n🔍 Expanded query: {expanded_query}\n")

    # Retrieve documents
    docs = retriever.get_relevant_documents(expanded_query)
    print(f"✅ Retrieved {len(docs)} chunks for query: '{query}'\n")

    for i, d in enumerate(docs):
        p = d.metadata.get("page", "?")
        print(f"\n--- Retrieved Chunk {i+1} (Page {p}) ---\n{d.page_content[:800]}\n---")

    return docs


def check_embedding_similarity():
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    q1 = "majors"
    q2 = "Bachelor of Science in Computer Science"
    emb1 = embedding_model.embed_query(q1)
    emb2 = embedding_model.embed_query(q2)
    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"✅ Cosine similarity between '{q1}' and '{q2}': {cosine_sim:.4f}")


if __name__ == "__main__":
    # Step 1: PDF text extraction
    docs = check_pdf_extraction()

    # Step 2: Chunking
    chunks = check_chunking(docs)

    # Step 3: Hybrid retrieval test
    check_retrieval(chunks, "what are the majors at AUI?")

    # Step 4: Embedding similarity
    check_embedding_similarity()
