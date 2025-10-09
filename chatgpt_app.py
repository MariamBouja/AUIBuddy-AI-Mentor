from rich import print
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os, shutil

# ====== SETTINGS ======
PDF_PATH = "24-25.pdf"              # change to your actual catalog file
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "auibuddy_catalog"

# Use a smaller Ollama model (fits 8GB RAM)
# Try dolphin-phi first, fallback to llama3-3b
local_llm = ChatOllama(
    model="dolphin-phi:latest",
    temperature=0
)  
# or: local_llm = ChatOllama(model="llama3:3b-instruct-q4_K_M")

# ====== VECTORSTORE ======
def build_vectorstore(pdf_path):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    loader = PyPDFLoader(pdf_path)
    pdf_documents = loader.load()
    print(f"[bold cyan]Loaded {len(pdf_documents)} pages from {pdf_path}[/bold cyan]")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(pdf_documents)
    print(f"[bold cyan]Created {len(chunks)} chunks from PDF[/bold cyan]")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
    )
    return vectorstore, pdf_documents


def load_vectorstore():
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
    )

# ====== LOAD OR BUILD DB ======
if os.path.exists(CHROMA_PATH):
    vectorstore = load_vectorstore()
    print(f"[bold green]Chroma DB loaded from {CHROMA_PATH}[/bold green]")
else:
    vectorstore, docs = build_vectorstore(PDF_PATH)

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 24, "lambda_mult": 0.3}
    )
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 6

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # adjust if one dominates
)
    
    
    # --- NEW: BM25 retriever ---
from langchain_community.retrievers import BM25Retriever
bm25 = BM25Retriever.from_documents(docs)
bm25.k = 5

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 24, "lambda_mult": 0.3}
)


# Format retrieved docs into a single string
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# ====== PROMPT ======
prompt_template = """
You are an academic mentor for AUI students.
Answer ONLY with facts from the context. If the answer is not clearly present, say:
"I couldn’t find this in the AUI catalog."

Task: If the user asks for “majors”, list each major exactly as named in the catalog.

Context:
{context}

Question:
{question}

Provide a concise answer and include "Sources: p.X, p.Y".
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# ====== RAG CHAIN ======
chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | local_llm
    | StrOutputParser()
)

# ====== TERMINAL CHAT LOOP ======
print("[bold magenta]🤖 AUIBuddy Chatbot ready! Type your questions (type 'exit' to quit)[/bold magenta]\n")
while True:
    query = input("[You]: ")
    if query.strip().lower() in ["exit", "quit"]:
        print("[bold red]Goodbye![/bold red]")
        break
    answer = chain.invoke(query)
    print(f"[Bot]: {answer}\n")
    
    
def format_docs(docs):
    parts = []
    for d in docs:
        p = d.metadata.get("page", None)
        parts.append(f"[p.{p+1 if p is not None else '?'}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

user_query = query
if "major" in query.lower():
    user_query += " undergraduate programs, degree programs, bachelor of science, school of"
docs = retriever.get_relevant_documents(user_query)


docs = retriever.get_relevant_documents(user_query)
print("\n[bold yellow]Retrieved Context:[/bold yellow]\n")
for d in docs[:3]:
    p = d.metadata.get("page", "?")
    print(f"Page {p+1}:\n{d.page_content[:400]}...\n---\n")
