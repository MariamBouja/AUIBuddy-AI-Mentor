# AUIBuddy — AI Campus Mentor Chatbot

AUIBuddy is a Retrieval-Augmented Generation (RAG) chatbot built to help
Al Akhawayn University students get quick, grounded answers about majors,
programs, and campus information from the official course catalog — instead
of digging through a PDF by hand.

## Why hybrid retrieval

Pure vector search struggles with exact terms (program names, course codes)
that don't have strong semantic neighbors. AUIBuddy combines two retrievers:

- **Vector search (MMR)** over `nomic-embed-text` embeddings in ChromaDB,
  for semantic/paraphrased questions
- **BM25** keyword search, for exact-term matches vector search tends to miss

Results are merged with `EnsembleRetriever` (60% vector / 40% BM25 by
default, tunable in `config.py`).

A lightweight **query expansion** step also appends related terms when a
question mentions "majors," since the catalog uses several different phrasings
(Bachelor of Science, undergraduate programs, etc.) for the same concept.

## Project structure

```
auibuddy/
├── config.py         # paths, model names, chunking & retrieval settings
├── vectorstore.py     # PDF loading, chunking, ChromaDB build/load
├── retrieval.py        # hybrid retriever + query expansion + context formatting
├── rag_chain.py        # prompt template + LangChain RAG chain
├── main.py             # terminal chat loop (entry point)
├── diagnostics.py      # stage-by-stage pipeline inspection (chunking, retrieval, embedding similarity)
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
# Ollama running locally with dolphin-phi and nomic-embed-text pulled
python main.py
```

Run `python diagnostics.py` to inspect chunking quality, retrieved
passages, and embedding similarity scores independently of the chat loop —
useful when tuning retrieval parameters.

## Status

Actively iterating on retrieval quality — current focus areas are chunk-size
tuning for catalog-style tabular content and expanding the query-expansion
list beyond "majors."

## Stack

Python · LangChain · Ollama (local LLM + embeddings) · ChromaDB · BM25
