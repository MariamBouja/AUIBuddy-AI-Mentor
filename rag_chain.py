"""Build the RAG chain: retrieval + prompt + LLM."""

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import LLM_MODEL
from retrieval import format_docs

PROMPT_TEMPLATE = """
You are an academic mentor for AUI students.
Answer ONLY with facts from the context. If the answer is not clearly
present, say exactly: "I couldn't find this in the AUI catalog."

If the user asks about "majors", list each major exactly as named in
the catalog.

Context:
{context}

Question:
{question}

Provide a concise answer and include "Sources: p.X, p.Y".
"""


def build_rag_chain(retriever):
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
