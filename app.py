from rich import print
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

local_llm = ChatOllama(model="llama3.1")

#RAG
def rag(chunks, collection_name):
    vectorstore = Chroma.from_documents(
        documents=Document,
        collection_name=collection_name,
        embedding=embeddings.ollam.OllamaEmbeddings(model='llama3.1'),
    )
retriever = vectorstores.as_retriever()

prompt_template = """"Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt : ChatPromptTemplate.from_template(prompt_template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | local_llm
    | StrOutputParser()
    
)
result = chain.invoke("What is the use of Text Splitting?")
print(result)

# 1. Character Text Splitting
print("### Character Text Splitting ###")

text = "Text splitting in Langchain is a criticial feature that facilitates the division"

# Manual Splitting
chunks = []
chunks_size = 35 #Characters
for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)
documents = [Document(page_content=chunk, metadate={"source": "local"}) ]
print(documents)

# Automatic Texy Splitting