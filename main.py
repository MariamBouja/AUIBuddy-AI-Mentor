"""AUIBuddy — terminal chat loop entry point."""

from dotenv import load_dotenv
load_dotenv()

from rich import print

from vectorstore import get_vectorstore_and_chunks
from retrieval import build_hybrid_retriever, expand_query
from rag_chain import build_rag_chain


def main():
    vectorstore, chunks = get_vectorstore_and_chunks()
    retriever = build_hybrid_retriever(vectorstore, chunks)
    chain = build_rag_chain(retriever)

    print("[bold magenta]🤖 AUIBuddy is ready! Ask a question (type 'exit' to quit)[/bold magenta]\n")
    while True:
        query = input("[You]: ")
        if query.strip().lower() in ("exit", "quit"):
            print("[bold red]Goodbye![/bold red]")
            break

        expanded_query = expand_query(query)
        answer = chain.invoke(expanded_query)
        print(f"[Bot]: {answer}\n")


if __name__ == "__main__":
    main()
