"""
    Starter script for querying PersonalLibrary with RAG.

    Run:
        python main.py
"""

from RAG import query

if __name__ == "__main__":
    user_query = "what is harness design for long-running agent applications?"

    print(f"Query: {user_query}\n")
    result = query(user_query, top_k_docs=5, synthesize=True)

    print("\n=== Sources ===")
    for i, doc in enumerate(result["sources"], start=1):
        print(f"{i}. [{doc['score']:.2f}] {doc['file_name']}")
        if doc["url"]:
            print(f"   {doc['url']}")

    print("\n=== Answer ===")
    print(result["answer"])
