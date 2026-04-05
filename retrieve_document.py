"""
    Starter script for querying PersonalLibrary with RAG.

    Usage:
        python retrieve_document.py "your question here"
        python retrieve_document.py "your question here" --top-k 3 --no-answer
"""

import argparse
import logging

from RAG import query

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Query the PersonalLibrary RAG system.")
    parser.add_argument("--query", type=str, required=True, help="Natural language query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of source documents to retrieve (default: 5)")
    parser.add_argument("--no-answer", action="store_true", help="Skip LLM answer synthesis, retrieval only")
    args = parser.parse_args()

    logger.info("Query: %s", args.query)
    result = query(args.query, top_k_docs=args.top_k, synthesize=not args.no_answer)
    if result["answer"]:
        print(f"\n{YELLOW}=== References ==={RESET}")
        print(f"\n{YELLOW}{result['answer']}{RESET}")

    print(f"\n{GREEN}=== References ==={RESET}")
    for i, doc in enumerate(result["sources"], start=1):
        print(f"{GREEN}{i}. [{doc['score']:.2f}] {doc['file_name']}{RESET} \n")


if __name__ == "__main__":
    main()
