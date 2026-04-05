"""
    Starter script for querying PersonalLibrary with RAG.

    Usage:
        python retrieve_document.py --query "your question here"
        python retrieve_document.py --query "your question here" --top-k 3 --retrieval-only
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    parser.add_argument("--retrieval-only", action="store_true", help="Return ranked sources only; skip LLM answer synthesis")
    args = parser.parse_args()

    logger.info("Query: %s", args.query)
    result = query(args.query, top_k_docs=args.top_k, synthesize=not args.retrieval_only)
    if result["answer"]:
        print(f"\n{YELLOW}=== Answer ==={RESET}")
        print(f"\n{YELLOW}{result['answer']}{RESET}")

    print(f"\n{GREEN}=== References ==={RESET}")
    for i, doc in enumerate(result["sources"], start=1):
        print(f"{GREEN}{i}. [{doc['score']:.2f}] {doc['file_name']}{RESET} \n")


if __name__ == "__main__":
    main()
