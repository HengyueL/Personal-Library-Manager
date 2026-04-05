import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.index import build_index


def main():
    parser = argparse.ArgumentParser(description="Build the PersonalLibrary RAG index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the existing index and rebuild from scratch.",
    )
    args = parser.parse_args()
    build_index(rebuild=args.rebuild)


if __name__ == "__main__":
    main()

