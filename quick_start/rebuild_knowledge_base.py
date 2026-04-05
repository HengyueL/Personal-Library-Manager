"""
Rebuild the PersonalLibrary RAG index from scratch.

Usage:
    python quick_start/rebuild_knowledge_base.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.index import build_index


def main():
    build_index(rebuild=True)


if __name__ == "__main__":
    main()
