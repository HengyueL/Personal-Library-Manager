"""
plib — PersonalLibrary command-line interface.

Usage:
    plib add --url <url> [--name <filename>]
    plib query --query "..." [--top-k N] [--retrieval-only]
    plib rebuild [--incremental]
    plib gui [--port N] [--share]
"""

import argparse
import sys


def _run_add(args) -> None:
    argv = ["add_document", "--url", args.url]
    if args.name:
        argv += ["--name", args.name]
    sys.argv = argv
    from quick_start.add_document import main as add_main
    add_main()


def _run_query(args) -> None:
    argv = ["retrieve_document", "--query", args.query, "--top-k", str(args.top_k)]
    if args.retrieval_only:
        argv.append("--retrieval-only")
    sys.argv = argv
    from quick_start.retrieve_document import main as query_main
    query_main()


def _run_rebuild(args) -> None:
    from RAG.index import build_index
    build_index(rebuild=not args.incremental)


def _run_gui(args) -> None:
    from quick_start.gui import build_app
    build_app().launch(server_port=args.port, share=args.share)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plib",
        description="PersonalLibrary: manage and query your document library.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # --- plib add ---
    add_parser = subparsers.add_parser("add", help="Fetch a document and add it to the library.")
    add_parser.add_argument("--url", required=True, help="URL of the document to fetch (HTML or PDF).")
    add_parser.add_argument(
        "--name",
        default=None,
        metavar="FILENAME",
        help="Output filename (e.g. My_Article.md). Auto-generated if omitted.",
    )

    # --- plib query ---
    query_parser = subparsers.add_parser("query", help="Query the library with natural language.")
    query_parser.add_argument("--query", required=True, help="Your natural-language question.")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of source documents to retrieve (default: 5).",
    )
    query_parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Return ranked sources only; skip LLM answer synthesis.",
    )

    # --- plib rebuild ---
    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild the RAG vector index.")
    rebuild_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only index new documents; skip already-indexed files. Default: full rebuild.",
    )

    # --- plib gui ---
    gui_parser = subparsers.add_parser("gui", help="Launch the Gradio web UI.")
    gui_parser.add_argument("--port", type=int, default=7860, help="Port to serve the UI on (default: 7860).")
    gui_parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")

    args = parser.parse_args()

    if args.command == "add":
        _run_add(args)
    elif args.command == "query":
        _run_query(args)
    elif args.command == "rebuild":
        _run_rebuild(args)
    elif args.command == "gui":
        _run_gui(args)


if __name__ == "__main__":
    main()
