"""Gradio web UI for PersonalLibrary."""

import contextlib
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr


@contextlib.contextmanager
def _capture_output():
    """Redirect both stdout and logging output to a StringIO buffer."""
    buf = io.StringIO()
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(handler)
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old_stdout
        root_logger.removeHandler(handler)


def add_document_handler(url: str, name: str) -> str:
    url = url.strip()
    name = name.strip() if name else None

    if not url:
        return "ERROR: Please enter a document URL."

    try:
        from utils.fetch_document import fetch_document
    except Exception as e:
        return f"ERROR importing fetch_document: {e}"

    try:
        from utils.generate_summary import generate_summary, save_summary
    except KeyError:
        return (
            "ERROR: HF_TOKEN environment variable is not set.\n"
            "Export it before launching the GUI:\n"
            "  export HF_TOKEN=<your_huggingface_token>"
        )
    except Exception as e:
        return f"ERROR importing generate_summary: {e}"

    from utils.file_naming import derive_file_name
    from RAG.index import build_index

    with _capture_output() as buf:
        try:
            print(f"Fetching: {url}")
            content = fetch_document(url=url)

            if name:
                file_name = name
            else:
                file_name = derive_file_name(url, content)
                print(f"Auto-generated filename: {file_name}")

            print(f"Generating summary for: {file_name}")
            summary = generate_summary(content)
            save_summary(file_name=file_name, summary_text=summary, url=url)

            print("Updating RAG index...")
            build_index(rebuild=False)

            print(f"\nDone. '{file_name}' is now searchable via RAG.")
        except Exception as e:
            print(f"\nERROR: {e}")

    return buf.getvalue()


def query_handler(user_query: str, top_k: int, retrieval_only: bool):
    user_query = user_query.strip()
    if not user_query:
        return "Please enter a question.", []

    try:
        from RAG import query as rag_query
    except Exception as e:
        return f"ERROR importing RAG: {e}", []

    try:
        result = rag_query(user_query, top_k_docs=int(top_k), synthesize=not retrieval_only)
    except Exception as e:
        return f"ERROR: {e}", []

    answer = result.get("answer") or "_No answer synthesized (retrieval-only mode or no HF_TOKEN)._"
    sources = result.get("sources", [])
    rows = [
        [i, doc["file_name"], f"{doc['score']:.2f}", doc.get("url", "")]
        for i, doc in enumerate(sources, 1)
    ]
    return answer, rows


def rebuild_handler(incremental: bool) -> str:
    from RAG.index import build_index

    with _capture_output() as buf:
        try:
            build_index(rebuild=not incremental)
            print("\nDone.")
        except Exception as e:
            print(f"\nERROR: {e}")

    return buf.getvalue()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PersonalLibrary", theme=gr.themes.Soft()) as app:
        gr.Markdown("## PersonalLibrary\nManage and query your personal document collection.")

        with gr.Tab("Add Document"):
            url_input = gr.Textbox(
                label="Document URL",
                placeholder="https://example.com/article  or  https://arxiv.org/pdf/...",
            )
            name_input = gr.Textbox(
                label="Filename (optional)",
                placeholder="My-Article.md — leave blank to auto-generate from title",
            )
            add_btn = gr.Button("Add to Library", variant="primary")
            add_output = gr.Textbox(label="Status", interactive=False, lines=8)
            add_btn.click(
                fn=add_document_handler,
                inputs=[url_input, name_input],
                outputs=add_output,
            )

        with gr.Tab("Query Library"):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="What does X say about Y?",
                lines=2,
            )
            with gr.Row():
                top_k_slider = gr.Slider(
                    label="Max Sources",
                    minimum=1, maximum=20, value=5, step=1,
                )
                retrieval_only_check = gr.Checkbox(label="Retrieval only (skip LLM answer)")
            query_btn = gr.Button("Search", variant="primary")
            answer_md = gr.Markdown(label="Answer")
            sources_df = gr.Dataframe(
                headers=["Rank", "Document", "Score", "URL"],
                label="Sources",
                interactive=False,
            )
            query_btn.click(
                fn=query_handler,
                inputs=[query_input, top_k_slider, retrieval_only_check],
                outputs=[answer_md, sources_df],
            )

        with gr.Tab("Rebuild Index"):
            gr.Markdown(
                "**Wipes the existing vector index and re-embeds all documents** in `doc_summary/`.\n\n"
                "Use *Incremental* mode to only add newly-added documents without wiping existing data."
            )
            incremental_check = gr.Checkbox(label="Incremental (skip already-indexed documents)")
            rebuild_btn = gr.Button("Rebuild Index", variant="stop")
            rebuild_output = gr.Textbox(label="Output", interactive=False, lines=8)
            rebuild_btn.click(
                fn=rebuild_handler,
                inputs=incremental_check,
                outputs=rebuild_output,
            )

    return app


if __name__ == "__main__":
    build_app().launch()
