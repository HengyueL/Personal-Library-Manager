"""Gradio web UI for PersonalLibrary."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root  → utils/, RAG/
sys.path.insert(0, str(_HERE))         # quick_start/ → gui_utils/

import gradio as gr

from gui_utils.documents import _doc_summary_dir, list_documents
from gui_utils.streaming import _run_with_streaming

CUSTOM_CSS = """
.gradio-container, .gradio-container * {
    font-size: 16px !important;
}
.gradio-container h1 { font-size: 1.6rem !important; }
.gradio-container h2 { font-size: 1.35rem !important; }
.gradio-container h3 { font-size: 1.15rem !important; }
"""


# ── Handlers ──────────────────────────────────────────────────────────────────

def add_document_handler(url: str, name: str, cookies_path: str):
    url = url.strip()
    name = name.strip() if name else None
    cookies_path = cookies_path.strip() if cookies_path else None

    if not url:
        yield "ERROR: Please enter a document URL."
        return

    try:
        from utils.fetch_document import AuthRequiredError, fetch_document
    except Exception as e:
        yield f"ERROR importing fetch_document: {e}"
        return

    try:
        from utils.generate_summary import generate_summary, generate_summary_with_filename, save_summary
    except KeyError:
        yield (
            "ERROR: HF_TOKEN environment variable is not set.\n"
            "Export it before launching the GUI:\n"
            "  export HF_TOKEN=<your_huggingface_token>"
        )
        return
    except Exception as e:
        yield f"ERROR importing generate_summary: {e}"
        return

    from utils.file_naming import derive_file_name
    from RAG.index import build_index

    def _work():
        print(f"Fetching: {url}")
        try:
            content = fetch_document(url=url, cookies_path=cookies_path)
        except AuthRequiredError as e:
            print(str(e))
            return
        if name:
            file_name = name
            print(f"Generating summary for: {file_name}")
            summary = generate_summary(content)
        else:
            print("Generating summary and filename via LLM...")
            summary, file_name = generate_summary_with_filename(content, url)
            print(f"LLM-proposed filename: {file_name}")
        save_summary(file_name=file_name, summary_text=summary, url=url)
        print("Updating RAG index...")
        build_index(rebuild=False)
        print(f"\nDone. '{file_name}' is now searchable via RAG.")

    last = ""
    try:
        for chunk in _run_with_streaming(_work):
            last = chunk
            yield chunk
    except Exception as e:
        yield last + f"\nERROR: {e}"


def view_document_handler(file_name: str) -> tuple[str, str]:
    """Return (source_url, rendered_markdown) for the selected document."""
    if not file_name:
        return "", ""
    path = _doc_summary_dir() / file_name
    if not path.exists():
        return "", f"_File not found: {file_name}_"
    raw = path.read_text(encoding="utf-8")
    url = ""
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            frontmatter = raw[3:end].strip()
            for line in frontmatter.splitlines():
                if line.startswith("url:"):
                    url = line[4:].strip()
            raw = raw[end + 3:].strip()
    return url, raw


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


def rebuild_handler(incremental: bool):
    from RAG.index import build_index

    def _work():
        build_index(rebuild=not incremental)
        print("\nDone.")

    last = ""
    try:
        for chunk in _run_with_streaming(_work):
            last = chunk
            yield chunk
    except Exception as e:
        yield last + f"\nERROR: {e}"


# ── UI ────────────────────────────────────────────────────────────────────────

def on_source_select(evt: gr.SelectData):
    """Navigate to View Document tab when a Document cell is clicked in Find Document."""
    if evt.index[1] != 1:
        return gr.update(), gr.update(), "", ""
    file_name = str(evt.value)
    url, content = view_document_handler(file_name)
    return gr.update(selected="view_tab"), gr.update(value=file_name), url, content


def build_app() -> gr.Blocks:
    # Pre-load the embedding model so the first search isn't slow.
    try:
        from RAG.embedder import get_model
        get_model()
    except Exception as e:
        print(f"Warning: could not pre-load embedding model: {e}")

    with gr.Blocks(title="PersonalLibraryManager", theme=gr.themes.Soft(), css=CUSTOM_CSS) as app:
        gr.Markdown("## PersonalLibraryManager\nManage and query your personal document collection.")

        with gr.Tabs() as tabs:
            with gr.Tab("Add Document"):
                gr.Markdown(
                    "_Fetch a document (HTML or PDF) from a URL, generate an AI summary, "
                    "save it to your local library, and register it in the RAG vector index._"
                )
                url_input = gr.Textbox(
                    label="Document URL",
                    placeholder="https://example.com/article  or  https://arxiv.org/pdf/...",
                )
                name_input = gr.Textbox(
                    label="Filename (optional)",
                    placeholder="My-Article.md — leave blank to auto-generate",
                )
                cookies_input = gr.Textbox(
                    label="Cookies file (optional — for login-gated URLs)",
                    placeholder="/path/to/cookies.txt — exported via 'Get cookies.txt LOCALLY'",
                )
                add_btn = gr.Button("Add to Library", variant="primary")
                add_output = gr.Textbox(label="Status", interactive=False, lines=8)
                add_btn.click(
                    fn=add_document_handler,
                    inputs=[url_input, name_input, cookies_input],
                    outputs=add_output,
                )

            with gr.Tab("Find Document"):
                gr.Markdown(
                    "_Search your library with a natural-language query. Returns ranked source documents "
                    "and an AI-synthesized answer (answer synthesis requires `HF_TOKEN`)._"
                )
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="What is Anthropic agent harness design?",
                    lines=2,
                )
                with gr.Row():
                    top_k_slider = gr.Slider(
                        label="Max Sources",
                        minimum=1, maximum=20, value=5, step=1,
                    )
                    retrieval_only_check = gr.Checkbox(label="Retrieval only (no LLM summary)")
                query_btn = gr.Button("Search", variant="primary")
                answer_md = gr.Markdown(label="Answer")
                sources_df = gr.Dataframe(
                    headers=["Rank", "Document", "Score", "URL"],
                    label="Sources — click a Document name to open it in View Document",
                    interactive=False,
                )
                query_btn.click(
                    fn=query_handler,
                    inputs=[query_input, top_k_slider, retrieval_only_check],
                    outputs=[answer_md, sources_df],
                )

            with gr.Tab("View Document", id="view_tab"):
                gr.Markdown("_Browse and read the AI-generated summaries saved in your library._")
                doc_dropdown = gr.Dropdown(
                    label="Select document",
                    choices=list_documents(),
                    value=None,
                    interactive=True,
                )
                refresh_btn = gr.Button("Refresh list")
                source_url = gr.Textbox(label="Source URL", interactive=False)
                doc_md = gr.Markdown()
                refresh_btn.click(
                    fn=lambda: gr.update(choices=list_documents()),
                    inputs=[],
                    outputs=doc_dropdown,
                )
                doc_dropdown.change(
                    fn=view_document_handler,
                    inputs=doc_dropdown,
                    outputs=[source_url, doc_md],
                )

            with gr.Tab("Rebuild Index"):
                gr.Markdown(
                    "_Rebuild the RAG vector index from all summaries in `doc_summary/`. "
                    "Use **Incremental** mode to add only new documents without wiping the existing index._\n\n"
                    "Click **Rebuild Index** to wipe the existing vector index and re-embed all documents."
                )
                incremental_check = gr.Checkbox(label="Incremental (skip already-indexed documents)")
                rebuild_btn = gr.Button("Rebuild Index", variant="stop")
                rebuild_output = gr.Textbox(label="Output", interactive=False, lines=8)
                rebuild_btn.click(
                    fn=rebuild_handler,
                    inputs=incremental_check,
                    outputs=rebuild_output,
                )

        sources_df.select(
            fn=on_source_select,
            inputs=[],
            outputs=[tabs, doc_dropdown, source_url, doc_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    try:
        app.launch()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        print("\nShutdown complete.")
