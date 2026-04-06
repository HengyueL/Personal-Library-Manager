"""Gradio web UI for PersonalLibrary."""

import logging
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

CUSTOM_CSS = """
/* ── Font size ─────────────────────────────────────────── */
.gradio-container, .gradio-container * {
    font-size: 16px !important;
}
.gradio-container h1 { font-size: 1.6rem !important; }
.gradio-container h2 { font-size: 1.35rem !important; }
.gradio-container h3 { font-size: 1.15rem !important; }

/* ── Light mode — explicit overrides so OS dark-mode is ignored ── */
:root[data-theme="light"] body,
:root[data-theme="light"] .gradio-container {
    background-color: #f3f4f6 !important;
    color: #111827 !important;
}
:root[data-theme="light"] .block,
:root[data-theme="light"] .panel,
:root[data-theme="light"] .form {
    background-color: #ffffff !important;
    border-color: #e5e7eb !important;
}
:root[data-theme="light"] label,
:root[data-theme="light"] .label-wrap span,
:root[data-theme="light"] p,
:root[data-theme="light"] li,
:root[data-theme="light"] .markdown * {
    color: #111827 !important;
}
:root[data-theme="light"] input,
:root[data-theme="light"] textarea,
:root[data-theme="light"] select {
    background-color: #ffffff !important;
    color: #111827 !important;
    border-color: #d1d5db !important;
}
:root[data-theme="light"] .tabs > .tab-nav > button {
    background-color: #f9fafb !important;
    color: #374151 !important;
    border-color: #e5e7eb !important;
}
:root[data-theme="light"] .tabs > .tab-nav > button.selected {
    background-color: #ffffff !important;
    color: #111827 !important;
}
:root[data-theme="light"] table,
:root[data-theme="light"] th,
:root[data-theme="light"] td {
    background-color: #ffffff !important;
    color: #111827 !important;
    border-color: #e5e7eb !important;
}

/* ── Dark mode — grey palette ───────────────────────────── */
:root[data-theme="dark"] body,
:root[data-theme="dark"] .gradio-container {
    background-color: #2e2e2e !important;
    color: #e0e0e0 !important;
}
:root[data-theme="dark"] .block,
:root[data-theme="dark"] .panel,
:root[data-theme="dark"] .form,
:root[data-theme="dark"] footer {
    background-color: #3a3a3a !important;
    border-color: #505050 !important;
}
:root[data-theme="dark"] label,
:root[data-theme="dark"] .label-wrap span,
:root[data-theme="dark"] p,
:root[data-theme="dark"] li,
:root[data-theme="dark"] .markdown * {
    color: #e0e0e0 !important;
}
:root[data-theme="dark"] input,
:root[data-theme="dark"] textarea,
:root[data-theme="dark"] select {
    background-color: #444444 !important;
    color: #e0e0e0 !important;
    border-color: #606060 !important;
}
:root[data-theme="dark"] .tabs > .tab-nav > button {
    background-color: #3a3a3a !important;
    color: #c8c8c8 !important;
    border-color: #505050 !important;
}
:root[data-theme="dark"] .tabs > .tab-nav > button.selected {
    background-color: #505050 !important;
    color: #ffffff !important;
}
:root[data-theme="dark"] button.primary {
    background-color: #5a7a9a !important;
    color: #ffffff !important;
}
:root[data-theme="dark"] button.stop {
    background-color: #8a4a4a !important;
    color: #ffffff !important;
}
:root[data-theme="dark"] table,
:root[data-theme="dark"] th,
:root[data-theme="dark"] td {
    background-color: #3a3a3a !important;
    color: #e0e0e0 !important;
    border-color: #555555 !important;
}
"""

DARK_MODE_JS = """
() => {
    const stored = localStorage.getItem('plibTheme') || 'light';
    document.documentElement.setAttribute('data-theme', stored);
}
"""


class _ThreadLocalWriter:
    """Stdout proxy that routes writes from one target thread into a queue."""

    def __init__(self, target_thread, q, original):
        self._target = target_thread
        self._q = q
        self._original = original

    def write(self, text):
        if threading.current_thread() is self._target:
            if text:
                self._q.put(text)
            return len(text)
        return self._original.write(text)

    def flush(self):
        self._original.flush()


def _run_with_streaming(fn, *args, **kwargs):
    """Run *fn* in a background thread and yield the growing log string in real-time.

    Captures both sys.stdout and root logger output from the worker thread.
    Each yield emits the full accumulated log so far (suitable for a Gradio
    Textbox that replaces content on each yield).

    Raises whatever exception *fn* raised, after the thread finishes.
    """
    q = queue.Queue()
    exc_holder = [None]

    def worker():
        original_stdout = sys.stdout
        writer = _ThreadLocalWriter(threading.current_thread(), q, original_stdout)
        sys.stdout = writer
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(writer)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root_logger.addHandler(handler)
        try:
            fn(*args, **kwargs)
        except Exception as e:
            exc_holder[0] = e
        finally:
            sys.stdout = original_stdout
            root_logger.removeHandler(handler)
            q.put(None)  # sentinel

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    log = ""
    while True:
        try:
            chunk = q.get(timeout=0.1)
        except queue.Empty:
            if log:
                yield log
            continue
        if chunk is None:
            break
        log += chunk
        yield log

    t.join()
    if exc_holder[0] is not None:
        raise exc_holder[0]


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
        from utils.generate_summary import generate_summary, save_summary
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
        file_name = name if name else derive_file_name(url, content)
        if not name:
            print(f"Auto-generated filename: {file_name}")
        print(f"Generating summary for: {file_name}")
        summary = generate_summary(content)
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


def _doc_summary_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "doc_summary"


def list_documents() -> list[str]:
    return sorted(p.name for p in _doc_summary_dir().glob("*.md"))


def view_document_handler(file_name: str) -> tuple[str, str]:
    """Return (source_url, rendered_markdown) for the selected document."""
    if not file_name:
        return "", ""
    path = _doc_summary_dir() / file_name
    if not path.exists():
        return "", f"_File not found: {file_name}_"
    raw = path.read_text(encoding="utf-8")
    # Strip YAML frontmatter and extract url
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


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PersonalLibraryManager", theme=gr.themes.Soft(), css=CUSTOM_CSS, js=DARK_MODE_JS) as app:
        with gr.Row():
            gr.Markdown("## PersonalLibraryManager\nManage and query your personal document collection.")
            dark_toggle = gr.Checkbox(label="Dark mode", value=False, scale=0, min_width=120)
        dark_toggle.change(
            fn=None,
            inputs=dark_toggle,
            outputs=[],
            js="""(v) => {
                const theme = v ? 'dark' : 'light';
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('plibTheme', theme);
            }""",
        )

        with gr.Tab("Add Document"):
            gr.Markdown(
                "_Fetch a document (HTML or PDF) from a URL, generate an AI summary, "
                "save it to your local library, and register it in the RAG vector index._"
            )
            url_input = gr.Textbox(
                label="🤔 Paste the Document URL you want to add to knowledgebase. ",
                placeholder="https://example.com/article  or  https://arxiv.org/pdf/...",
            )
            name_input = gr.Textbox(
                label="Filename (optional)",
                placeholder="😎 Give a name 'My-Article.md' — You can leave it blank, but you may not like what I generate for you.",
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

        with gr.Tab("Find document."):
            gr.Markdown(
                "_Search your library with a natural-language query. Returns ranked source documents "
                "and an AI-synthesized answer (answer synthesis requires `HF_TOKEN`)._"
            )
            query_input = gr.Textbox(
                label="A text query to find your document 📝.",
                placeholder="What is Anthropic agent harness design?",
                lines=2,
            )
            with gr.Row():
                top_k_slider = gr.Slider(
                    label="Max Sources",
                    minimum=1, maximum=20, value=5, step=1,
                )
                retrieval_only_check = gr.Checkbox(label="ℹ️ Turn this ON if you only want document list without LLM summary.")
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
        
        with gr.Tab("View Document"):
            gr.Markdown(
                "_Browse and read the AI-generated summaries saved in your library._"
            )
            doc_dropdown = gr.Dropdown(
                label="Select document",
                choices=list_documents(),
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
