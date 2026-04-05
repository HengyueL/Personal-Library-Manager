"""Gradio web UI for PersonalLibrary."""

import logging
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr


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
