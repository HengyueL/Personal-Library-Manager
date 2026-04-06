"""Background-thread streaming helpers for the Gradio GUI."""

import logging
import queue
import sys
import threading


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

    def isatty(self):
        return False

    def fileno(self):
        return self._original.fileno()


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
