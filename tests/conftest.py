import os

# Must be set before any module that does os.environ["HF_TOKEN"] at import time
# (utils/generate_summary.py and RAG/synthesizer.py both do this).
os.environ.setdefault("HF_TOKEN", "dummy-token-for-testing")
