# PersonalLibrary
A private project to effectively manage the paper/projects that I am interested in.

## Design

1. @doc_raw folder store all raw documents. (Assume all of them are fetched by @utils.fetch_html)
2. @doc_summary folder stores the summary/abstract of a corresponding raw document (with the same name) in @doc_raw, by @utils.generate_summary.py
3. @RAG implements a RAG system, so that I can quickly find a list of source information stored in @doc_raw and @doc_summary for me to deep dive.