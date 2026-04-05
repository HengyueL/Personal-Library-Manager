from unittest.mock import MagicMock, patch

import pytest


def _make_chroma_result(ids, documents, metadatas, distances):
    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def _mock_collection(chroma_result, count=10):
    col = MagicMock()
    col.query.return_value = chroma_result
    col.count.return_value = count
    return col


class TestRetrieve:
    def _run_retrieve(self, collection, query_vec=None):
        if query_vec is None:
            query_vec = [[0.1, 0.2]]

        with patch("RAG.retriever._get_collection", return_value=collection):
            with patch("RAG.retriever.embedder.embed", return_value=query_vec):
                from RAG.retriever import retrieve
                return retrieve("test query")

    def test_returns_ranked_docs_and_chunks_keys(self):
        result_data = _make_chroma_result(
            ids=["raw::doc.md::0"],
            documents=["chunk text"],
            metadatas=[{"source": "doc.md", "content_type": "raw_chunk", "original_url": "http://x.com", "chunk_index": 0}],
            distances=[0.2],
        )
        result = self._run_retrieve(_mock_collection(result_data))
        assert "ranked_docs" in result
        assert "chunks" in result

    def test_cosine_score_computed_correctly(self):
        # distance 0.2 → score = 1 - 0.2 = 0.8
        result_data = _make_chroma_result(
            ids=["raw::doc.md::0"],
            documents=["text"],
            metadatas=[{"source": "doc.md", "content_type": "raw_chunk", "original_url": "", "chunk_index": 0}],
            distances=[0.2],
        )
        result = self._run_retrieve(_mock_collection(result_data))
        assert abs(result["ranked_docs"][0]["score"] - 0.8) < 1e-6

    def test_distance_above_1_clamped_to_0(self):
        result_data = _make_chroma_result(
            ids=["raw::doc.md::0"],
            documents=["text"],
            metadatas=[{"source": "doc.md", "content_type": "raw_chunk", "original_url": "", "chunk_index": 0}],
            distances=[1.5],
        )
        result = self._run_retrieve(_mock_collection(result_data))
        assert result["ranked_docs"][0]["score"] == 0.0

    def test_deduplication_keeps_best_score_per_file(self):
        result_data = _make_chroma_result(
            ids=["raw::doc.md::0", "raw::doc.md::1"],
            documents=["chunk a", "chunk b"],
            metadatas=[
                {"source": "doc.md", "content_type": "raw_chunk", "original_url": "", "chunk_index": 0},
                {"source": "doc.md", "content_type": "raw_chunk", "original_url": "", "chunk_index": 1},
            ],
            distances=[0.3, 0.1],  # chunk 1 is better (lower distance)
        )
        result = self._run_retrieve(_mock_collection(result_data))
        assert len(result["ranked_docs"]) == 1
        assert abs(result["ranked_docs"][0]["score"] - 0.9) < 1e-6  # 1 - 0.1

    def test_summaries_excluded_from_chunks(self):
        result_data = _make_chroma_result(
            ids=["summary::doc.md", "raw::doc.md::0"],
            documents=["summary text", "raw text"],
            metadatas=[
                {"source": "doc.md", "content_type": "summary", "original_url": ""},
                {"source": "doc.md", "content_type": "raw_chunk", "original_url": "", "chunk_index": 0},
            ],
            distances=[0.1, 0.2],
        )
        result = self._run_retrieve(_mock_collection(result_data))
        assert all(c["content_type"] == "raw_chunk" for c in result["chunks"])

    def test_ranked_docs_sorted_by_score_descending(self):
        result_data = _make_chroma_result(
            ids=["raw::a.md::0", "raw::b.md::0"],
            documents=["text a", "text b"],
            metadatas=[
                {"source": "a.md", "content_type": "raw_chunk", "original_url": "", "chunk_index": 0},
                {"source": "b.md", "content_type": "raw_chunk", "original_url": "", "chunk_index": 0},
            ],
            distances=[0.4, 0.1],  # b.md has better score
        )
        result = self._run_retrieve(_mock_collection(result_data))
        scores = [d["score"] for d in result["ranked_docs"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_docs_limits_ranked_docs(self):
        n = 10
        result_data = _make_chroma_result(
            ids=[f"raw::doc{i}.md::0" for i in range(n)],
            documents=[f"text {i}" for i in range(n)],
            metadatas=[
                {"source": f"doc{i}.md", "content_type": "raw_chunk", "original_url": "", "chunk_index": 0}
                for i in range(n)
            ],
            distances=[0.1 * i for i in range(n)],
        )
        with patch("RAG.retriever._get_collection", return_value=_mock_collection(result_data, count=n)):
            with patch("RAG.retriever.embedder.embed", return_value=[[0.1]]):
                from RAG.retriever import retrieve
                result = retrieve("q", top_k_chunks=n, top_k_docs=3)
        assert len(result["ranked_docs"]) <= 3
