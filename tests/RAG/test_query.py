from unittest.mock import MagicMock, patch

import pytest


def _mock_retriever_result(file_name="doc.md", score=0.9):
    return {
        "ranked_docs": [{"file_name": file_name, "url": "https://x.com", "score": score, "content_type": "raw_chunk"}],
        "chunks": [{"file_name": file_name, "text": "chunk text", "score": score, "content_type": "raw_chunk", "chunk_index": 0}],
    }


class TestQuery:
    def _run_query(self, retriever_result, answer="synthesized answer", synthesize=True):
        with patch("RAG.query._ensure_index_built"):
            with patch("RAG.query.retriever.retrieve", return_value=retriever_result):
                with patch("RAG.query.synthesize_answer", return_value=answer, create=True):
                    with patch("RAG.synthesizer.synthesize_answer", return_value=answer):
                        from RAG.query import query
                        return query("test question", synthesize=synthesize)

    def test_returns_required_keys(self):
        result = self._run_query(_mock_retriever_result())
        assert "query" in result
        assert "sources" in result
        assert "answer" in result

    def test_query_field_matches_input(self):
        with patch("RAG.query._ensure_index_built"):
            with patch("RAG.query.retriever.retrieve", return_value=_mock_retriever_result()):
                with patch("RAG.synthesizer.synthesize_answer", return_value="ans"):
                    from RAG.query import query
                    result = query("my specific question")
        assert result["query"] == "my specific question"

    def test_sources_come_from_retriever(self):
        retriever_result = _mock_retriever_result(file_name="paper.md", score=0.85)
        result = self._run_query(retriever_result)
        assert result["sources"][0]["file_name"] == "paper.md"
        assert result["sources"][0]["score"] == 0.85

    def test_no_synthesis_returns_empty_answer(self):
        with patch("RAG.query._ensure_index_built"):
            with patch("RAG.query.retriever.retrieve", return_value=_mock_retriever_result()):
                from RAG.query import query
                result = query("question", synthesize=False)
        assert result["answer"] == ""

    def test_synthesize_true_calls_synthesizer(self):
        mock_syn = MagicMock(return_value="the answer")
        with patch("RAG.query._ensure_index_built"):
            with patch("RAG.query.retriever.retrieve", return_value=_mock_retriever_result()):
                with patch("RAG.synthesizer.synthesize_answer", mock_syn):
                    from RAG.query import query
                    result = query("q", synthesize=True)
        mock_syn.assert_called_once()
        assert result["answer"] == "the answer"
