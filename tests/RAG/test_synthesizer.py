from unittest.mock import MagicMock, patch

import pytest


class TestSynthesizeAnswer:
    def test_empty_chunks_returns_no_context_message(self):
        from RAG.synthesizer import synthesize_answer
        result = synthesize_answer("What is X?", [])
        assert "No relevant context" in result

    def test_returns_llm_response(self):
        with patch("RAG.synthesizer.complete", return_value="The answer is 42."):
            from RAG.synthesizer import synthesize_answer
            result = synthesize_answer("What is X?", [{"file_name": "doc.md", "text": "X is 42."}])
        assert result == "The answer is 42."

    def test_prompt_includes_query_and_chunk_text(self):
        mock_complete = MagicMock(return_value="ok")
        with patch("RAG.synthesizer.complete", mock_complete):
            from RAG.synthesizer import synthesize_answer
            synthesize_answer("My query", [{"file_name": "src.md", "text": "Relevant content."}])

        messages = mock_complete.call_args[1]["messages"]
        prompt = messages[0]["content"]
        assert "My query" in prompt
        assert "Relevant content." in prompt
        assert "src.md" in prompt

    def test_prompt_lists_allowed_citations(self):
        mock_complete = MagicMock(return_value="ok")
        chunks = [
            {"file_name": "a.md", "text": "text a"},
            {"file_name": "b.md", "text": "text b"},
        ]
        with patch("RAG.synthesizer.complete", mock_complete):
            from RAG.synthesizer import synthesize_answer
            synthesize_answer("q", chunks)

        prompt = mock_complete.call_args[1]["messages"][0]["content"]
        assert "a.md" in prompt
        assert "b.md" in prompt

    def test_llm_called_with_correct_params(self):
        mock_complete = MagicMock(return_value="answer")
        with patch("RAG.synthesizer.complete", mock_complete):
            from RAG.synthesizer import synthesize_answer
            synthesize_answer("q", [{"file_name": "f.md", "text": "t"}])

        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 800
