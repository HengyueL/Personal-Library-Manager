from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_client_singleton():
    import RAG.synthesizer as syn_mod
    syn_mod._client = None
    yield
    syn_mod._client = None


def _make_mock_client(answer_text: str) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = answer_text
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


class TestSynthesizeAnswer:
    def test_empty_chunks_returns_no_context_message(self):
        from RAG.synthesizer import synthesize_answer
        result = synthesize_answer("What is X?", [])
        assert "No relevant context" in result

    def test_returns_llm_response(self):
        mock_client = _make_mock_client("The answer is 42.")
        with patch("RAG.synthesizer._get_client", return_value=mock_client):
            from RAG.synthesizer import synthesize_answer
            result = synthesize_answer("What is X?", [{"file_name": "doc.md", "text": "X is 42."}])
        assert result == "The answer is 42."

    def test_prompt_includes_query_and_chunk_text(self):
        mock_client = _make_mock_client("ok")
        with patch("RAG.synthesizer._get_client", return_value=mock_client):
            from RAG.synthesizer import synthesize_answer
            synthesize_answer("My query", [{"file_name": "src.md", "text": "Relevant content."}])

        call_kwargs = mock_client.chat.completions.create.call_args
        prompt = call_kwargs[1]["messages"][0]["content"]
        assert "My query" in prompt
        assert "Relevant content." in prompt
        assert "src.md" in prompt

    def test_prompt_lists_allowed_citations(self):
        mock_client = _make_mock_client("ok")
        chunks = [
            {"file_name": "a.md", "text": "text a"},
            {"file_name": "b.md", "text": "text b"},
        ]
        with patch("RAG.synthesizer._get_client", return_value=mock_client):
            from RAG.synthesizer import synthesize_answer
            synthesize_answer("q", chunks)

        prompt = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
        assert "a.md" in prompt
        assert "b.md" in prompt

    def test_llm_called_with_correct_params(self):
        mock_client = _make_mock_client("answer")
        with patch("RAG.synthesizer._get_client", return_value=mock_client):
            from RAG.synthesizer import synthesize_answer
            synthesize_answer("q", [{"file_name": "f.md", "text": "t"}])

        create_call = mock_client.chat.completions.create.call_args[1]
        assert create_call["temperature"] == 0.3
        assert create_call["max_tokens"] == 800
