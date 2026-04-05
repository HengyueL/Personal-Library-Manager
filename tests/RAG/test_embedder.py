import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_model_singleton():
    """Reset the module-level _model singleton between tests."""
    import RAG.embedder as embedder_mod
    original = embedder_mod._model
    embedder_mod._model = None
    yield
    embedder_mod._model = None


class TestEmbed:
    def _make_mock_model(self, n_texts=1, dim=4):
        mock = MagicMock()
        mock.encode.return_value = np.ones((n_texts, dim), dtype=float)
        return mock

    def test_embed_returns_list_of_lists(self):
        mock_model = self._make_mock_model(n_texts=1)
        with patch("RAG.embedder.get_model", return_value=mock_model):
            from RAG.embedder import embed
            result = embed(["hello"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_embed_one_text_gives_one_vector(self):
        mock_model = self._make_mock_model(n_texts=1, dim=4)
        with patch("RAG.embedder.get_model", return_value=mock_model):
            from RAG.embedder import embed
            result = embed(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4

    def test_embed_multiple_texts(self):
        mock_model = self._make_mock_model(n_texts=3, dim=4)
        with patch("RAG.embedder.get_model", return_value=mock_model):
            from RAG.embedder import embed
            result = embed(["a", "b", "c"])
        assert len(result) == 3

    def test_embed_calls_encode_with_normalize(self):
        mock_model = self._make_mock_model(n_texts=1)
        with patch("RAG.embedder.get_model", return_value=mock_model):
            from RAG.embedder import embed
            embed(["test"])
        mock_model.encode.assert_called_once_with(
            ["test"], normalize_embeddings=True, show_progress_bar=False
        )


class TestGetModel:
    def test_get_model_loads_once(self):
        mock_st = MagicMock()
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance

        with patch("RAG.embedder.SentenceTransformer", mock_st, create=True):
            with patch("RAG.embedder._is_cached", return_value=True):
                # Patch the import inside get_model
                with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}):
                    import RAG.embedder as embedder_mod
                    embedder_mod._model = None  # ensure reset
                    m1 = embedder_mod.get_model()
                    m2 = embedder_mod.get_model()
        # Second call must return the cached instance, not create a new one
        assert m1 is m2
