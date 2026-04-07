"""
Unified LLM client adapter.

Select backend via LLM_BACKEND env var (default: huggingface).
  - huggingface: requires HF_TOKEN
  - ollama:      requires OLLAMA_API_KEY
"""

import os

from RAG.config import LLM_BASE_URL, LLM_MODEL_ID, OLLAMA_HOST, OLLAMA_MODEL_ID, BACKEND

_hf_client = None
_ollama_client = None


def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        from openai import OpenAI
        _hf_client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=os.environ["HF_TOKEN"],
        )
    return _hf_client


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        from ollama import Client
        _ollama_client = Client(
            host=OLLAMA_HOST,
            headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]},
        )
    return _ollama_client


def complete(
    messages: list[dict],
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """Send a chat completion request to the configured LLM backend.

    Args:
        messages: List of {'role': ..., 'content': ...} dicts.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Generated text string.
    """

    if BACKEND == "ollama":
        client = _get_ollama_client()
        response = client.chat(
            model=OLLAMA_MODEL_ID,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response.message.content

    # default: huggingface
    client = _get_hf_client()
    completion = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content
