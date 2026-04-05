"""
    Synthesize an LLM answer grounded in retrieved chunks, with inline citations.
    Reuses the HuggingFace OpenAI-compatible client pattern from utils/generate_summary.py.
"""

import os
from openai import OpenAI

from RAG.config import LLM_MODEL_ID, LLM_BASE_URL

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=os.environ["HF_TOKEN"],
        )
    return _client


def synthesize_answer(query: str, chunks: list[dict]) -> str:
    """
    Generate an LLM answer grounded only in the provided chunks.

    Args:
        query: The user's question.
        chunks: List of dicts with keys 'text' and 'file_name'.

    Returns:
        Synthesized answer string with inline [file_name] citations.
    """
    if not chunks:
        return "No relevant context found to answer this question."

    # Build numbered context blocks labeled by file_name
    context_blocks = []
    allowed_citations = set()
    for i, chunk in enumerate(chunks, start=1):
        file_name = chunk["file_name"]
        allowed_citations.add(file_name)
        context_blocks.append(f"[{i}] Source: {file_name}\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)
    allowed_str = ", ".join(sorted(allowed_citations))

    prompt = f"""You are a research assistant. Answer the question below using ONLY the provided context.
After each claim, cite the source using its file name in brackets, e.g. [Anthropic_Harness_Design.md].
Allowed citation tokens: {allowed_str}
If the answer is not in the context, say "I don't have enough information in the provided sources to answer this."

Question: {query}

Context:
{context_str}

Answer:"""

    client = _get_client()
    completion = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )
    return completion.choices[0].message.content
