"""
    Synthesize an LLM answer grounded in retrieved chunks, with inline citations.
"""

from utils.llm_client import complete


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

    return complete(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3,
    )
