import re


def strip_frontmatter(text: str) -> tuple[str, dict]:
    """
    Strip YAML frontmatter from markdown text.

    Returns:
        (body, frontmatter_dict) where frontmatter_dict contains parsed key-value pairs.
        If no frontmatter is found, returns (text, {}).
    """
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if not match:
        return text, {}

    frontmatter_block = match.group(1)
    body = text[match.end():]

    frontmatter = {}
    for line in frontmatter_block.splitlines():
        if ':' in line:
            key, _, value = line.partition(':')
            frontmatter[key.strip()] = value.strip()

    return body, frontmatter


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Split text into chunks using paragraph-aware greedy packing.

    Splits on double newlines (paragraph boundaries), then greedily packs
    paragraphs into chunks up to `size` characters, carrying `overlap`
    characters from the previous chunk into the next.

    Args:
        text: The text to chunk.
        size: Maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

    if not paragraphs:
        return []

    chunks = []
    current_parts = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph exceeds chunk size, split it by characters
        if para_len > size:
            # Flush current buffer first
            if current_parts:
                chunks.append('\n\n'.join(current_parts))
                current_parts = []
                current_len = 0

            start = 0
            while start < para_len:
                end = min(start + size, para_len)
                chunks.append(para[start:end])
                start = end - overlap if end < para_len else para_len
            continue

        # Would adding this paragraph exceed the limit?
        separator_len = 2 if current_parts else 0  # len('\n\n')
        if current_len + separator_len + para_len > size and current_parts:
            chunks.append('\n\n'.join(current_parts))
            # Start next chunk with overlap from end of last chunk
            overlap_text = chunks[-1][-overlap:] if overlap > 0 else ''
            current_parts = [overlap_text, para] if overlap_text else [para]
            current_len = len(overlap_text) + 2 + para_len if overlap_text else para_len
        else:
            current_parts.append(para)
            current_len += separator_len + para_len

    if current_parts:
        chunks.append('\n\n'.join(current_parts))

    return chunks
