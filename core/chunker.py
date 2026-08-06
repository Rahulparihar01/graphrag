from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str) -> list[dict]:
    """Split text into overlapping chunks with sentence-boundary awareness."""
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_content = text[start:end]

        # Try to end on a sentence boundary
        last_period = chunk_content.rfind('. ')
        if last_period > CHUNK_SIZE // 2:
            end = start + last_period + 1
            chunk_content = text[start:end]

        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk_content.strip(),
            "start_char": start,
        })

        chunk_id += 1
        start = end - CHUNK_OVERLAP

    print(f"✅ Created {len(chunks)} text chunks")
    return chunks
