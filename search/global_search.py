from gemini_client import call_gemini


def global_search(query: str, community_summaries: dict, client) -> str:
    """
    Answer a question using high-level community summaries of the entire document.
    Best for broad, thematic questions.
    """
    summaries_text = "\n\n".join([
        f"**Group {i+1}**: {data['summary']}\n  Key entities: {', '.join(data['nodes'][:8])}"
        for i, (_, data) in enumerate(community_summaries.items())
    ])

    prompt = f"""Based on these high-level summaries of the document's knowledge groups, answer the question.

Document Knowledge Groups:
{summaries_text}

Question: {query}

Provide a comprehensive answer that synthesizes information across groups."""

    return call_gemini(client, prompt, max_tokens=700)
