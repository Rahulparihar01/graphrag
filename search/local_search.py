import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gemini_client import call_gemini
from config import TOP_K


def local_search(
    query: str,
    G: nx.Graph,
    node_embeddings: dict,
    embed_model: SentenceTransformer,
    client,
) -> str:
    """
    Answer a question by finding the most relevant graph nodes via
    semantic similarity and building a focused context prompt.
    Best for specific, fact-based questions.
    """
    if not node_embeddings:
        return "⚠️ No graph data available. The PDF may not have been indexed properly."

    # Find top-K most similar nodes
    query_emb = embed_model.encode([query])[0]
    nodes     = list(node_embeddings.keys())
    sims      = cosine_similarity([query_emb], np.array([node_embeddings[n] for n in nodes]))[0]
    top_nodes = [nodes[i] for i in np.argsort(sims)[::-1][:TOP_K]]

    # Build context from those nodes
    context_parts = []
    for node in top_nodes:
        data      = G.nodes[node]
        edge_info = [
            f"  → {nb}: {', '.join(G.get_edge_data(node, nb, {}).get('relations', ['related'])[:2])}"
            for nb in list(G.neighbors(node))[:5]
        ]
        chunk_txt = (data.get("source_chunks") or [""])[0]
        context_parts.append(
            f"**{node}** ({data.get('type', '?')}): {data.get('description', '')}\n"
            f"Connections:\n" + "\n".join(edge_info) +
            (f"\nContext: {chunk_txt[:300]}" if chunk_txt else "")
        )

    prompt = f"""You are an AI assistant analyzing a document through its knowledge graph.
Below are the most relevant entities and their relationships extracted from the document.

Knowledge Graph Context:
---
{chr(10).join(context_parts)}
---

User Question: {query}

Instructions:
- Answer directly and concisely using ONLY the context above.
- If the answer is clearly present, state it with confidence.
- Mention specific entity names and relationships from the context to support your answer.
- If the context does not contain enough information to answer, say:
  "The document does not seem to cover this topic. Try rephrasing or ask something related to: [list 2-3 relevant entities from the context]."
- Never say the context IS a knowledge graph or explain what a knowledge graph is.
- Never refuse to answer — always try to give something useful."""

    return call_gemini(client, prompt, max_tokens=600)