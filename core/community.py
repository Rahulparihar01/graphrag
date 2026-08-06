import networkx as nx
from gemini_client import call_gemini


def detect_communities(G: nx.Graph) -> dict:
    """Detect communities in the graph using greedy modularity."""
    if G.number_of_nodes() == 0:
        return {}
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        raw = list(greedy_modularity_communities(G))
    except Exception:
        raw = [list(G.nodes())]

    communities = {f"community_{i}": list(c) for i, c in enumerate(raw)}
    print(f"✅ Detected {len(communities)} communities")
    return communities


def summarize_communities(communities: dict, G: nx.Graph, client) -> dict:
    """Generate a natural-language summary for each community using Gemini."""
    print(f"📝 Summarizing {len(communities)} communities...")
    summaries = {}

    for comm_id, nodes in communities.items():
        node_info = []
        for node in nodes[:15]:
            node_data  = G.nodes[node]
            neighbors  = list(G.neighbors(node))[:5]
            relations  = [
                f"{node} --[{', '.join(G.get_edge_data(node, nb, {}).get('relations', ['related'])[:2])}]--> {nb}"
                for nb in neighbors
            ]
            node_info.append(
                f"Entity: {node} ({node_data.get('type', '?')})\n"
                f"  Relations: {'; '.join(relations[:3])}"
            )

        prompt = f"""Summarize what this group of related entities is about in 2-3 sentences.

Entities and their relationships:
{chr(10).join(node_info)}

Write a clear, concise summary that captures the main theme of this group."""

        try:
            summary = call_gemini(client, prompt, max_tokens=200)
        except Exception:
            summary = f"Group of {len(nodes)} related entities: {', '.join(nodes[:5])}"

        summaries[comm_id] = {"nodes": nodes, "summary": summary}

    print(f"✅ Community summarization complete")
    return summaries
