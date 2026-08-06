import networkx as nx
from sentence_transformers import SentenceTransformer


def embed_graph_nodes(G: nx.Graph, embed_model: SentenceTransformer) -> dict:
    """Create vector embeddings for each node using its description and neighbors."""
    node_texts = {}
    for node in G.nodes():
        data      = G.nodes[node]
        neighbors = list(G.neighbors(node))[:5]
        text = (
            f"{node} is a {data.get('type', '')}. "
            f"{data.get('description', '')}. "
            f"Related to: {', '.join(neighbors)}"
        )
        node_texts[node] = text

    if not node_texts:
        return {}

    nodes      = list(node_texts.keys())
    embeddings = embed_model.encode([node_texts[n] for n in nodes], show_progress_bar=False)

    print(f"✅ Embedded {len(nodes)} graph nodes")
    return {nodes[i]: embeddings[i] for i in range(len(nodes))}
