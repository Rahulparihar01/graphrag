import networkx as nx
from collections import defaultdict


def build_knowledge_graph(extractions: list[dict]) -> nx.Graph:
    """Build a NetworkX graph from extracted entities and relationships."""
    G = nx.Graph()
    entity_chunks = defaultdict(list)

    for extraction in extractions:
        chunk_txt = extraction.get("chunk_text", "")

        # Add entity nodes
        for entity in extraction.get("entities", []):
            name = entity["name"].strip()
            if not name:
                continue
            entity_chunks[name].append(chunk_txt)
            if G.has_node(name):
                G.nodes[name]["count"] += 1
            else:
                G.add_node(
                    name,
                    type=entity.get("type", "UNKNOWN"),
                    description=entity.get("description", ""),
                    count=1,
                    source_chunks=[],
                )
            G.nodes[name]["source_chunks"].append(chunk_txt[:200])

        # Add relationship edges
        for rel in extraction.get("relationships", []):
            src      = rel.get("source", "").strip()
            tgt      = rel.get("target", "").strip()
            relation = rel.get("relation", "related_to")

            if src and tgt and src != tgt:
                for node in [src, tgt]:
                    if not G.has_node(node):
                        G.add_node(node, type="UNKNOWN", description="", count=1, source_chunks=[])

                if G.has_edge(src, tgt):
                    G[src][tgt]["relations"].append(relation)
                else:
                    G.add_edge(src, tgt, relations=[relation], weight=1)

    print(f"✅ Knowledge Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G
