from sentence_transformers import SentenceTransformer
from gemini_client import get_client
from config import EMBED_MODEL, MAX_CHUNKS_TO_PROCESS
from core import (
    load_pdf,
    chunk_text,
    extract_entities_and_relations,
    build_knowledge_graph,
    detect_communities,
    summarize_communities,
    embed_graph_nodes,
)
from search import local_search, global_search


class GraphRAG:
    """
    Main GraphRAG orchestrator.

    Usage:
        rag = GraphRAG(api_key="your-gemini-key")
        rag.index("document.pdf")
        answer = rag.query("What is this about?", mode="local")
    """

    def __init__(self, api_key: str):
        self.client             = get_client(api_key)
        self.embed_model        = SentenceTransformer(EMBED_MODEL)
        self.G                  = None
        self.node_embeddings    = {}
        self.community_summaries = {}
        self.is_indexed         = False

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(self, pdf_path: str):
        """Parse a PDF, build the knowledge graph, and prepare for querying."""
        print(f"\n{'='*50}\n🚀 Starting GraphRAG Indexing: {pdf_path}\n{'='*50}\n")

        text   = load_pdf(pdf_path)
        chunks = chunk_text(text)

        chunks_to_process = chunks[:MAX_CHUNKS_TO_PROCESS]
        print(f"ℹ️  Processing first {len(chunks_to_process)} chunks")

        extractions              = extract_entities_and_relations(chunks_to_process, self.client)
        self.G                   = build_knowledge_graph(extractions)
        communities              = detect_communities(self.G)
        self.community_summaries = summarize_communities(communities, self.G, self.client)
        self.node_embeddings     = embed_graph_nodes(self.G, self.embed_model)
        self.is_indexed          = True

        print(f"\n{'='*50}")
        print("✅ GraphRAG Indexing Complete!")
        print(f"   Nodes: {self.G.number_of_nodes()} | "
              f"Edges: {self.G.number_of_edges()} | "
              f"Communities: {len(self.community_summaries)}")
        print(f"{'='*50}\n")

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(self, question: str, mode: str = "local") -> str:
        """
        Query the indexed document.

        Args:
            question: Natural-language question.
            mode:     'local'  → specific facts via node similarity
                      'global' → broad themes via community summaries
        """
        if not self.is_indexed:
            return "❌ Please index a PDF first using graphrag.index('your_file.pdf')"

        print(f"\n🔎 [{mode.upper()} SEARCH] {question}")

        if mode == "local":
            return local_search(question, self.G, self.node_embeddings, self.embed_model, self.client)
        elif mode == "global":
            return global_search(question, self.community_summaries, self.client)

        return "❌ Invalid mode. Use 'local' or 'global'"

    # ── Utilities ─────────────────────────────────────────────────────────────

    def graph_stats(self) -> dict:
        """Return summary statistics about the knowledge graph."""
        if not self.G or self.G.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "communities": 0, "top_entities": []}

        return {
            "nodes":       self.G.number_of_nodes(),
            "edges":       self.G.number_of_edges(),
            "communities": len(self.community_summaries),
            "top_entities": sorted(
                [(n, self.G.nodes[n].get("count", 0)) for n in self.G.nodes()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }
