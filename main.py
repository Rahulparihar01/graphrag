"""
GraphRAG CLI — Knowledge Graph + RAG for PDFs
Powered by Gemini
"""

import os
import sys
from dotenv import load_dotenv
from graphrag import GraphRAG

load_dotenv()


def main():
    print("""
╔══════════════════════════════════════╗
║         GraphRAG Demo                ║
║   Knowledge Graph + RAG for PDFs     ║
║         Powered by Gemini            ║
╚══════════════════════════════════════╝
""")

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("❌ Set your GEMINI_API_KEY environment variable first!")
        print("   export GEMINI_API_KEY='your-key-here'")
        print("   👉 Get a FREE key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    rag = GraphRAG(api_key=api_key)

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else input("📄 Enter PDF file path: ").strip()

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    rag.index(pdf_path)

    stats = rag.graph_stats()
    print(f"📊 Top Entities: {[e[0] for e in stats['top_entities'][:5]]}")

    print("\n💬 Ask questions about your PDF!")
    print("   local: your question  → specific facts")
    print("   global: your question → broad themes")
    print("   quit to exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Bye!")
            break

        mode     = "local"
        question = user_input

        if user_input.startswith("global:"):
            mode, question = "global", user_input[7:].strip()
        elif user_input.startswith("local:"):
            mode, question = "local", user_input[6:].strip()

        if question:
            print(f"\nGraphRAG: {rag.query(question, mode=mode)}\n")


if __name__ == "__main__":
    main()
