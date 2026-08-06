# GraphRAG for PDFs

Turn a PDF into an inspectable knowledge graph and ask grounded questions with Gemini. GraphRAG extracts entities and their relationships, groups related concepts into communities, and supports focused and high-level retrieval.

> This is a small, in-memory reference implementation. Indexes are rebuilt for each run and are not persisted to disk.

## What it does

1. Extracts text from a PDF with PyMuPDF.
2. Splits the text into overlapping, sentence-aware chunks.
3. Uses Gemini to identify entities and relationships in each chunk.
4. Builds an undirected NetworkX graph where entities are nodes and relationships are edges.
5. Detects graph communities and asks Gemini to summarize them.
6. Embeds graph nodes locally and answers questions using either node-level or community-level context.

## Query modes

| Mode | Retrieval source | Use it for |
| --- | --- | --- |
| `local` | Semantically similar entities, their edges, and source snippets | Specific facts, named entities, and relationships |
| `global` | Gemini summaries of graph communities | Document themes, overviews, and cross-topic questions |

## Project layout

```text
.
├── main.py                 # Interactive command-line interface
├── graphrag.py             # GraphRAG orchestration class
├── gemini_client.py        # Gemini client and generation wrapper
├── config.py               # Models and retrieval/indexing settings
├── requirements.txt
├── core/
│   ├── pdf_loader.py       # PDF text extraction
│   ├── chunker.py          # Overlapping text chunks
│   ├── extractor.py        # Gemini entity/relation extraction
│   ├── graph_builder.py    # NetworkX graph construction
│   ├── community.py        # Community detection and summaries
│   └── embedder.py         # Node embeddings
└── search/
    ├── local_search.py     # Focused entity-based search
    └── global_search.py    # Community-summary search
```

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- A text-based PDF (scanned PDFs need OCR before indexing)

The first run downloads the `all-MiniLM-L6-v2` SentenceTransformer model, so it needs internet access and may take a little longer.

## Installation

```bash
git clone <repository-url>
cd graphrag

python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

`main.py` loads this file automatically. You can instead set `GEMINI_API_KEY` in your shell environment.

## Run the interactive CLI

Pass a PDF path directly:

```bash
python main.py path/to/document.pdf
```

Or omit the path and enter one when prompted:

```bash
python main.py
```

After indexing, ask questions in the prompt:

```text
You: What are the main submission requirements?
You: local: Which organization is responsible for the deadline?
You: global: Summarize the document's main themes.
You: quit
```

Questions without a prefix use `local` mode. Use `global:` for a broad, community-summary-based answer; `local:` is the explicit form of the default mode.

## Use it as a library

```python
import os
from graphrag import GraphRAG

rag = GraphRAG(api_key=os.environ["GEMINI_API_KEY"])
rag.index("path/to/document.pdf")

answer = rag.query("What are the submission deadlines?", mode="local")
overview = rag.query("Summarize the document's main themes.", mode="global")

print(answer)
print(overview)
print(rag.graph_stats())
```

`graph_stats()` returns the number of nodes, edges, and communities, plus the ten most frequently extracted entities.

## Configuration

Edit `config.py` to tune the pipeline:

| Setting | Default | Meaning |
| --- | --- | --- |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for extraction, summaries, and answers |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model used to embed graph nodes |
| `CHUNK_SIZE` | `500` | Target characters in each text chunk |
| `CHUNK_OVERLAP` | `100` | Overlapping characters between chunks |
| `TOP_K` | `5` | Nodes included in local-search context |
| `MAX_CHUNKS_TO_PROCESS` | `10` | Maximum chunks indexed per run |

`MAX_CHUNKS_TO_PROCESS` intentionally defaults to `10` for quick experiments. Raise it to cover more of a long PDF; doing so increases Gemini calls, latency, and cost.

## How answers are grounded

Local answers are generated only from the retrieved graph nodes: entity descriptions, their relationships, and a short source-text snippet. Global answers are generated from community summaries rather than raw PDF text. Treat both modes as assistive output and verify important details against the original PDF.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `Set your GEMINI_API_KEY environment variable first!` | Add `GEMINI_API_KEY` to `.env` or export it in your shell. |
| `File not found` | Use a valid path to the PDF, relative to the directory you run from or absolute. |
| `No graph data available` | Confirm the PDF contains extractable text and that Gemini extraction succeeded. |
| Indexing is slow | Lower `MAX_CHUNKS_TO_PROCESS`; one extraction request is made for every processed chunk, and summaries add more requests. |
| Answers miss information | Increase `MAX_CHUNKS_TO_PROCESS`, adjust chunk settings, and use `global` for broad questions. |

## Dependencies

PyMuPDF, NetworkX, NumPy, SentenceTransformers, scikit-learn, Google Gen AI SDK, and python-dotenv. See [requirements.txt](requirements.txt) for the installable package list.

## License

No license file is currently included. Add one before redistributing the project.
