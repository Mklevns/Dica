# DICA — Dynamic In-Context Alignment

Async scaffolding around a local LLM (via [Ollama](https://ollama.com)) that mines gold-standard Python with AST analysis, injects it as rigid in-context references, and quality-gates model output with **ruff** + **mypy** in a multi-pass self-correction loop.

Use it from the CLI (`main.py`) or the Gradio web UI (`app.py`).

## Features

- **Corpus vault** — ingest reference Python modules via AST into an in-memory index
- **Intent dispatch** — hybrid lexical + semantic retrieval of relevant gold-standard chunks
- **Multi-pass refinement** — draft, then align one reference at a time with fail-fast `ast.parse` rollback
- **Sandbox verification** — ruff + mypy gate (Docker or local backend)
- **Gradio copilot UI** — live streaming of each pipeline stage

## Project layout

```
dica/
├── main.py                 # CLI pipeline entrypoint
├── app.py                  # Gradio refactoring copilot UI
├── config.toml             # Unified pipeline configuration
├── requirements.txt
├── reference_corpus/       # Gold-standard .py modules
├── sandbox_image/          # Docker sandbox image + runner
├── scripts/                # Ad-hoc helpers
└── dica/
    ├── config.py           # Config loading (TOML + defaults)
    ├── vault.py            # AST ingestor + in-memory index
    ├── dispatcher.py       # Keyword + structural-tag retrieval
    ├── embeddings.py       # Local embedding support
    ├── orchestrator.py     # Prompt payload assembly
    ├── extraction.py       # Code extraction from model output
    ├── sandbox.py          # ruff/mypy quality gate
    └── context.py          # Context budgeting helpers
```

## Prerequisites

- Python 3.11+ recommended
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- A generation model pulled (see `config.toml`; e.g. `qwen3-coder:30b` or `phi4`)
- Optional: Docker for isolated sandbox verification

## Installation

```bash
# Clone and enter the repo
git clone <your-repo-url> dica
cd dica

# Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models (names must match config.toml)
ollama pull qwen3-coder:30b
ollama pull nomic-embed-text
```

Optional Gradio UI dependency (if not already available in your environment):

```bash
pip install gradio
```

## Configuration

Defaults live in `config.toml` (Ollama host/model, dispatch weights, sandbox limits, retry budget). Override the file path with:

```bash
export DICA_CONFIG=/path/to/config.toml   # macOS / Linux
$env:DICA_CONFIG = "C:\path\to\config.toml"  # Windows PowerShell
```

Every key has a built-in default in `dica/config.py`; a partial or missing file is fine.

## Usage

### CLI

```bash
# Inspect the rendered payload without calling the model
python main.py --dry-run "Build an async CRUD router for a Product resource"

# Full lifecycle with self-correction
python main.py "Build an async CRUD router for a Product resource with Pydantic validation"

# Refactor an existing script against the corpus
python main.py --target messy.py "Refactor to Pydantic v2 + async I/O"

# Custom corpus / model / attempts
python main.py --corpus ./reference_corpus --model phi4 --max-attempts 5 "..."
```

**Exit codes:** `0` = verified output · `1` = infra failure (empty vault / no Ollama) · `2` = exhausted correction attempts

### Gradio UI

```bash
python app.py
# → http://127.0.0.1:7860
```

## Extending

- **Vector store** — `CodeVault.search` is the retrieval seam; swap keyword overlap for embedding similarity (LanceDB, Chroma, etc.).
- **Gate tuning** — adjust ruff/mypy args in `sandbox.py`.
- **Batch queries** — `run_pipeline` is a pure coroutine; fan out with `asyncio.gather`.
- **Sandbox image** — build from `sandbox_image/Dockerfile.sandbox` when using the Docker backend.

## License

Add a license file when you publish the repository (e.g. MIT, Apache-2.0).
