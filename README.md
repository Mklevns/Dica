# DICA — Dynamic In-Context Alignment

Local-first scaffolding around [Ollama](https://ollama.com) that mines gold-standard Python with AST analysis, retrieves it with hybrid lexical / structural / semantic scoring, injects it as rigid in-context references, and quality-gates model output with **ruff** + **mypy** in a multi-pass self-correction loop.

Use the **CLI** (`main.py`) or the **Gradio** copilot UI (`app.py`). Both share one engine: `dica.pipeline.PipelineEngine`.

## Features

- **Corpus vault** — AST-ingest gold-standard modules into an in-memory index
- **Hybrid dispatch** — Jaccard lexical scoring + structural tag boosts + optional Ollama embeddings; empty/stopword queries fall back to tag-richness ranking
- **Budgeted prompts** — `ContextBudget` keeps system / target / diagnostics / references inside `num_ctx` (middle-out diagnostic truncation)
- **Multi-pass refinement** — Pass 0 draft (no references), then single-reference alignment with `ast.parse` gates (abort on Pass 0 failure; rollback on later regressions)
- **Extraction gate** — only `ExtractionResult.ok` code advances; format retries use real failure diagnostics
- **Sandbox verification** — ruff + mypy (Docker hardened backend, or local fallback; hung checkers are killed on timeout)
- **Gradio UI** — streams stage logs and evolving code on `http://127.0.0.1:7860`
- **Config-driven** — `config.toml` / `$DICA_CONFIG` via `dica.config.get_config()`

## Pipeline (high level)

```
ingest corpus → dispatch (top_k)
  → Pass 0 draft (task ± target script)
  → Passes 1..N: align one gold chunk at a time
  → cloud polish (optional stub, fail-soft)
  → verify (ruff + mypy)
  → self-correction loop (budgeted diagnostics)
```

## Project layout

```
.
├── main.py                 # CLI adapter
├── app.py                  # Gradio UI adapter
├── config.toml             # Runtime configuration (Ollama, dispatch, sandbox, …)
├── requirements.txt        # Core CLI dependencies
├── requirements-ui.txt     # + Gradio
├── requirements-sandbox.txt# + Docker SDK
├── requirements-dev.txt    # + pytest
├── pytest.ini
├── reference_corpus/       # Gold-standard .py modules
├── sandbox_image/          # Dockerfile.sandbox + in-container runner
├── scripts/                # Sample “messy” scripts for demos / --target
├── tests/                  # Unit + scripted pipeline tests
├── .github/workflows/      # CI (pytest on push/PR)
└── dica/
    ├── pipeline.py         # Shared multi-pass engine
    ├── config.py           # TOML load + Pydantic defaults
    ├── vault.py            # AST ingest + Jaccard search
    ├── dispatcher.py       # Hybrid retrieval
    ├── embeddings.py       # Ollama embed side-table (fail-soft)
    ├── orchestrator.py     # Prompt assembly + corrections
    ├── context.py          # Token budget manager
    ├── extraction.py       # Fenced-code extract + ast.parse gate
    └── sandbox.py          # ruff/mypy gate (docker | local | auto)
```

## Prerequisites

- **Python 3.11+** (3.11 / 3.12 covered in CI)
- **[Ollama](https://ollama.com)** with `ollama serve` running for live generation
- Generation model from `config.toml` (default: `qwen3-coder:30b`)
- Embedding model for hybrid dispatch (default: `nomic-embed-text`); set `dispatch.semantic_weight = 0` to disable
- **Optional:** Docker for the isolated sandbox image

## Installation

```bash
git clone https://github.com/Mklevns/Dica.git
cd Dica

python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Core CLI
pip install -r requirements.txt

# Optional UI
pip install -r requirements-ui.txt

# Optional Docker sandbox SDK
pip install -r requirements-sandbox.txt
docker build -t dica-sandbox:latest -f sandbox_image/Dockerfile.sandbox sandbox_image/

# Models (must match config.toml)
ollama pull qwen3-coder:30b
ollama pull nomic-embed-text
```

## Configuration

`config.toml` is loaded at runtime (`dica.config.get_config()`). Every key has a built-in default in `dica/config.py`; a partial or missing file is fine.

| Section | Controls |
|---------|----------|
| `[ollama]` | host, generation model, embed model, `num_ctx`, temperature, timeouts |
| `[dispatch]` | `top_k`, lexical / semantic weights, structural & name boosts |
| `[context]` | chars/token heuristic, reserved output tokens, diagnostic / min-chunk caps |
| `[sandbox]` | `backend` (`auto` \| `docker` \| `local`), image, timeout, resource limits |
| `[engine]` | `max_retries` (self-correction budget) |

Override the config path:

```bash
export DICA_CONFIG=/path/to/config.toml          # macOS / Linux
$env:DICA_CONFIG = "C:\path\to\config.toml"      # Windows PowerShell
```

CLI flags **`--model`** and **`--max-attempts`** default to config values and can override them per run.

**Tips**

- Disable embeddings: `semantic_weight = 0.0` in `[dispatch]`
- Force local checkers (no Docker): `backend = "local"` in `[sandbox]`
- Orphan cleanup only matches DICA labels / `dica_sandbox_` name prefixes (safe on shared hosts)

## Usage

### CLI

```bash
# Render Pass 0 payload + refinement schedule (no model call)
python main.py --dry-run "Build an async CRUD router for a Product resource"

# Full multi-pass lifecycle + verification
python main.py "Build an async CRUD router for a Product resource with Pydantic validation"

# Refactor a messy script against the corpus
python main.py --target scripts/dirty_loot_parser.py.py "Refactor to Pydantic v2 + async I/O"

# Overrides
python main.py --corpus ./reference_corpus --model phi4 --max-attempts 5 "..."
```

| Exit code | Meaning |
|-----------|---------|
| `0` | Verified output (ruff + mypy clean) |
| `1` | Infrastructure failure (empty vault, target unreadable, model unreachable, …) |
| `2` | Correction budget exhausted; last unverified output is still printed |

### Gradio UI

```bash
pip install -r requirements-ui.txt
python app.py
# → http://127.0.0.1:7860  (loopback only; do not expose without auth)
```

Upload a UTF-8 `.py` file, enter refactoring instructions, and watch dispatch → draft → alignment → verify stream live.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Coverage includes extraction (`ok` gate), vault Jaccard search, hybrid dispatch + empty-query fallback, context budgeting, orchestrator corrections, sandbox JSON parsing / timeout kill, and a scripted `PipelineEngine` path. Most tests do **not** require a live Ollama instance.

CI runs the suite on push/PR to `main` (Python 3.11 and 3.12): `.github/workflows/ci.yml`.

## Extending

- **Retrieval** — `CodeVault.search` (Jaccard) and `IntentDispatcher` (hybrid + fallback) are the ranking seams; `SemanticIndex` is a fail-soft side-table over embeddings
- **Prompts** — `PromptPayload.render_budgeted(ContextBudget)` owns `num_ctx` packing; `PromptOrchestrator.build_correction` attaches diagnostics for middle-out truncation
- **Lifecycle** — add stages in `dica/pipeline.py` once; CLI and UI both consume `PipelineEvent`s
- **Sandbox** — checker args live in `sandbox.py` (host) and `sandbox_image/runner.py` (container); Docker path is network-disabled, resource-capped, non-root
- **Cloud polish** — `cloud_polish` is a fail-soft stub; wire a frontier API without changing the pipeline contract

## License

No license file is checked in yet. Add one (e.g. MIT, Apache-2.0) before public redistribution if you need clear terms.
