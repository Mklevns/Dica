# DICA Code Review — Comprehensive Audit

**Reviewer posture:** Principal software engineer / code quality auditor  
**Scope:** Full local codebase (`C:\Dica\dica`) as of review date  
**Method:** Static deep-read of all application modules; targeted runtime probes for extraction/`unwrap` behavior  

---

## Executive Summary

DICA (Dynamic In-Context Alignment) is a local-first Python code generation/refactoring pipeline: AST-ingest a gold corpus, retrieve relevant chunks, multi-pass LLM refinement via Ollama, then gate output with ruff + mypy (Docker or local). The core design is thoughtful—immutable Pydantic payloads, fail-soft cloud polish, hardened Docker sandbox options, and careful extraction heuristics. Overall quality is **7.0/10** for a research/MVP scaffold with strong engineering taste, but several **wiring gaps** leave substantial written modules unused, and one extraction bug can let unparseable code skip format retries and poison the pipeline.

### Top 5 issues to fix first

| # | Severity | Issue |
|---|----------|--------|
| 1 | **Critical** | `unwrap_extraction` ignores `ExtractionResult.ok`, so syntax-invalid fenced code is treated as success |
| 2 | **High** | `ContextBudget` and hybrid semantic dispatch exist but are never wired into `main`/`app`/`dispatcher` |
| 3 | **High** | Large duplicated pipeline between `main.py` and `app.py` (constants, stages, gates diverge over time) |
| 4 | **High** | Config system largely bypassed (CLI model default `phi4`, hardcoded Ollama URL, hardcoded `top_k=3` vs config `2`) |
| 5 | **High** | Zero automated tests; critical path has no regression safety |

### Key high-level recommendations

- Fix extraction unwrap + Pass-0 syntax gate.
- Single shared pipeline engine for CLI and UI.
- Wire `get_config()` + `SemanticIndex` + `ContextBudget`.
- Expand `requirements.txt` / packaging.
- Add unit tests around extraction, vault, dispatch, and orchestrator.

---

## Project Understanding

### Purpose and goals

Mine high-quality local Python via AST, inject structural references into an LLM prompt, iteratively refine generated/refactored code, and quality-gate with static analysis before accepting output.

### Tech stack and architecture

| Layer | Components |
|-------|------------|
| **Language** | Python 3.11+ (`tomllib`, `ast.unparse`, modern typing) |
| **Validation** | Pydantic v2 throughout |
| **LLM** | Ollama `/api/chat` (httpx async) |
| **Embeddings** | Ollama `/api/embed` (implemented, unwired) |
| **UI** | Gradio (`app.py`) |
| **Quality gate** | ruff + mypy; Docker image `sandbox_image/` or local subprocesses |
| **Config** | TOML + Pydantic (`dica/config.py`) |

### Entry points / data flow

```
reference_corpus/*.py
        │
        ▼
   CodeVault.ingest  ──► IntentDispatcher.dispatch  ──► PromptOrchestrator
        │                        │                            │
        │                   (SemanticIndex? NO)               │
        │                                                     ▼
        │                                              LocalLLMClient.complete
        │                                                     │
        │                                              extract_python_code
        │                                                     │
        │                         multi-pass refine + cloud_polish stub
        │                                                     │
        └────────────────────────────────────────────► verify (ruff/mypy)
                                                              │
                                                       correction loop
```

### Critical components

1. **Extraction + syntax gates** — decide what code advances
2. **Sandbox** — only real quality signal
3. **Dispatch/retrieval** — determines alignment quality
4. **Orchestrator** — prompt contract and correction framing
5. **Pipeline engines** (`main.run_pipeline`, `RefactorEngine.run`) — control flow

---

## Detailed Findings

### Critical Issues

#### C1. Extraction success ignores `ok` — invalid code skips format retries

- **Severity:** Critical
- **Where:**
  - `main.unwrap_extraction` (approx. L150–173)
  - Call sites: `main._generate_code` (L265–268), `app.RefactorEngine._generate` (L193–198)
  - Producer: `dica/extraction.py` `extract_python_code` (L146–154)
- **Evidence:** When a fence exists but does not parse, extraction returns `ok=False` **with** `code` still set:

```python
# dica/extraction.py (failure path)
return ExtractionResult(
    code=blocks[0],
    ok=False,
    error=(
        "The code block failed to parse as Python "
        f"(SyntaxError at {first_error}). Emit only valid Python inside "
        "the fence — no prose, no partial statements."
    ),
)
```

```python
# main.py
def unwrap_extraction(result: ExtractionResult | None) -> str | None:
    ...
    code = result.code
    if not code:
        return None
    return code   # never checks result.ok
```

**Runtime probe:** broken fenced code `def foo(` → `ok=False`, unwrap returns the broken string, format-retry path does **not** trigger (`would_retry=False`).

- **Impact:** Format-correction retries never fire for unparseable fences. Pass 0 can store broken code (no `ast.parse` gate). Correction loop can re-verify garbage. Pipeline wastes LLM calls on doomed code and may emit unverified nonsense.
- **Recommendation (effort: Low):**

```python
def unwrap_extraction(result: ExtractionResult | None) -> str | None:
    if result is None or not result.ok or not result.code:
        return None
    return result.code
```

Optionally feed `result.failure_prompt` into `build_correction` instead of a generic “no fence” message.

---

### High Priority Issues

#### H1. `ContextBudget` never applied — context overflow risk

- **Severity:** High
- **Where:** Full implementation in `dica/context.py`; **zero imports** from `main`/`app`/`orchestrator`. Sandbox docs point callers at it (`sandbox.py` L115–117) but correction path embeds raw diagnostics.
- **Impact:** `build_correction` concatenates original task + full failing code + full ruff/mypy output inside `target_task` (`orchestrator.py` L398–408). Under `--strict`, mypy output can dominate `num_ctx=8192`, truncating instructions at the model, not under DICA’s control. Multi-pass + large targets amplify this.
- **Recommendation:** Assemble correction/refinement prompts via `ContextBudget.assemble`, using `truncate_middle` on diagnostics and admitting references only if headroom allows.
- **Effort:** Medium

#### H2. Semantic retrieval module is dead code

- **Severity:** High
- **Where:** `dica/embeddings.py` (`SemanticIndex`, `OllamaEmbedder`); `DispatchConfig.semantic_weight` in `config.py` / `config.toml`; `IntentDispatcher` only uses lexical + structural boosts (`dispatcher.py`).
- **Impact:** Documented hybrid retrieval does not run. README/config promise semantic scoring that never executes. Operators tune `semantic_weight` with no effect.
- **Recommendation:** Inject optional `SemanticIndex` into dispatcher; score as  
  `total = lexical_w * lex + semantic_w * sem + structural`.
- **Effort:** Medium

#### H3. Dual pipeline implementations (CLI vs Gradio)

- **Severity:** High
- **Where:** `main.run_pipeline` vs `app.RefactorEngine.run`. Duplicated: `REFINEMENT_PASSES`, `DISPATCH_TOP_K`, `FORMAT_RETRIES`, cloud polish, verify, corrections.
- **Impact:** Bug fixes (e.g. C1, Pass-0 syntax gate) must land twice; already diverging (app has no shared `_syntax_regression` helper; different exception handling width).
- **Recommendation:** One `PipelineEngine` in `dica/` used by both CLI and UI.
- **Effort:** Medium–High

#### H4. Configuration largely unused by runtime

- **Severity:** High

| Setting | Config says | Runtime actually uses |
|---------|-------------|------------------------|
| `ollama.host` | `config.toml` | Hardcoded `OLLAMA_URL` in `main.py` |
| `ollama.model` | `qwen3-coder:30b` | CLI default `--model phi4`; app uses `LLMConfig()` default `qwen3-coder:30b` |
| `dispatch.top_k` | `2` | Hardcoded `DISPATCH_TOP_K = 3` in both entrypoints |
| Structural/name boosts | config | Module constants `_STRUCTURAL_BOOST` / `_KIND_NAME_BOOST` |
| `engine.max_retries` | config | CLI flag / app `MAX_ATTEMPTS` |

- **Impact:** Operator config is a false control panel; CLI vs UI model defaults differ; surprise production behavior.
- **Recommendation:** Single `cfg = get_config()` at startup; wire LLM client, dispatch, sandbox, retries.
- **Effort:** Medium

#### H5. No automated tests

- **Severity:** High
- **Where:** Repo-wide — no `tests/`, no pytest config, no CI.
- **Impact:** Extraction salvage, decorator reattachment, Docker parse path, and multi-pass rollback are high-regression surfaces with zero guardrails.
- **Recommendation:** Start with pure unit tests (extraction, vault keywords, distill_syntax, budget, dispatcher scoring) + one fake-LLM integration test for the engine.
- **Effort:** Medium (setup), ongoing

#### H6. Incomplete / misleading dependency surface

- **Severity:** High
- **Where:** `requirements.txt` only lists `pydantic`, `httpx`, `ruff`, `mypy`.
- **Missing for claimed features:** `gradio` (required by `app.py`), `docker` (optional but primary sandbox path).
- **Impact:** Fresh clone cannot run UI; Docker backend fails until ad-hoc install; README admits Gradio as “optional” after installing core deps.
- **Recommendation:** `requirements.txt` + `requirements-ui.txt` / extras, or a proper `pyproject.toml` with optional deps.
- **Effort:** Low

#### H7. Pass 0 has no syntax regression gate

- **Severity:** High
- **Where:** `main.py` accepts any non-`None` unwrap after Pass 0; only passes 1..N check `_syntax_regression`. Same pattern in `app.py`.
- **Impact:** Unparseable draft becomes the refinement target; later rollbacks still leave broken Pass-0 for final verify.
- **Recommendation:** Apply the same `ast.parse` gate (and prefer extraction `ok`) after every generation including Pass 0.
- **Effort:** Low

---

### Medium Priority Issues

#### M1. Local sandbox timeout may orphan checker processes

- **Where:** `sandbox._local_verify` uses `asyncio.wait_for` on `gather` of subprocesses without `proc.kill()` on timeout.
- **Impact:** Hung mypy/ruff can linger after `TimeoutError`, consuming CPU/memory. Docker path kills the container (good).
- **Recommendation:** Track process handles; on timeout, kill and wait.
- **Effort:** Low–Medium

#### M2. Correction diagnostics unbounded

- **Where:** `orchestrator.build_correction`; no call to `ContextBudget.truncate_middle`.
- **Impact:** See H1; also dilutes the original task in recency-anchored prompts.
- **Effort:** Low once H1 lands

#### M3. Orphan container sweep is broad

- **Where:** `cleanup_orphaned_containers` filters by label, name prefix, **and** `ancestor: cfg.image`.
- **Impact:** Force-removes **any** container based on `dica-sandbox:latest`, including manually started debug containers.
- **Recommendation:** Prefer label+name only; use ancestor only with explicit opt-in.
- **Effort:** Low

#### M4. Vault scoring is not Jaccard as documented

- **Where:** `vault.search`: `score = len(overlap) / (len(query) or 1)`. Comment claims mild penalty for huge chunks matching everything; denominator does not include `|chunk.keywords|` / union.
- **Impact:** Large keyword bags are not down-weighted; retrieval can prefer “broad” modules.
- **Effort:** Low

#### M5. Empty / stopword-only queries yield empty dispatch

- **Where:** `vault.search` skips non-overlapping chunks; empty query ⇒ no hits; pipeline still runs Pass 0 with zero refinements (warned, not failed).
- **Impact:** Confusing UX; generation without style anchors.
- **Effort:** Low (validate query / fallback to popular chunks)

#### M6. Package / import design

- **Where:** `app.py`: `from main import ...`; `main` is a script, not `dica.pipeline`.
- **Impact:** Hard to install as a package, hard to test, circular import risk if `main` ever imports `app`.
- **Recommendation:** Move client/pipeline into `dica/`; thin `main.py` / `app.py` CLIs.
- **Effort:** Medium

#### M7. `distill_syntax` via `ast.unparse` drops comments and type-ignore directives

- **Where:** `orchestrator.distill_syntax`.
- **Impact:** Gold references may lose `# type: ignore` / nuanced comments the model should mirror; formatting normalization can alter style teaching. Acceptable for token density, but should be documented as a tradeoff.
- **Effort:** Low (docs) / High (comment-preserving pruner)

#### M8. Dockerfile vs config path mismatch

- **Where:** `config.toml` says built from `docker/Dockerfile.sandbox`; actual path `sandbox_image/Dockerfile.sandbox`.
- **Impact:** Operator build failures.
- **Effort:** Low

#### M9. httpx client per completion request

- **Where:** `LocalLLMClient.complete` creates a new `AsyncClient` every call.
- **Impact:** Extra TLS/connection overhead across 4+ passes + corrections. Embeddings already batch better.
- **Effort:** Low

#### M10. Pydantic tag false positives

- **Where:** `_PYDANTIC_MARKERS` includes `"Field"` (`vault.py`).
- **Impact:** Non-Pydantic code mentioning `Field` gets `has_pydantic` and forces Pydantic constraints via `derive_constraints`.
- **Effort:** Low

---

### Low Priority / Improvement Opportunities

- **L1.** `scripts/*.py.py` double extensions and sample “messy” code committed without clear labeling.
- **L2.** UI markdown still says “Phi-4” (`app.py`) while defaults lean `qwen3-coder`.
- **L3.** `cloud_polish` stub always raises — fine, but surface a config flag `engine.cloud_polish_enabled` to skip the try/except noise.
- **L4.** Chunk IDs: SHA-1 of `file:name:lineno` truncated to 12 hex — collision risk tiny but real if re-ingest renames collide. Prefer full hash or content hash.
- **L5.** No LICENSE file despite README section.
- **L6.** No `pyproject.toml`, typecheck/lint config, pre-commit, or CI.
- **L7.** Gradio binds `127.0.0.1` (good); document that exposing `0.0.0.0` would let anyone drive LLM + sandbox on the host.
- **L8.** `get_config` is `lru_cache`’d forever — tests need cache clear; document `get_config.cache_clear()`.
- **L9.** Local verify does not share checker args with container runner — drift risk between `sandbox.py` and `runner.py`.

---

## Strengths & Well-Engineered Areas

1. **Clear modular boundaries** with module docstrings that explain *why* (AST vault, soft structural boosts, recency-anchored prompts).
2. **Immutable Pydantic models** for chunks, payloads, verification reports — excellent for logging and debugging.
3. **Extraction module is production-minded** — unclosed fences, salvage window, largest *parseable* block (when `ok` is respected).
4. **Docker sandbox hardening** is above average for a small project: `network_disabled`, read-only rootfs, tmpfs, mem/pids/CPU caps, `no-new-privileges`, `cap_drop=ALL`, non-root user, orphan sweep.
5. **Fail-soft patterns** done correctly: cloud polish, per-batch embed failures, vault skip on parse errors, Docker probe → local fallback.
6. **Async discipline** is thoughtful: Docker via `to_thread`, local checkers via `create_subprocess_exec`, embed `run_coro_blocking` for sync call sites.
7. **UI protocol injection** (`SupportsComplete`) is the right seam for testability once tests exist.
8. **Intermediate `ast.parse` rollback** on refinement passes is a smart anti-compounding-failure design (when extraction `ok` is honored).

---

## Architectural & Design Recommendations

1. **Promote a single pipeline module**  
   `dica/pipeline.py`: ingest → dispatch → multi-pass → polish → verify → correct. CLI and Gradio become thin adapters.

2. **Actually use the config root object**  
   Every tunable constant currently duplicated should come from `DicaConfig`. Kill module-level magic numbers that shadow TOML.

3. **Wire the “side table” architecture as designed**  
   Keep vectors off `CodeChunk`; build `SemanticIndex` after ingest in both entrypoints; pass into dispatcher.

4. **Prompt assembly should own budgeting**  
   `PromptPayload.render()` + separate budget pass, or make orchestrator return `AssembledPayload` from `ContextBudget`.

5. **Packaging**  
   ```
   pyproject.toml
   [project.scripts]
   dica = "dica.cli:main"
   dica-ui = "dica.ui:main"
   ```
   Stop importing `main` from `app`.

6. **Separate “demo corpus / dirty scripts” from library code** clearly (e.g. `examples/`).

---

## Testing Strategy Recommendations

| Layer | What to test | Why |
|-------|----------------|-----|
| **Unit** | `extract_python_code` (closed/open fences, salvage, multi-block, ok=False) | C1-class bugs |
| **Unit** | `CodeVault` decorator reattachment, keyword split, empty files | Retrieval quality |
| **Unit** | `IntentDispatcher` scoring order / structural boosts | Ranking regressions |
| **Unit** | `distill_syntax`, `derive_constraints`, `build_correction` shape | Prompt contract |
| **Unit** | `ContextBudget.truncate_middle` / assemble drop order | Token safety |
| **Unit** | `_parse_runner_output` JSON line selection | Sandbox protocol |
| **Integration** | Fake `SupportsComplete` through full multi-pass + correction | Engine parity CLI/UI |
| **Optional e2e** | Docker verify against known clean/dirty snippets | Gate trust |
| **Property** | Round-trip: random valid AST subset survives distill + extract | Robustness |

No property-based or integration tests exist today — that is the largest quality gap relative to the sophistication of the design.

---

## Security Assessment Summary

| Area | Assessment |
|------|------------|
| **Secrets** | No hardcoded API keys; cloud polish stub only. Good for local-first. |
| **Sandbox (Docker)** | Strong isolation posture for *static analysis*. Suitable baseline if you later execute generated code. |
| **Sandbox (local)** | Runs checkers in host Python env — acceptable for trusted local use; not multi-tenant safe. |
| **Gradio** | Loopback bind is correct. No auth — fine locally; dangerous if exposed. |
| **Input** | Uploaded scripts and prompts are not sanitized beyond UTF-8 read; risk is mainly resource/LLM prompt injection, not RCE today (static gate only). |
| **Supply chain** | Unpinned mins in `requirements.txt` (`>=`); Docker image pins ruff/mypy — good contrast. Prefer pins or lockfile for reproducibility. |
| **Orphan cleanup** | Over-broad ancestor filter can delete non-DICA containers sharing the image name. |
| **SSRF / host** | Ollama URL currently fixed to localhost; when configurable, validate host allowlist so config cannot point HTTP client at internal metadata endpoints in cloud deploys. |

Overall: **reasonable local security**, not ready as a shared multi-user service without auth, rate limits, and mandatory Docker backend.

---

## Performance & Scalability Observations

- **In-memory vault** is fine for small corpora (`reference_corpus` ~7 files); will not scale to large monorepos without the LanceDB/Chroma swap the code already anticipates.
- **Full re-ingest every CLI run**; Gradio ingests once at startup — good for UI, wasteful for CLI batching.
- **Multi-pass LLM** (Pass 0 + up to 3 alignments + up to 3 corrections + format retries) ⇒ high latency and GPU load; expected for quality, but no caching of identical prompts.
- **Semantic index** (when wired) batches embeds with concurrency cap — well designed.
- **Token heuristic** (3.2 chars/token) is conservative; without budget assembly, the heuristic’s care is unused.
- **Blocking risk:** mostly avoided; Docker path correctly offloaded.

Bottleneck in practice: **serial Ollama generation**, not Python.

---

## Maintainability & Technical Debt Assessment

| Debt | Severity | Why it hurts later |
|------|----------|-------------------|
| Dual engines (`main` / `app`) | High | Behavioral drift |
| Unwired modules (`embeddings`, `context`, most of `config`) | High | Dead code rots; docs lie |
| Script-as-library (`from main import`) | Medium | Packaging / testing pain |
| Magic constants vs TOML | Medium | Ops confusion |
| No tests / CI | High | Every refactor is risky |
| Duplicated ruff/mypy args (host vs container runner) | Medium | Gate divergence |
| Incomplete requirements | Medium | Onboarding friction |

Documentation quality inside modules is **excellent** (better than many production codebases). The debt is less “messy code” and more **incomplete integration of well-written parts**.

---

## Actionable Prioritized Roadmap

### Quick wins (Low effort, High impact)

1. Fix `unwrap_extraction` to require `result.ok`; use `failure_prompt` in format retries.
2. Apply syntax gate after Pass 0 (and any generation).
3. Align CLI `--model` default with `config.toml` / `LLMConfig`; document one source of truth.
4. Fix `config.toml` Dockerfile path comment; add `gradio` (and optional `docker`) to dependency files.
5. Narrow orphan cleanup filters to label + name prefix.

### Next sprint (Medium effort, High impact)

6. Extract shared `dica/pipeline.py` + shared constants from config.
7. Wire `get_config()` into LLM client, dispatch `top_k`/weights, sandbox, retries.
8. Wire `ContextBudget` into correction (and preferably all) prompt assembly.
9. Wire `SemanticIndex` into dispatcher behind `semantic_weight == 0` kill-switch.
10. Add `tests/` for extraction, vault, dispatcher, budget; run in CI.

### Structural (Medium–High effort)

11. Proper `pyproject.toml` package layout; remove `from main import`.
12. Deduplicate sandbox checker args (shared module imported by runner build).
13. Connection-pooled `httpx.AsyncClient` lifecycle on the LLM client.
14. Pin dependencies / lockfile; pin sandbox image contents in docs.

---

## Conclusion

DICA reads like a carefully designed local coding-agent scaffold written by someone who understands LLM failure modes (fence truncation, syntax regression compounding, sandbox isolation). The **module-level craftsmanship is strong**; the main gap is **integration completeness**: budget manager, semantic index, and config system are largely spectators while the real control plane is hardcoded in two parallel entrypoints. The single most dangerous concrete bug is **`unwrap_extraction` ignoring `ok`**, which undermines the extraction module’s entire contract.

If you fix extraction + unify the pipeline + wire config/budget/semantics + add a small high-value test suite, this codebase would sit comfortably in the **8.5+/10** range as a maintainable local tooling product rather than a partially connected prototype.

---

## Limitations of This Analysis

- Reviewed the local tree as of this session (~5k LOC application code); did not run full end-to-end against a live Ollama model or Docker daemon.
- Did not audit `reference_corpus/*` line-by-line as production product code (treated as gold examples).
- Did not perform dependency CVE scanning or runtime load tests.
- Gradio/async streaming behavior under concurrent multi-user load was reasoned about statically, not load-tested.
