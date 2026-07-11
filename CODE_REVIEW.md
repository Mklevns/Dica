# DICA Code Review — Comprehensive Audit

**Reviewer posture:** Principal software engineer / code quality auditor  
**Original scope:** Full local codebase (`C:\Dica\dica`) as of initial review  
**Method:** Static deep-read of all application modules; targeted runtime probes for extraction/`unwrap` behavior  
**Remediation status:** Updated after follow-on implementation work (commit `ce9739c` on `main`)

---

## Remediation Status (post-review)

Work landed on `main` (pushed to `origin`) addresses the **critical** and **high-priority** findings from this audit. Status legend: **Done** · **Partial** · **Open**.

| ID | Finding | Status | Resolution notes |
|----|---------|--------|------------------|
| **C1** | `unwrap_extraction` ignores `ok` | **Done** | Requires `result.ok`; format retries use `failure_prompt` + failed snippet (`dica/pipeline.py`) |
| **H7** | Pass 0 lacks syntax gate | **Done** | `syntax_regression` after Pass 0 aborts pipeline (no prior good target) |
| **H4** | Config unused by runtime | **Done** | `get_config()` drives host/model/timeouts, dispatch weights/`top_k`, retries; CLI defaults match TOML |
| **H6** | Incomplete dependencies | **Done** | Split `requirements.txt` / `-ui` / `-sandbox` / `-dev`; README install paths |
| **H3** | Dual CLI/UI pipelines | **Done** | Shared `dica/pipeline.py` `PipelineEngine`; thin `main.py` + `app.py` adapters |
| **H1** | `ContextBudget` unwired | **Done** | `assemble_recency` + `PromptPayload.render_budgeted`; corrections use diagnostics section |
| **H2** | Semantic index dead code | **Done** | `SemanticIndex` built at engine init; hybrid score in `IntentDispatcher` (fail-soft) |
| **H5** | No automated tests | **Done** | `tests/` + `pytest.ini` — **37 tests** covering extraction, vault, dispatch, budget, orchestrator, sandbox parse, pipeline |
| **M1** | Local sandbox timeout orphans checkers | **Done** | `_local_verify` kills + reaps ruff/mypy on timeout (`_kill_process`) |
| **M3** | Broad orphan container `ancestor` filter | **Done** | Sweep uses label + name prefix only (no image ancestor) |
| **M4** | Vault score not true Jaccard | **Done** | `search` uses `|∩|/|∪|` over token bags |
| **M5** | Empty query → empty dispatch | **Done** | Tag-richness fallback schedule when no lexical/semantic hits |
| **M2** | Unbounded correction diagnostics | **Done** | Folded into H1 (`diagnostics` field + middle-out truncation) |
| **L6** | No CI | **Partial** | GitHub Actions runs `pytest` on push/PR; `pyproject.toml` still open |
| **M6** | `from main import` / packaging | **Partial** | Pipeline lives in `dica/pipeline.py`; adapters re-export helpers. Full `pyproject.toml` still open |
| **M8** | Dockerfile path in config | **Done** | Comment points at `sandbox_image/Dockerfile.sandbox` |
| **L2** | UI “Phi-4” copy | **Done** | Gradio copy references config.toml model |
| **L8** | Config cache in tests | **Partial** | Tests call `get_config.cache_clear()` in `conftest.py`; public docs still light |

### Still open (not addressed in remediation)

| ID | Severity | Topic |
|----|----------|--------|
| **M7** | Medium/Low | `distill_syntax` drops comments / type-ignores |
| **M9** | Low | New `httpx.AsyncClient` per LLM completion |
| **M10** | Low | `"Field"` in Pydantic markers → false positives |
| **L1, L3–L5, L7, L9** | Low | Script naming, cloud-polish flag, chunk IDs, LICENSE, bind docs, checker-arg drift |
| **L6** | Low | `pyproject.toml` / pre-commit (CI pytest workflow **Done**) |

---

## Executive Summary

DICA (Dynamic In-Context Alignment) is a local-first Python code generation/refactoring pipeline: AST-ingest a gold corpus, retrieve relevant chunks, multi-pass LLM refinement via Ollama, then gate output with ruff + mypy (Docker or local).

**Original review score: 7.0/10** — strong module craftsmanship, incomplete integration (budget, semantics, config, dual engines, no tests), and a critical extraction-unwrap bug.

**Post-remediation score: ~8.5/10** — critical/high issues closed; shared engine, live config, budgeted prompts, hybrid retrieval, and a focused pytest suite. Remaining work is medium/low polish (process kill, packaging/CI, Jaccard honesty, multi-tenant readiness).

### Original top 5 (all addressed)

| # | Severity | Issue | Outcome |
|---|----------|--------|---------|
| 1 | Critical | `unwrap` ignores `ok` | **Fixed** |
| 2 | High | Budget + semantic unwired | **Fixed** |
| 3 | High | Dual pipelines | **Fixed** (`PipelineEngine`) |
| 4 | High | Config bypassed | **Fixed** |
| 5 | High | Zero tests | **Fixed** (37 tests) |

### Key recommendations (updated)

- ~~Fix extraction unwrap + Pass-0 syntax gate~~ **Done**
- ~~Single shared pipeline engine~~ **Done**
- ~~Wire `get_config()` + `SemanticIndex` + `ContextBudget`~~ **Done**
- ~~Expand dependency surface~~ **Done** (split requirements; `pyproject.toml` still open)
- ~~Unit/integration tests~~ **Done** (local pytest; CI wiring still open)
- **Next:** M1 process kill, M3 orphan filter, packaging/CI, optional true Jaccard / dispatch fallbacks

---

## Project Understanding

### Purpose and goals

Mine high-quality local Python via AST, inject structural references into an LLM prompt, iteratively refine generated/refactored code, and quality-gate with static analysis before accepting output.

### Tech stack and architecture

| Layer | Components |
|-------|------------|
| **Language** | Python 3.11+ (`tomllib`, `ast.unparse`, modern typing) |
| **Validation** | Pydantic v2 throughout |
| **LLM** | Ollama `/api/chat` (httpx async) via `LocalLLMClient` |
| **Embeddings** | Ollama `/api/embed` → `SemanticIndex` (wired; fail-soft) |
| **UI** | Gradio thin adapter (`app.py`) |
| **Quality gate** | ruff + mypy; Docker image `sandbox_image/` or local subprocesses |
| **Config** | TOML + Pydantic (`dica/config.py`) — used at runtime |
| **Engine** | Shared `dica.pipeline.PipelineEngine` |

### Entry points / data flow (current)

```
reference_corpus/*.py
        │
        ▼
   CodeVault.ingest  ──► SemanticIndex.build (optional)
        │                        │
        ▼                        ▼
   IntentDispatcher.dispatch (lexical + structural + semantic)
        │
        ▼
   PromptOrchestrator → render_budgeted(ContextBudget)
        │
        ▼
   LocalLLMClient.complete
        │
        ▼
   extract_python_code → unwrap (ok=True only)
        │
        ▼
   multi-pass refine + cloud_polish stub
        │
        ▼
   verify (ruff + mypy) → correction loop
```

Adapters: **`main.py`** (CLI) and **`app.py`** (Gradio) both drive `PipelineEngine.run` event streams.

### Critical components

1. **Extraction + syntax gates** — decide what code advances  
2. **Sandbox** — only real quality signal  
3. **Dispatch/retrieval** — hybrid ranking  
4. **Orchestrator + ContextBudget** — prompt contract under `num_ctx`  
5. **PipelineEngine** — single control plane for CLI and UI  

---

## Detailed Findings

### Critical Issues

#### C1. Extraction success ignores `ok` — invalid code skips format retries

- **Severity:** Critical  
- **Status:** **Done**

**Original problem:** `unwrap_extraction` returned any non-empty `code` even when `ok=False`, so unparseable fences skipped format retries.

**Resolution:** `unwrap_extraction` requires `result.ok` and non-empty code. Generation path uses `failure_prompt` and the broken snippet for format-correction payloads. Covered by `tests/test_extraction.py` and `tests/test_pipeline.py`.

---

### High Priority Issues

#### H1. `ContextBudget` never applied — context overflow risk

- **Severity:** High  
- **Status:** **Done**

**Resolution:** `ContextBudget.assemble_recency` + `PromptPayload.render_budgeted`. Diagnostics live on `PromptPayload.diagnostics` with middle-out truncation; oversized failing code capped in `build_correction(..., budget=)`. Pipeline emits budget telemetry per model call. Tests in `tests/test_context.py`, `tests/test_orchestrator.py`.

#### H2. Semantic retrieval module is dead code

- **Severity:** High  
- **Status:** **Done**

**Resolution:** Engine builds `SemanticIndex` when `semantic_weight > 0`; dispatcher hybrid total is  
`lexical_weight * lex + structural + semantic_weight * cosine`. Unavailable Ollama/embed degrades to lexical+structural. Tests with fake index in `tests/test_dispatcher.py`.

#### H3. Dual pipeline implementations (CLI vs Gradio)

- **Severity:** High  
- **Status:** **Done**

**Resolution:** `dica/pipeline.py` owns the lifecycle; `main.py` / `app.py` are thin adapters over `PipelineEvent` streams.

#### H4. Configuration largely unused by runtime

- **Severity:** High  
- **Status:** **Done**

**Resolution:** Runtime uses `get_config()` for Ollama host/model/timeouts/`num_ctx`, dispatch `top_k`/weights/boosts, engine retries, sandbox cleanup config. CLI `--model` / `--max-attempts` default from TOML and remain overridable. `dispatch.top_k` default aligned to **3** for multi-pass schedule.

#### H5. No automated tests

- **Severity:** High  
- **Status:** **Done** (local suite; CI still open as L6)

**Resolution:** `pytest` suite under `tests/` (37 tests). See README **Testing** section: `pip install -r requirements-dev.txt && pytest`.

#### H6. Incomplete / misleading dependency surface

- **Severity:** High  
- **Status:** **Done**

**Resolution:**

| File | Contents |
|------|----------|
| `requirements.txt` | Core CLI |
| `requirements-ui.txt` | + Gradio |
| `requirements-sandbox.txt` | + Docker SDK |
| `requirements-dev.txt` | + pytest |

#### H7. Pass 0 has no syntax regression gate

- **Severity:** High  
- **Status:** **Done**

**Resolution:** After Pass 0, `syntax_regression` abort on failure; refinement passes still roll back. Documented in pipeline module docstring.

---

### Medium Priority Issues

#### M1. Local sandbox timeout may orphan checker processes — **Done**

- **Resolution:** `_local_verify` tracks both checker processes, applies a shared `wait_for` budget, and on timeout calls `_kill_process` (kill + wait) for each. Cancelled collectors also kill their process. Covered by `tests/test_sandbox.py`.

#### M2. Correction diagnostics unbounded — **Done** (via H1)

#### M3. Orphan container sweep is broad — **Done**

- **Resolution:** `cleanup_orphaned_containers` matches only `dica.sandbox` label and `dica_sandbox_` name prefix. Image-ancestor filtering removed so manual containers from the sandbox image are not force-deleted.

#### M4. Vault scoring is not Jaccard as documented — **Done**

- **Resolution:** `CodeVault.search` scores ``|query ∩ keywords| / |query ∪ keywords|``.

#### M5. Empty / stopword-only queries yield empty dispatch — **Done**

- **Resolution:** `IntentDispatcher` falls back to a tag-richness ranking (typing, docstring, decorators, async, pydantic, class) when there are no retrieval tokens or no lexical/semantic hits.

#### M6. Package / import design — **Partial**

- Pipeline is in `dica/`; full `pyproject.toml` + console scripts still recommended  

#### M7. `distill_syntax` drops comments — **Open** (document or preserve)

#### M8. Dockerfile path mismatch — **Done**

#### M9. httpx client per completion — **Open**

#### M10. Pydantic `"Field"` false positives — **Open**

---

### Low Priority / Improvement Opportunities

| ID | Status | Note |
|----|--------|------|
| L1 | Open | `scripts/*.py.py` naming |
| L2 | **Done** | UI no longer hardcodes Phi-4 |
| L3 | Open | `cloud_polish_enabled` config flag |
| L4 | Open | Chunk ID hash length / content hash |
| L5 | Open | LICENSE file |
| L6 | Open | `pyproject.toml`, pre-commit, CI |
| L7 | Open | Document Gradio bind exposure risk |
| L8 | **Partial** | Tests clear cache; brief docs still useful |
| L9 | Open | Shared checker args host vs container |

---

## Strengths & Well-Engineered Areas

1. Clear modular boundaries and “why” docstrings  
2. Immutable Pydantic models for chunks, payloads, reports  
3. Production-minded extraction (unclosed fences, salvage, parse gate)  
4. Docker sandbox hardening (network off, caps, non-root, orphan sweep)  
5. Fail-soft cloud polish, embeds, vault skip, Docker→local fallback  
6. Async discipline (`to_thread`, subprocesses, `run_coro_blocking`)  
7. `SupportsComplete` protocol for testable LLM injection  
8. Intermediate `ast.parse` rollback + Pass 0 abort  
9. **(Post-remediation)** Shared engine, budgeted prompts, hybrid dispatch, pytest baseline  

---

## Architectural & Design Recommendations

| # | Recommendation | Status |
|---|----------------|--------|
| 1 | Single `dica/pipeline.py` | **Done** |
| 2 | Use config root object at runtime | **Done** (REFINEMENT_PASSES / FORMAT_RETRIES still module constants — acceptable) |
| 3 | Wire semantic side-table | **Done** |
| 4 | Prompt assembly owns budgeting | **Done** (`render_budgeted`) |
| 5 | Proper packaging (`pyproject.toml`) | **Open** |
| 6 | Separate demo/messy scripts under `examples/` | **Open** |

---

## Testing Strategy Recommendations

| Layer | Target | Status |
|-------|--------|--------|
| Unit | Extraction / unwrap `ok` | **Done** |
| Unit | Vault ingest / search | **Done** |
| Unit | Dispatcher hybrid scores | **Done** |
| Unit | Distill / constraints / correction | **Done** |
| Unit | ContextBudget assemble / truncate | **Done** |
| Unit | `_parse_runner_output` | **Done** |
| Integration | Fake LLM through `PipelineEngine` | **Done** (scripted client) |
| Optional e2e | Docker verify clean/dirty snippets | **Open** |
| Property | AST distill/extract round-trips | **Open** |
| CI | Run pytest on push/PR | **Done** (`.github/workflows/ci.yml`) |

---

## Security Assessment Summary

| Area | Assessment |
|------|------------|
| **Secrets** | No hardcoded API keys; cloud polish stub only. Good for local-first. |
| **Sandbox (Docker)** | Strong isolation for static analysis. |
| **Sandbox (local)** | Host checkers — trusted local use only. |
| **Gradio** | Loopback bind; no auth — fine locally. |
| **Config host** | Ollama host now configurable — consider allowlist if ever multi-tenant. |
| **Orphan cleanup** | Label + name prefix only (M3 fixed). |
| **Supply chain** | Unpinned mins; prefer lockfile later. |

Overall: **reasonable local security**; not multi-user-ready without auth, rate limits, and mandatory Docker backend.

---

## Performance & Scalability Observations

- In-memory vault fine for small corpora; vector-DB swap still the scale path  
- CLI re-ingests per run; UI indexes once — unchanged  
- Multi-pass LLM remains the latency bottleneck  
- Semantic index batches embeds with concurrency cap — **now active when weight > 0**  
- Token heuristic **now applied** via budgeted prompt assembly  
- Remaining: connection-pool LLM client (M9)  


---

## Maintainability & Technical Debt Assessment

| Debt | Severity | Status |
|------|----------|--------|
| Dual engines | High | **Resolved** |
| Unwired embeddings/context/config | High | **Resolved** |
| Script-as-library | Medium | **Partial** (pipeline in package; no pyproject yet) |
| Magic constants vs TOML | Medium | **Mostly resolved** |
| No tests / CI | High | **Tests + CI workflow done; pyproject still open** |
| Duplicated ruff/mypy args | Medium | **Open** (L9) |
| Incomplete requirements | Medium | **Resolved** |

---

## Actionable Prioritized Roadmap

### Completed in remediation

1. ~~Fix `unwrap_extraction` / format retries~~  
2. ~~Pass 0 syntax gate~~  
3. ~~CLI/UI config defaults from `config.toml`~~  
4. ~~Dependency split (core / UI / sandbox / dev)~~  
5. ~~Dockerfile path comment~~  
6. ~~Shared `dica/pipeline.py`~~  
7. ~~Wire `get_config()` end-to-end~~  
8. ~~Wire `ContextBudget`~~  
9. ~~Wire `SemanticIndex`~~  
10. ~~Add `tests/` + pytest~~  

### Still recommended

11. ~~**M1** — Kill local checker subprocesses on sandbox timeout~~ **Done**  
12. ~~**M3** — Narrow orphan container filters~~ **Done**  
13. ~~CI workflow for `pytest`~~ **Done** (`.github/workflows/ci.yml`)  
14. **`pyproject.toml`** + console scripts; optional pre-commit  
15. ~~**M4/M5** — Honest Jaccard + empty-query fallback~~ **Done**  
16. **M9** — Long-lived `httpx.AsyncClient` on `LocalLLMClient`  
17. **L5/L9** — LICENSE; shared checker arg module for host + container runner  

---

## Conclusion

### At original review

DICA showed strong module-level design with incomplete integration: budget, semantics, and config were spectators; two engines drifted; extraction `ok` was ignored. Score **7.0/10**.

### After remediation

Critical and high-priority gaps are closed. The system now has one control plane, live configuration, budgeted prompts, hybrid retrieval, and a regression suite operators can run offline. Estimated quality **~8.5/10**. Remaining items are operational polish, packaging/CI, and edge-case retrieval honesty—not foundational architecture rewrites.

---

## Limitations of This Analysis

- Original review did not run full e2e against a live Ollama model or Docker daemon for every path.  
- Remediation status reflects code and tests as of commit **`ce9739c`** on `main`; did not re-audit every medium/low item line-by-line after the fact.  
- Did not perform dependency CVE scanning or load tests.  
- Gradio multi-user concurrency still reasoned about statically.  
- Property-based tests and CI automation remain future work.  
