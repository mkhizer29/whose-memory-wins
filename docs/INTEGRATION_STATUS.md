# Integration status (what was actually tested)

Checked on 2026-11-03, Windows 11, Python 3.12.10, inside the project venv.

## Verified

| Component | Version | How verified |
|---|---|---|
| Mem0 open source (`mem0ai`) | 2.2.1 | API read from the installed source. `add(infer=False)`, `get`, `get_all`, `search`, `update`, `delete`, `history` exercised in tests against real local storage |
| Mem0 normal ingestion pipeline (`add(infer=True)`) | 2.2.1 | Full pipeline exercised with a **mock** extraction LLM swapped in for `Memory.llm` (context gathering, parsing, embedding, storage). Not yet run with a real LLM |
| Local vector store | qdrant-client 1.20.0, local on-disk mode | Used by every fixture. No server or Docker needed |
| Embeddings | fastembed 0.9.1, `BAAI/bge-small-en-v1.5` (384-d, ONNX on CPU) | Used by every fixture. Retrieval re-checked deterministic (live search equals frozen snapshot) |
| Anthropic SDK | anthropic 2.4.0 (sends API version 2026-10-21) | Request shape written from the bundled SDK documentation and checked against the installed SDK's signatures and exception classes. **No live call made** (no credentials available) |
| Streamlit dashboard | 1.66.1 | All seven pages rendered in a headless test run without exceptions |
| Offline test suite | pytest 9.2.0 | `50 passed, 1 skipped` (the skip is the real-API smoke test) |
| Full 648-trial pipeline | – | A 648-trial MOCK run: created, stopped after 200, resumed to 648 with no duplicates, then analyzed |

The exact environment is pinned in `requirements.lock` (output of `uv pip freeze`). The direct
dependencies are pinned in `pyproject.toml`.

## NOT verified (needs your credentials)

1. **Real decision model calls** (`claude-opus-5-5` via the Anthropic API). The adapter is
   implemented (strict tools, `tool_choice` auto with parallel tool use disabled,
   `output_config.effort`, refusal handling, transient/permanent error classification), but it
   has never been called. First real step: `wmw check-config --ping`.
2. **Mem0 normal ingestion with a real LLM** (Mem0's `anthropic` LLM provider with
   `enable_sampling_parameters: False`, because API version 2026-10-21 rejects `temperature`).
   Whether Mem0 2.2.1's Anthropic wrapper parses Claude Opus 5.5's output into memories reliably
   is unverified. Ingestion errors are recorded, never replaced with mock output.
3. **The real-API pytest smoke test** (`tests/test_real_smoke.py`). It runs only with
   `ANTHROPIC_API_KEY` and `WMW_RUN_REAL_TESTS=1`.

## Settings that could not be pinned

* `temperature`/`top_p`/`top_k`: rejected by the API version that anthropic 2.4.0 sends, so
  provider-default sampling is used and recorded as such in every manifest.
* Claude Opus 5.5 always uses adaptive thinking. The depth is set with `effort` (`medium` by
  default in `configs/default.yaml`). Thinking content is not requested and never used.
* Token usage of Mem0's internal LLM calls is not exposed by Mem0's wrapper, so Track 2 counts
  calls only.

## Not used / not installed

* spaCy (`mem0ai[nlp]`) is not installed, so Mem0's entity-boost and lemmatization features are
  inactive (Mem0 logs a warning). Retrieval therefore uses semantic similarity plus BM25 on
  unlemmatized text. This is recorded in each fixture manifest (`spacy_installed: false`).
* Mem0 telemetry is disabled (`MEM0_TELEMETRY=False`).
* The Anthropic server-side refusal fallback is deliberately not enabled, because it would switch
  models mid-experiment.
