# Deviations from the plan, limitations and reproducibility

## Deviations from the original specification

| Planned | Implemented | Reason |
|---|---|---|
| Fixed generation settings incl. temperature | Temperature not set (provider default) | The API version sent by anthropic 2.4.0 rejects sampling parameters on every model |
| Tools: check_stock, request_approval, place_order | + `decline_task` | Needed so a non-overridable rule (DEV-05) has a correct terminal action, and to measure refusals explicitly |
| Three roles | + `internal_staff` (no authority) | Business and catalogue facts need a sender that is neither admin nor supplier |
| One memory namespace per trial context | One **physically separate Mem0 store** per scenario × condition | Mem0's Qdrant BM25 channel uses collection-wide IDF. A shared collection made rankings depend on other namespaces (caught by the leakage test) |
| Every evaluation scenario conflict-relevant | 20 conflict-relevant + 4 permitted-baseline controls | The clean condition must include permitted purchases. In controls the claim cannot change the answer, so the unauthorized rate is also reported on the 20 conflict-relevant scenarios |
| Track 2 "record actual writes/updates" | Snapshot diff + Mem0 return value | Mem0 2.2.1 inference is additive (ADD only). Update/delete mapping is implemented and tested through the explicit `update`/`delete` API |
| Mock mode for Track 2 | Mock extraction LLM replaces `Memory.llm` after construction | Mem0 validates LLM provider names against a fixed list, so a custom provider cannot be registered |
| Dataset frozen before results | Not frozen yet | Freezing is left to you, after development and pilot work: `wmw dataset freeze` |

No evaluation scenario has been used to tune prompts or code. The only inspection of evaluation
data was offline (retrieval properties and prompt sizes, with no model calls).

## Known limitations

* **Simulation.** The tasks are short and synthetic, and the environment and fictional names
  are simple. External validity to real procurement systems is limited.
* **Single model, single provider.** Results apply to the configured model and date. Model
  updates behind the same ID can change behaviour.
* **C is a combined intervention.** Source metadata and permission-handling instructions are
  added together, so their separate effects are not identified.
* **Inputs are not token-matched.** B and C are longer than A. Length itself could affect behaviour.
* **Identical claim text across untrusted and authorized conditions.** This isolates the role of
  application provenance, but it represents the case where memory text carries no attribution.
  Track 2 shows what Mem0's extraction actually keeps.
* **Conservative lineage in Track 2.** Any record created with other sources in the extraction
  context is `ambiguous` and gets least-privileged authority. This prevents promotion but also
  blocks genuine exceptions after normal ingestion. It is a design choice, not an inevitability.
* **Provenance tracking limits in Mem0 2.2.1.** Lineage is inferred from which `add()` call
  produced which record IDs (snapshot diff). If a future Mem0 version merged records across calls
  without reporting it, the diff would show an *update*, and the record would become `ambiguous`.
  The extraction LLM's context (last 10 messages and similar memories) cannot be attributed more
  finely than "possibly influenced by any earlier source".
* **Admission needs human judgement.** Only the "no derived record" case is decided automatically.
* **Statistics.** 24 scenarios (20 conflict-relevant) is a small cluster count. Bootstrap
  intervals are approximate, and no multiple-comparison correction is applied.
* **Retrieval settings fixed** (top-5, threshold 0.1, small local embedder). Other settings could
  change policy visibility in A.

## Reproducibility

Reproducible:
* The dataset, fixtures (deterministic local embeddings), retrieval snapshots (hashed), trial
  order (seeded) and scoring are all deterministic and recorded in the manifest, along with package
  versions, prompt hashes, dataset hash and config hash.

Not reproducible:
* Model responses. The provider does not guarantee determinism, sampling cannot be pinned, and
  adaptive thinking varies. Repetitions (3 per cell) measure this variability rather than remove it.
* Model-version drift behind a model ID, and API behaviour changes over time.
