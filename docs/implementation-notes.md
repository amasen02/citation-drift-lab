# Implementation notes

## Design tradeoffs

- ATX section chunks keep offsets easy to verify and heading ancestry clear, at the cost
  of supporting less Markdown syntax and potentially producing long chunks.
- Rank fusion uses reciprocal ranks for deterministic hybrid behavior; it is not trained
  relevance calibration.
- Drift uses exact global quote search first, then heading-aware sequence similarity.
  Numeric and negation deltas flag likely material changes but remain heuristics.
- The extractive default cites the complete selected section. This favors reproducible
  provenance over polished prose.
- Optional Ollama chat generation is labelled `ollama_chat_unverified`. Its citations
  identify supplied evidence, but no semantic entailment or sentence-level attribution
  validation is performed.

## Limitations and interview notes

Lexical quote equality is not entailment. A surviving quote may mean something else in
new surrounding context. Duplicate quotes resolve to the first exact occurrence. Drift
classification should be reviewed by a person before documentation or policy action.
For an interview walkthrough, follow `docs/interview.md` and demonstrate the actual
half-open slice, the bounded graph trace, provider request tests, and escaped report.

## Verification results

Verified locally on Python 3.12.9 on 2026-09-05:

- `python -m ruff format --check .` — 22 files already formatted.
- `python -m ruff check .` — all checks passed.
- `python -m pytest` — 29 tests passed.
- `python -m pip check` — no broken requirements found.
- fresh model-free demo — produced four artifacts; JSON parsed, all three citation spans
  and source hashes matched the supplied fixture, audit statuses included `changed`,
  `deleted`, and `relocated`, affected claims and bounded abstention were recorded, and
  hostile `<script>` markup was HTML-escaped.

The Ollama response-shape tests use `httpx.MockTransport`; they validate the mocked API
contract, never live LLM quality. These are fixture-level functional checks, not
performance or answer-quality benchmarks. Optional Ollama chat generation remains
unverified.

## API references checked

The implementation was checked against the current official documentation on 2026-09-05:

- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) for
  `StateGraph`, state-update nodes, conditional edges, compilation, and invocation.
- [Ollama embeddings API](https://docs.ollama.com/api/embed) for the native
  `POST /api/embed` request and `embeddings` response shape.
- [Ollama chat API](https://docs.ollama.com/api/chat) for the native `POST /api/chat`
  messages request and `message.content` response shape.
- [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/) for strict model
  validation and serialization behavior.
