# When a citation survives but the answer does not

An answer can contain a real quote and still become unsafe to reuse after its source changes. A citation that points at a line number or a document URL is not a durable statement of what the answer relied on. The source may change around it, move the text, alter a number, or remove an exception.

`citation-drift-lab` is a small, offline way to make that failure reproducible. It ingests Markdown, keeps the source version and exact character offsets, answers with extractive claims, and audits those claims against a later source version. The project is deliberately model-free by default, so the interesting evidence is in the receipt rather than in a model’s prose.

## The fixture failure

The demo starts with `examples/fixtures/policy-v1.md`. It contains a 30-record project limit, an incident exception, an audit-export statement, prompt-like text, and an HTML-looking heading. Version 2 changes the limit to 20, removes the exception, moves the audit-export section, and edits unrelated text.

Run it in a fresh environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\citation-drift-lab demo --output-dir demo-output
```

On POSIX, use `.venv/bin/python` and `.venv/bin/citation-drift-lab`. The command needs no API key, model, or network service. It writes `answer.json`, `audit.json`, `report.html`, and an unsupported-question receipt.

The committed receipt has three claims. The audit classifies them as follows:

* The project limit is `changed` and `material_change` is true: the cited text says 30 in the answer and 20 in the updated section.
* The incident exception is `deleted`, and it affects the claim.
* The audit-export quote is `relocated`: the exact quote remains, but its offsets changed, so the claim is still lexically valid.

The audit reports two affected claim IDs. The unsupported question produces `abstained`, with a bounded trace containing one revision and a final `abstain` node. This is useful behavior for an agent pipeline: insufficient evidence becomes a recorded outcome instead of an invented answer.

## What the program records

Markdown ingestion creates a source version with a SHA-256 hash. Chunks retain ATX heading ancestry and half-open Python string offsets. A citation stores the document ID, source hash, path, heading path, chunk ID, quote, and offsets. The quote can therefore be checked by slicing the original source string. The hash verifies that the supplied text matches the receipt; it does not establish who authored the file or whether the source is trustworthy.

The graph is a compiled [LangGraph `StateGraph`](https://docs.langchain.com/oss/python/langgraph/graph-api) with prepare, retrieve, validate, answer, revise, and abstain paths. The default retriever is named `lexical_bm25`: deterministic BM25-like token scoring followed by a lexical coverage threshold. A bounded revision counter and graph recursion limit keep an unsupported request finite.

The default answer mode is extractive. It cites the selected section rather than asking a language model to paraphrase it. Optional `hybrid_bm25_ollama` retrieval can fuse lexical and cosine ranks through Ollama’s local embedding endpoint. Optional Ollama chat generation is labelled `ollama_chat_unverified`; its citations identify supplied evidence, but the project does not claim semantic entailment or sentence-level attribution validation.

The audit searches for the exact quote globally first. If it is absent, it compares likely sections using heading-aware sequence similarity. Numeric and negation changes flag likely material changes. These rules explain the fixture’s changed, deleted, and relocated statuses, but they are heuristics intended to support human review.

## Why the boundaries matter

The ingestion boundary treats Markdown as data. Prompt-like text is preserved as quoted evidence and is never executed. Strict Pydantic models reject unknown fields and invalid spans. Provider URLs must be explicit credential-free loopback HTTP(S) URLs; provider calls have timeouts and fail explicitly. Static HTML escapes dynamic values, so the fixture’s `<script>` heading is displayed as text.

Those controls answer different questions. Exact quote matching asks whether the evidence is still present. Hash verification asks whether a receipt describes the supplied source bytes. Neither answers whether a surviving sentence still supports the same interpretation in its new context. A stable quote can acquire different meaning when surrounding paragraphs change. That is why the report says lexical integrity is not semantic entailment.

There are costs. ATX parsing supports a practical subset of Markdown, not all CommonMark. Chunks can be long. Lexical retrieval is deterministic but is not a relevance benchmark. Duplicate quotes resolve to the first exact occurrence. Drift classifications and materiality flags require a person before policy or documentation action.

## Exercise for the owner

Start by inspecting `answer.json`, then verify one citation with `source_text[start:end]`. Change `30` to `20`, remove the incident section, move the audit-export section, and run `audit` yourself. Read each finding before opening `docs/interview.md`. Then add a test where the exact quote survives but a surrounding sentence reverses its meaning. That exercise exposes the boundary this lab intentionally leaves open: provenance can be reproducible while interpretation still needs judgment.

The repository was built with AI assistance from a user-directed brief and an agent-developed plan. The implementation, tests, model-free receipts, and stated limitations are available for inspection. The fresh verification recorded 29 passing tests; semantic answer quality and live model quality were not measured.

Repository: [citation-drift-lab](https://github.com/amasen02/citation-drift-lab).
