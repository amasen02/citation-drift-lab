# citation-drift-lab

`citation-drift-lab` is a local, source-version-aware Markdown RAG agent and citation
drift auditor. It answers with exact character-span citations, stores an answer receipt,
then checks every citation independently when the documentation changes.

## One-command offline demo

After installing Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\citation-drift-lab demo --output-dir demo-output
```

On POSIX systems, replace `.venv\Scripts\` with `.venv/bin/`. The demo requires no key,
model, or network service. It writes `answer.json`, `audit.json`, `report.html`, and an
unsupported-question receipt. Committed copies live in `examples/demo-receipts/`.

## Commands

```powershell
citation-drift-lab ask --source examples/fixtures/policy-v1.md --document-id team-policy --question "What is the project record limit?" --output answer.json
citation-drift-lab audit --answer answer.json --source examples/fixtures/policy-v2.md --document-id team-policy --output audit.json --html report.html
citation-drift-lab demo --output-dir demo-output
```

Without `--output`, JSON is written to stdout. Output paths may not overwrite input
files, and audit `--output` and `--html` must resolve to different paths. Invalid input
produces a concise nonzero error without a traceback.

## Architecture and retrieval

Markdown is ingested as immutable-ish Pydantic v2 boundary models. SHA-256 identifies
the full source version; chunks retain ATX heading ancestry and exact Python string
offsets. A compiled LangGraph `StateGraph` performs prepare → retrieve → validate and
conditionally answer, revise and loop, or abstain. An explicit counter (0–3) and graph
recursion limit bound revision.

The offline component is honestly named `lexical_bm25`: deterministic BM25-like token
scoring, followed by a lexical coverage threshold. When explicitly configured,
`hybrid_bm25_ollama` calls Ollama's native `/api/embed` and deterministically fuses
lexical and cosine-similarity ranks. Optional Ollama chat generation calls native
`/api/chat` and is explicitly unverified. Its citations identify the supplied evidence;
no semantic entailment or sentence-level attribution validation is performed. There are
no hash “embeddings,” and network access is opt-in only.

See [architecture](docs/architecture.md), [demo guide](docs/demo.md), and
[implementation notes](docs/implementation-notes.md).

## Security posture

- Source Markdown is never executed. The optional chat prompt labels it as untrusted
  evidence; this does not prove that a model will resist prompt injection.
- Ollama URLs must be explicit credential-free HTTP(S) loopback URLs using `localhost`,
  `127.0.0.1`, or `[::1]`; query strings and fragments are rejected.
- Provider calls have timeouts and fail explicitly—chat never silently falls back.
- Static report values are escaped with `html.escape`.
- Strict Pydantic models reject extra fields and invalid spans.

Please report security issues using [SECURITY.md](SECURITY.md).

## Limitations

Exact quote presence proves lexical integrity, not semantic entailment. Citations
identify supplied evidence but do not validate semantic entailment or sentence-level
attribution. Surrounding meaning can change while a quote remains identical. Markdown
parsing intentionally supports ATX headings, not the entire CommonMark syntax. Lexical
retrieval is small and deterministic, not a relevance benchmark. Drift similarity and
numeric/negation flags are practical heuristics and require human review.

## Owner exercise and explanation path

Start with `examples/fixtures/policy-v1.md`, run `ask`, then explain the citation's
half-open offsets by slicing the original Python string. Change `30` to `20`, remove the
incident section, move the audit-export section, and run `audit`. Walk through each
finding before reading `docs/interview.md`; then add a test for a context change where
the exact quote survives but its meaning does not.

## Construction disclosure

This repository was constructed with AI assistance from a user-directed brief and
an agent-developed implementation plan. Tests, model-free receipts, and limitations are
included so reviewers can verify behavior instead of trusting generation claims.

Copyright © 2026 Ama Senevirathne. MIT licensed.
