# Architecture

```mermaid
flowchart LR
    MD[Untrusted Markdown] --> I[ATX ingestion\nSHA-256 + exact spans]
    I --> P[Pydantic boundaries]
    P --> Q[prepare query]
    Q --> R[retrieve\nlexical_bm25 or hybrid_bm25_ollama]
    R --> V[validate lexical evidence]
    V -->|strong| A[answer\nexact citations]
    V -->|weak, budget left| X[deterministic revise]
    X --> R
    V -->|weak, exhausted| Z[abstain]
    A --> J[answer JSON]
    J --> D[audit against new SourceVersion]
    D --> H[JSON + escaped HTML]
```

`StateGraph(GraphState)` uses a `TypedDict` internally and Pydantic models at every
public boundary. Nodes return state updates. The graph is compiled before invocation;
conditional edges route validation outcomes, and the revision edge loops only while the
validated counter permits it.

Each citation carries document ID, source hash and path, heading path, chunk ID, quote,
and exact half-open offsets. Audit comparisons are per citation: exact quote at the old
offset is unchanged, exact quote elsewhere is relocated, a similar replacement section
is changed, absent content is deleted, and an absent document is source-missing.
