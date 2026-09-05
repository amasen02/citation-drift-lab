What happens to an agent answer when its documentation changes?

`citation-drift-lab` makes the failure reproducible offline. It answers from Markdown with exact quotes, source hashes, heading paths, and character offsets, then audits the receipt against a new version.

The included fixture changes a 30-record limit to 20, removes an incident exception, and moves an unchanged audit-export quote. The measured audit reports `changed` (material), `deleted`, and `relocated`, with two affected claims. An unsupported question abstains after a bounded revision trace.

Run it with Python 3.11+ and no model or network service:

`python -m venv .venv` → install editable → `.venv\Scripts\citation-drift-lab demo --output-dir demo-output`

The project favors reproducible provenance over polished generation. Its bounded graph follows the [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api). Exact quote equality still cannot prove semantic entailment; human review remains part of the workflow.

Built with AI assistance from a user-directed brief; code, tests, receipts, and limitations are included. Repository: [citation-drift-lab](https://github.com/amasen02/citation-drift-lab)
