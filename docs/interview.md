# Interview explanation path

1. **Why source versions?** A document ID identifies the logical source; SHA-256
   identifies the exact content against which an answer was produced.
2. **Why exact spans?** A citation can be checked directly with
   `source_text[start:end] == quote`, independent of a vector store.
3. **Why LangGraph?** The branching and bounded revision loop are explicit runtime
   topology, not hidden control flow dressed up as a graph.
4. **Why BM25-like retrieval?** It is deterministic, inspectable, and runs offline.
   Real Ollama embeddings are optional and honestly labelled.
5. **What does auditing prove?** It proves quote survival or detects likely textual
   drift. It does not prove that a claim follows from the quote.
6. **What happens on model failure?** Configured embedding or chat errors surface as
   provider errors. There is no silent downgrade that could mislabel an answer.
7. **How is prompt injection handled?** Markdown is only tokenized, sliced, cited, and
   optionally placed inside an explicit inert-evidence prompt. It never drives tools.
