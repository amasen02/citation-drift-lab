"""Deterministic lexical BM25 and optional real Ollama hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import Chunk, Evidence
from .providers import OllamaClient

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
SYNONYMS = {"keep": "retain retention", "retention": "retain keep", "cap": "limit maximum"}


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    return [token[:-1] if token.endswith("s") and len(token) > 3 else token for token in tokens]


def informative_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1}


def revise_query(query: str) -> str:
    tokens = [token for token in tokenize(query) if token not in STOPWORDS]
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(SYNONYMS.get(token, "").split())
    return " ".join(dict.fromkeys(expanded))


def _bm25(query: str, chunks: list[Chunk]) -> list[float]:
    query_tokens = tokenize(query)
    documents = [tokenize(chunk.text) for chunk in chunks]
    if not documents:
        return []
    average_length = sum(map(len, documents)) / len(documents) or 1.0
    document_frequency = Counter(
        token for token in set(query_tokens) for doc in documents if token in doc
    )
    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse = math.log(
                1
                + (len(documents) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(document) / average_length)
            score += inverse * frequency * 2.5 / denominator
        scores.append(score)
    return scores


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return (
        sum(x * y for x, y in zip(left, right, strict=True)) / denominator if denominator else 0.0
    )


def retrieve(
    query: str,
    chunks: list[Chunk],
    top_k: int,
    embedding_model: str | None = None,
    provider: OllamaClient | None = None,
) -> list[Evidence]:
    lexical = _bm25(query, chunks)
    component = "lexical_bm25"
    fused = lexical[:]
    if embedding_model:
        if provider is None:
            raise ValueError("an Ollama provider is required for embedding retrieval")
        vectors = provider.embed(embedding_model, [query, *[chunk.text for chunk in chunks]])
        semantic = [_cosine(vectors[0], vector) for vector in vectors[1:]]
        lexical_order = sorted(range(len(chunks)), key=lambda i: (-lexical[i], chunks[i].chunk_id))
        semantic_order = sorted(
            range(len(chunks)), key=lambda i: (-semantic[i], chunks[i].chunk_id)
        )
        lexical_rank = {index: rank for rank, index in enumerate(lexical_order, 1)}
        semantic_rank = {index: rank for rank, index in enumerate(semantic_order, 1)}
        fused = [
            1 / (60 + lexical_rank[i]) + 1 / (60 + semantic_rank[i]) for i in range(len(chunks))
        ]
        component = "hybrid_bm25_ollama"
    maximum = max(fused, default=0.0)
    normalized = [score / maximum if maximum else 0.0 for score in fused]
    order = sorted(range(len(chunks)), key=lambda i: (-normalized[i], chunks[i].chunk_id))[:top_k]
    return [
        Evidence(
            document_id=chunks[i].document_id,
            version_hash=chunks[i].version_hash,
            source_path=chunks[i].source_path,
            heading_path=chunks[i].heading_path,
            quote=chunks[i].text,
            start=chunks[i].start,
            end=chunks[i].end,
            chunk_id=chunks[i].chunk_id,
            score=normalized[i],
            retrieval_component=component,
        )
        for i in order
    ]
