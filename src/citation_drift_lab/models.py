"""Validated public data contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpanModel(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_half_open_span(self):
        if self.start >= self.end:
            raise ValueError("character offsets must be a non-empty half-open span")
        return self


class ParentSection(SpanModel):
    section_id: NonEmpty
    heading_path: list[NonEmpty]
    text: NonEmpty


class Chunk(SpanModel):
    chunk_id: NonEmpty
    document_id: NonEmpty
    version_hash: NonEmpty
    source_path: NonEmpty
    heading_path: list[NonEmpty]
    text: NonEmpty


class SourceVersion(StrictModel):
    document_id: NonEmpty
    version_hash: NonEmpty
    source_path: NonEmpty
    source_text: str
    parent_sections: list[ParentSection]
    chunks: list[Chunk]

    @model_validator(mode="after")
    def spans_match_source(self):
        expected_hash = sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.version_hash != expected_hash:
            raise ValueError("version_hash must be the SHA-256 hash of source_text")
        for item in [*self.parent_sections, *self.chunks]:
            if self.source_text[item.start : item.end] != item.text:
                item_id = item.chunk_id if isinstance(item, Chunk) else item.section_id
                raise ValueError(f"span for {item_id} is not exact")
        for index, chunk in enumerate(self.chunks):
            for field in ("document_id", "version_hash", "source_path"):
                if getattr(chunk, field) != getattr(self, field):
                    raise ValueError(f"chunk {index}: {field} must match its source version")
        return self


class Evidence(SpanModel):
    document_id: NonEmpty
    version_hash: NonEmpty
    source_path: NonEmpty
    heading_path: list[NonEmpty]
    quote: NonEmpty
    chunk_id: NonEmpty
    score: float = Field(ge=0)
    retrieval_component: Literal["lexical_bm25", "hybrid_bm25_ollama"]

    @model_validator(mode="after")
    def quote_matches_span_length(self):
        if self.end - self.start != len(self.quote):
            raise ValueError("evidence quote length must equal its character span")
        return self


class Citation(SpanModel):
    document_id: NonEmpty
    version_hash: NonEmpty
    source_path: NonEmpty
    heading_path: list[NonEmpty]
    quote: NonEmpty
    chunk_id: NonEmpty

    @model_validator(mode="after")
    def quote_matches_span_length(self):
        if self.end - self.start != len(self.quote):
            raise ValueError("citation quote length must equal its character span")
        return self


class Claim(StrictModel):
    claim_id: NonEmpty
    text: NonEmpty
    citations: list[Citation] = Field(min_length=1)


class AnswerRecord(StrictModel):
    answer_id: NonEmpty
    question: NonEmpty
    response: NonEmpty
    mode: Literal["extractive_offline", "ollama_chat_unverified", "abstained"]
    claims: list[Claim]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    integrity_note: NonEmpty = (
        "Exact quote matching verifies lexical integrity, not semantic entailment."
    )

    @model_validator(mode="after")
    def mode_matches_claims(self):
        if self.mode == "abstained" and self.claims:
            raise ValueError("an abstained answer cannot contain claims")
        if self.mode != "abstained" and not self.claims:
            raise ValueError("a supported answer must contain at least one claim")
        return self


class GraphConfig(StrictModel):
    top_k: int = Field(default=3, ge=1, le=20)
    max_revisions: int = Field(default=1, ge=0, le=3)
    evidence_threshold: float = Field(default=0.12, ge=0, le=1)
    ollama_base_url: str | None = None
    embedding_model: NonEmpty | None = None
    chat_model: NonEmpty | None = None
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)

    @model_validator(mode="after")
    def local_provider_only(self):
        if (self.embedding_model or self.chat_model) and not self.ollama_base_url:
            raise ValueError("ollama_base_url is required when an Ollama model is configured")
        if self.ollama_base_url:
            parsed = urlsplit(self.ollama_base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
            ):
                raise ValueError(
                    "Ollama URL must be an explicit credential-free loopback HTTP(S) URL"
                )
        return self


class GraphRequest(StrictModel):
    question: NonEmpty
    sources: list[SourceVersion] = Field(min_length=1)
    config: GraphConfig = Field(default_factory=GraphConfig)

    @model_validator(mode="after")
    def unique_source_document_ids(self):
        seen: set[str] = set()
        for source in self.sources:
            if source.document_id in seen:
                raise ValueError(f"duplicate source document_id: {source.document_id}")
            seen.add(source.document_id)
        return self


class GraphResult(StrictModel):
    answer: AnswerRecord
    evidence: list[Evidence]
    abstained: bool
    revision_count: int = Field(ge=0, le=3)
    node_trace: list[NonEmpty]
    retrieval_component: Literal["lexical_bm25", "hybrid_bm25_ollama"]


class AuditFinding(StrictModel):
    finding_id: NonEmpty
    claim_id: NonEmpty
    document_id: NonEmpty
    status: Literal["unchanged", "relocated", "changed", "deleted", "source_missing"]
    affects_claim: bool
    material_change: bool
    old_hash: NonEmpty
    new_hash: str | None = None
    old_start: int = Field(ge=0)
    old_end: int = Field(gt=0)
    new_start: int | None = Field(default=None, ge=0)
    new_end: int | None = Field(default=None, gt=0)
    old_quote: NonEmpty
    candidate_quote: str | None = None
    heading_path: list[NonEmpty]
    explanation: NonEmpty


class AuditReport(StrictModel):
    answer_id: NonEmpty
    findings: list[AuditFinding]
    affected_claim_ids: list[NonEmpty]
    integrity_note: NonEmpty = (
        "Quote presence is not entailment; an unchanged quote can acquire "
        "different surrounding meaning."
    )
