"""Independent per-citation drift auditing against new source versions."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .models import AnswerRecord, AuditFinding, AuditReport, Citation, SourceVersion

_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_NEGATION = re.compile(r"\b(?:no|not|never|mustn['’]?t|cannot|can['’]?t|except)\b", re.I)


def _candidate(citation: Citation, source: SourceVersion):
    same_heading = [chunk for chunk in source.chunks if chunk.heading_path == citation.heading_path]
    pool = same_heading or source.chunks
    if not pool:
        return None, 0.0
    ranked = [(SequenceMatcher(None, citation.quote, chunk.text).ratio(), chunk) for chunk in pool]
    similarity, chunk = max(ranked, key=lambda pair: (pair[0], -pair[1].start))
    return chunk, similarity


def _material(old: str, new: str) -> bool:
    return set(_NUMBER.findall(old)) != set(_NUMBER.findall(new)) or bool(
        _NEGATION.search(old)
    ) != bool(_NEGATION.search(new))


def audit_answer(answer: AnswerRecord, new_sources: list[SourceVersion]) -> AuditReport:
    by_id: dict[str, SourceVersion] = {}
    for source in new_sources:
        if source.document_id in by_id:
            raise ValueError(f"duplicate updated document_id: {source.document_id}")
        by_id[source.document_id] = source
    findings: list[AuditFinding] = []
    affected: list[str] = []
    for claim in answer.claims:
        for index, citation in enumerate(claim.citations, 1):
            source = by_id.get(citation.document_id)
            common = dict(
                finding_id=f"{answer.answer_id}:{claim.claim_id}:{index}",
                claim_id=claim.claim_id,
                document_id=citation.document_id,
                old_hash=citation.version_hash,
                old_start=citation.start,
                old_end=citation.end,
                old_quote=citation.quote,
                heading_path=citation.heading_path,
            )
            if source is None:
                finding = AuditFinding(
                    **common,
                    status="source_missing",
                    affects_claim=True,
                    material_change=False,
                    explanation="The cited document is absent from the updated source set.",
                )
            else:
                still_at_old_span = (
                    source.source_text[citation.start : citation.end] == citation.quote
                )
                new_start = (
                    citation.start if still_at_old_span else source.source_text.find(citation.quote)
                )
                if new_start >= 0:
                    status = "unchanged" if still_at_old_span else "relocated"
                    finding = AuditFinding(
                        **common,
                        status=status,
                        affects_claim=False,
                        material_change=False,
                        new_hash=source.version_hash,
                        new_start=new_start,
                        new_end=new_start + len(citation.quote),
                        candidate_quote=citation.quote,
                        explanation=(
                            "The exact cited quote is still present; lexical validity is preserved."
                        ),
                    )
                else:
                    candidate, similarity = _candidate(citation, source)
                    same_heading_survives = any(
                        chunk.heading_path == citation.heading_path for chunk in source.chunks
                    )
                    threshold = 0.35 if same_heading_survives else 0.65
                    changed = candidate is not None and similarity >= threshold
                    candidate_text = candidate.text if candidate else None
                    finding = AuditFinding(
                        **common,
                        status="changed" if changed else "deleted",
                        affects_claim=True,
                        material_change=_material(citation.quote, candidate_text or "")
                        if changed
                        else False,
                        new_hash=source.version_hash,
                        new_start=candidate.start if changed else None,
                        new_end=candidate.end if changed else None,
                        candidate_quote=candidate_text if changed else None,
                        explanation=(
                            "The exact quote disappeared and the closest section changed."
                            if changed
                            else "The exact quote and a sufficiently similar section are absent."
                        ),
                    )
            findings.append(finding)
            if finding.affects_claim and claim.claim_id not in affected:
                affected.append(claim.claim_id)
    return AuditReport(answer_id=answer.answer_id, findings=findings, affected_claim_ids=affected)
