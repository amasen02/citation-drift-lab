"""Markdown ingestion with exact source spans."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import Chunk, ParentSection, SourceVersion

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*(?:\n|$)", re.MULTILINE)


def _document_id(path: Path) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return value or "document"


def ingest_markdown(path: str | Path, document_id: str | None = None) -> SourceVersion:
    """Read one Markdown file without interpreting any of its content as instructions."""
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8", newline="") as source_file:
        text = source_file.read()
    doc_id = (document_id or _document_id(source_path)).strip()
    if not doc_id:
        raise ValueError("document_id must not be empty")
    version_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    matches = list(_HEADING.finditer(text))
    spans: list[tuple[int, int, list[str]]] = []
    hierarchy: list[str] = []
    if matches and text[: matches[0].start()].strip():
        spans.append((0, matches[0].start(), ["(document)"]))
    for index, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip().rstrip("#").rstrip()
        hierarchy = hierarchy[: level - 1] + [heading]
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans.append((match.start(), end, list(hierarchy)))
    if not matches and text.strip():
        spans.append((0, len(text), ["(document)"]))
    parents: list[ParentSection] = []
    chunks: list[Chunk] = []
    for index, (start, end, headings) in enumerate(spans):
        segment = text[start:end]
        if not segment.strip():
            continue
        section_id = f"{doc_id}:section:{index}"
        chunk_id = f"{doc_id}:{version_hash[:12]}:{index}"
        parents.append(
            ParentSection(
                section_id=section_id, heading_path=headings, text=segment, start=start, end=end
            )
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                version_hash=version_hash,
                source_path=str(source_path),
                heading_path=headings,
                text=segment,
                start=start,
                end=end,
            )
        )
    return SourceVersion(
        document_id=doc_id,
        version_hash=version_hash,
        source_path=str(source_path),
        source_text=text,
        parent_sections=parents,
        chunks=chunks,
    )
