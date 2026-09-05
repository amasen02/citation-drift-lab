"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .audit import audit_answer
from .demo import run_demo
from .graph import ask
from .ingest import ingest_markdown
from .models import AnswerRecord, GraphConfig, GraphRequest
from .providers import ProviderError
from .report import render_html


def _different_output(output: Path | None, inputs: list[Path]) -> None:
    if output and any(output.resolve() == item.resolve() for item in inputs):
        raise ValueError("refusing to overwrite an input file; choose a different --output path")


def _emit(value: str, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(value, encoding="utf-8")
    else:
        print(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citation-drift-lab", description="Local Markdown RAG and citation drift auditor"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ask_parser = commands.add_parser("ask", help="answer from Markdown sources")
    ask_parser.add_argument("--source", action="append", required=True, type=Path)
    ask_parser.add_argument("--document-id", action="append")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--output", type=Path)
    ask_parser.add_argument("--top-k", type=int, default=3)
    ask_parser.add_argument("--max-revisions", type=int, default=1)
    ask_parser.add_argument("--ollama-url")
    ask_parser.add_argument("--embedding-model")
    ask_parser.add_argument("--chat-model")
    audit_parser = commands.add_parser("audit", help="audit a stored answer against new Markdown")
    audit_parser.add_argument("--answer", required=True, type=Path)
    audit_parser.add_argument("--source", action="append", required=True, type=Path)
    audit_parser.add_argument("--document-id", action="append")
    audit_parser.add_argument("--output", type=Path)
    audit_parser.add_argument("--html", type=Path)
    demo_parser = commands.add_parser("demo", help="run the committed offline scenario")
    demo_parser.add_argument("--output-dir", type=Path, default=Path("demo-output"))
    demo_parser.add_argument("--fixture-dir", type=Path)
    return parser


def _ids(paths: list[Path], document_ids: list[str] | None):
    if document_ids and len(document_ids) != len(paths):
        raise ValueError("provide exactly one --document-id for each --source, or omit all IDs")
    return document_ids or [None] * len(paths)


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "demo":
        paths = run_demo(args.output_dir, args.fixture_dir)
        print(json.dumps({"written": [str(path) for path in paths]}, indent=2))
        return 0
    if args.command == "ask":
        _different_output(args.output, args.source)
        sources = [
            ingest_markdown(path, doc_id)
            for path, doc_id in zip(args.source, _ids(args.source, args.document_id), strict=True)
        ]
        config = GraphConfig(
            top_k=args.top_k,
            max_revisions=args.max_revisions,
            ollama_base_url=args.ollama_url,
            embedding_model=args.embedding_model,
            chat_model=args.chat_model,
        )
        result = ask(GraphRequest(question=args.question, sources=sources, config=config))
        _emit(result.answer.model_dump_json(indent=2), args.output)
        return 0
    inputs = [args.answer, *args.source]
    _different_output(args.output, inputs)
    _different_output(args.html, inputs)
    if args.output and args.html and args.output.resolve() == args.html.resolve():
        raise ValueError("--output and --html must resolve to different paths")
    answer = AnswerRecord.model_validate(json.loads(args.answer.read_text(encoding="utf-8")))
    sources = [
        ingest_markdown(path, doc_id)
        for path, doc_id in zip(args.source, _ids(args.source, args.document_id), strict=True)
    ]
    report = audit_answer(answer, sources)
    _emit(report.model_dump_json(indent=2), args.output)
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render_html(report), encoding="utf-8")
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except (OSError, ValueError, ValidationError, json.JSONDecodeError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
