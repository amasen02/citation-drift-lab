"""Reproducible model-free demonstration."""

from __future__ import annotations

from pathlib import Path

from .audit import audit_answer
from .graph import ask
from .ingest import ingest_markdown
from .models import AnswerRecord, GraphRequest
from .report import render_html


def _dump(model, path: Path) -> None:
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def run_demo(output_dir: str | Path, fixture_dir: str | Path | None = None) -> list[Path]:
    repository_fixtures = Path("examples") / "fixtures"
    fixtures = (
        Path(fixture_dir)
        if fixture_dir
        else repository_fixtures
        if repository_fixtures.is_dir()
        else Path(__file__).with_name("fixtures")
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    old = ingest_markdown(fixtures / "policy-v1.md", document_id="team-policy")
    new = ingest_markdown(fixtures / "policy-v2.md", document_id="team-policy")
    questions = [
        "What is the project record limit?",
        "What exception applies during incidents?",
        "Where are audit exports stored?",
        "What color is the deployment button?",
    ]
    results = [ask(GraphRequest(question=question, sources=[old])) for question in questions]
    supported = [result for result in results if not result.abstained]
    claims = []
    for number, result in enumerate(supported, 1):
        for claim in result.answer.claims[:1]:
            claims.append(claim.model_copy(update={"claim_id": f"demo:claim:{number}"}))
    answer = AnswerRecord(
        answer_id="citation-drift-lab-demo-v01",
        question="; ".join(questions[:3]),
        response="\n\n".join(claim.text for claim in claims),
        mode="extractive_offline",
        claims=claims,
    )
    report = audit_answer(answer, [new])
    paths = [
        output / "answer.json",
        output / "audit.json",
        output / "report.html",
        output / "unsupported-answer.json",
    ]
    _dump(answer, paths[0])
    _dump(report, paths[1])
    paths[2].write_text(render_html(report), encoding="utf-8")
    _dump(results[-1], paths[3])
    return paths
