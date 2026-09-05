"""Dependency-free static HTML audit report."""

from html import escape

from .models import AuditReport


def _e(value: object) -> str:
    return escape("" if value is None else str(value), quote=True)


def render_html(report: AuditReport) -> str:
    rows = []
    for finding in report.findings:
        rows.append(
            "<tr>"
            + "".join(
                [
                    f"<td>{_e(finding.claim_id)}</td>",
                    f"<td>{_e(finding.document_id)}</td>",
                    f"<td>{_e(finding.status)}</td>",
                    f"<td>{_e(finding.old_quote)}</td>",
                    f"<td>{_e(finding.candidate_quote)}</td>",
                    f"<td>{_e(finding.explanation)}</td>",
                ]
            )
            + "</tr>"
        )
    affected = ", ".join(_e(item) for item in report.affected_claim_ids) or "None"
    return (
        """<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">
<title>Citation drift report</title>
<style>
body{font:16px system-ui;margin:2rem;max-width:1100px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #bbb;padding:.5rem;text-align:left;vertical-align:top}
.changed,.deleted,.source_missing{color:#9b1c1c}
</style>
<h1>Citation drift report</h1>"""
        + f"<p>Answer: {_e(report.answer_id)}</p><p>Affected claims: {affected}</p>"
        + "<table><thead><tr><th>Claim</th><th>Document</th><th>Status</th>"
        + "<th>Old quote</th><th>Candidate</th><th>Explanation</th></tr></thead><tbody>"
        + "".join(rows)
        + f"</tbody></table><p>{_e(report.integrity_note)}</p></html>"
    )
