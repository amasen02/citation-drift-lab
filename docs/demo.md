# Demo walkthrough

Run `citation-drift-lab demo --output-dir demo-output`. Version 1 contains a 30-record
limit, an incident exception, an unchanged audit-export statement, hostile prompt-like
text, and an HTML `<script>` heading. Version 2 changes the limit to 20, removes the
exception, edits unrelated text, and retains the stable quote.

Inspect `answer.json` first: every claim has a quote and offsets. Then inspect
`audit.json`: the numeric claim is changed and material, the removed exception is
deleted, and the audit-export quote remains valid (it may be relocated after edits).
`unsupported-answer.json` records bounded abstention. Open `report.html` as a static
file and confirm hostile markup is displayed as text rather than interpreted.
