# Security policy

## Supported version

Version 0.1.x receives security fixes while it is the current release line.

## Reporting

Do not include secrets, proprietary documents, or exploit payloads in a public issue.
Contact the repository owner privately with the affected version, reproduction steps,
and impact. Expect acknowledgement before public disclosure.

## Boundaries

The tool reads paths explicitly supplied by the operator and can optionally connect only
to a loopback Ollama endpoint. Markdown and report fields are untrusted data. This is a
local analysis utility, not an authorization boundary or an entailment verifier.
