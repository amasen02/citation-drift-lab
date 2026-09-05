"""Local, source-version-aware Markdown retrieval and citation auditing."""

from .audit import audit_answer
from .graph import ask
from .ingest import ingest_markdown

__all__ = ["ask", "audit_answer", "ingest_markdown"]
__version__ = "0.1.0"
