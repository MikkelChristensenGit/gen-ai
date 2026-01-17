from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    source_name: str
    page: int | None


def _safe_page_to_human(page: int | None) -> str:
    # PyPDFLoader typically stores 0-indexed page numbers in metadata["page"]
    if page is None:
        return "?"
    return str(page + 1)


def format_context(chunks: Iterable[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {c.source_name} (p. {_safe_page_to_human(c.page)})\n{c.text}")
    return "\n\n---\n\n".join(parts)
