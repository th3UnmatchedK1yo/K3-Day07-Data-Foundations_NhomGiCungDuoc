"""UIT student-procedures preprocessing.

Turns the raw, single-blob Markdown file exported for
``data/uit/quy-trinh-danh-cho-sinh-vien.md`` into a list of per-procedure
sections that downstream chunking/embedding code can consume.

The source file mixes many unrelated procedures in one Markdown document.
Some procedures start with a real Markdown/bold heading, others start
mid-paragraph (e.g. directly with "Bước 1") because the heading was lost
during the original crawl. Because there is no reliable structural marker
for every procedure, section boundaries are declared explicitly in
``data/uit/procedure_boundaries.json`` as exact "anchor phrases" rather than
hard-coded in Python. This keeps the detection logic generic while making
the mapping auditable and editable without touching code.

This module never rewrites, paraphrases, or removes any regulatory content.
It only normalizes whitespace/unicode noise and slices the text along the
configured anchor phrases; the wording, numbers, URLs, form names,
conditions and step order are preserved exactly as in the source file.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BOUNDARIES_PATH = _PACKAGE_ROOT / "data" / "uit" / "procedure_boundaries.json"
DEFAULT_SOURCE_PATH = _PACKAGE_ROOT / "data" / "uit" / "quy-trinh-danh-cho-sinh-vien.md"

# Bullet glyphs sometimes produced by PDF/HTML crawlers. A plain ASCII "-"
# (already used throughout the UIT source) is left untouched.
_BULLET_GLYPHS = "•◦‣▪●○∙·"
_BULLET_LINE_RE = re.compile(rf"^(\s*)[{re.escape(_BULLET_GLYPHS)}]\s*")

# Unicode code points that render as whitespace but are not a plain ASCII
# space; normalized to a regular space so tokenizers/matchers behave
# consistently. Newlines are handled separately.
_UNICODE_SPACE_MAP = {
    0x00A0: " ",  # NO-BREAK SPACE
    0x2007: " ",  # FIGURE SPACE
    0x2009: " ",  # THIN SPACE
    0x202F: " ",  # NARROW NO-BREAK SPACE
    0xFEFF: "",   # ZERO WIDTH NO-BREAK SPACE / BOM
}


class ProcedureBoundaryError(ValueError):
    """Raised when an anchor phrase from procedure_boundaries.json cannot be located."""


@dataclass(frozen=True)
class ProcedureSection:
    """A single detected procedure section of the UIT source document."""

    procedure_slug: str
    procedure_title: str
    text: str
    section_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "procedure_slug": self.procedure_slug,
            "procedure_title": self.procedure_title,
            "text": self.text,
            "section_index": self.section_index,
            "metadata": dict(self.metadata),
        }


def read_uit_markdown(path: str | Path = DEFAULT_SOURCE_PATH) -> str:
    """Read the raw UIT Markdown file as UTF-8 text without modifying it on disk."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"UIT source markdown not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def normalize_text(raw_text: str) -> str:
    """Clean up formatting noise while preserving the original wording exactly.

    Performed (formatting-only) normalizations:
        - newline style unification (\\r\\n, \\r -> \\n)
        - Unicode NFC normalization
        - non-breaking / exotic Unicode spaces -> regular space
        - unification of leading bullet glyphs to "- "
        - trailing whitespace trim per line
        - collapsing 3+ consecutive blank lines into exactly one blank line

    Never touches: numbers, URLs, form names ("Mẫu ..."), conditions, the
    order of steps, or the wording of any sentence.
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_UNICODE_SPACE_MAP)

    normalized_lines = []
    for line in text.split("\n"):
        stripped_line = line.rstrip()
        bullet_match = _BULLET_LINE_RE.match(stripped_line)
        if bullet_match:
            stripped_line = f"{bullet_match.group(1)}- {stripped_line[bullet_match.end():]}"
        normalized_lines.append(stripped_line)
    text = "\n".join(normalized_lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def load_procedure_boundaries(path: str | Path = DEFAULT_BOUNDARIES_PATH) -> dict[str, Any]:
    """Load the procedure-boundary configuration (anchors + source metadata)."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"procedure_boundaries.json not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if "procedures" not in config or not config["procedures"]:
        raise ValueError(f"procedure_boundaries.json at {config_path} defines no procedures")
    return config


def detect_sections(normalized_text: str, boundaries: dict[str, Any]) -> list[ProcedureSection]:
    """Slice ``normalized_text`` into ordered procedure sections using anchor phrases.

    Anchors must appear, in order, inside ``normalized_text``. Each section
    spans from its own anchor's start offset to the next anchor's start
    offset (or end-of-document for the last section). Missing anchors raise
    ``ProcedureBoundaryError`` instead of silently skipping content, so a
    stale config can never quietly drop regulatory text.
    """
    source_meta = boundaries.get("source_document", {})
    procedures = boundaries["procedures"]

    offsets: list[tuple[str, str, int]] = []
    search_from = 0
    for procedure in procedures:
        slug = procedure["slug"]
        anchor = procedure["anchor"]
        index = normalized_text.find(anchor, search_from)
        if index == -1:
            raise ProcedureBoundaryError(
                f"Anchor phrase for procedure '{slug}' not found at or after offset "
                f"{search_from}. Anchor={anchor!r}. The source document may have "
                "changed, or procedure_boundaries.json is out of date."
            )
        offsets.append((slug, procedure.get("title", slug), index))
        search_from = index + 1

    sections: list[ProcedureSection] = []
    for section_index, (slug, title, start) in enumerate(offsets):
        end = offsets[section_index + 1][2] if section_index + 1 < len(offsets) else len(normalized_text)
        section_text = normalized_text[start:end].strip()
        metadata = {
            "source_id": source_meta.get("source_id", "uit_student_procedures"),
            "source_title": source_meta.get("source_title", ""),
            "source_url": source_meta.get("source_url", ""),
            "institution": source_meta.get("institution", "UIT"),
            "audience": source_meta.get("audience", "student"),
            "document_type": source_meta.get("document_type", "procedure"),
            "procedure_slug": slug,
            "procedure_title": title,
            "section_index": section_index,
        }
        sections.append(
            ProcedureSection(
                procedure_slug=slug,
                procedure_title=title,
                text=section_text,
                section_index=section_index,
                metadata=metadata,
            )
        )

    logger.info("Detected %d procedure sections from UIT source document", len(sections))
    return sections


def preprocess_uit_document(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    boundaries_path: str | Path = DEFAULT_BOUNDARIES_PATH,
) -> list[dict[str, Any]]:
    """End-to-end: read raw file -> normalize -> detect procedure sections.

    Returns a list of plain dicts (see ``ProcedureSection.to_dict``), ready
    to be handed to ``src.structure_chunking.StructureAwareChunker``.
    """
    raw_text = read_uit_markdown(source_path)
    normalized = normalize_text(raw_text)
    boundaries = load_procedure_boundaries(boundaries_path)
    sections = detect_sections(normalized, boundaries)
    return [section.to_dict() for section in sections]


__all__ = [
    "ProcedureSection",
    "ProcedureBoundaryError",
    "read_uit_markdown",
    "normalize_text",
    "load_procedure_boundaries",
    "detect_sections",
    "preprocess_uit_document",
    "DEFAULT_BOUNDARIES_PATH",
    "DEFAULT_SOURCE_PATH",
]
