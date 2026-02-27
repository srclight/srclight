"""Plain text and reStructuredText extractor (stdlib only)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from .base import make_document, make_section

if TYPE_CHECKING:
    from ..db import Database

# RST underline characters (in order of conventional precedence)
_RST_UNDERLINE_RE = re.compile(r"^([=\-~`'\"^_*+#<>])\1{2,}$")


class TextExtractor:
    language = "text"
    extensions = (".txt", ".rst", ".log", ".text")

    def extract(
        self,
        file_id: int,
        rel_path: str,
        source: bytes,
        db: Database,
    ) -> int:
        text = source.decode("utf-8", errors="replace")
        lines = text.splitlines()
        stem = Path(rel_path).stem

        # Try RST heading detection first
        sections = self._detect_rst_sections(lines)
        if sections:
            return self._emit_rst(file_id, rel_path, stem, lines, sections, db)

        # Small files: single document symbol
        if len(lines) < 40:
            sym = make_document(file_id, rel_path, text.strip())
            db.insert_symbol(sym, rel_path)
            return 1

        # Large files: chunk into ~80-line sections
        return self._emit_chunks(file_id, rel_path, stem, lines, db)

    def _detect_rst_sections(self, lines: list[str]) -> list[tuple[str, int]]:
        """Detect RST headings (title on line N, underline on line N+1).

        Returns list of (heading_text, line_index).
        """
        sections: list[tuple[str, int]] = []
        for i in range(len(lines) - 1):
            title = lines[i].strip()
            underline = lines[i + 1].strip()
            if (
                title
                and not _RST_UNDERLINE_RE.match(title)
                and _RST_UNDERLINE_RE.match(underline)
                and len(underline) >= len(title)
            ):
                sections.append((title, i))
        return sections

    def _emit_rst(
        self,
        file_id: int,
        rel_path: str,
        stem: str,
        lines: list[str],
        sections: list[tuple[str, int]],
        db: Database,
    ) -> int:
        count = 0
        for idx, (title, start_line) in enumerate(sections):
            # Content runs from heading to next heading (or EOF)
            end_line = sections[idx + 1][1] if idx + 1 < len(sections) else len(lines)
            content = "\n".join(lines[start_line:end_line]).strip()
            qualified = f"{stem} > {title}"

            sym = make_section(
                file_id,
                rel_path,
                name=title,
                qualified_name=qualified,
                content=content,
                start_line=start_line + 1,
                end_line=end_line,
            )
            db.insert_symbol(sym, rel_path)
            count += 1
        return count

    def _emit_chunks(
        self,
        file_id: int,
        rel_path: str,
        stem: str,
        lines: list[str],
        db: Database,
    ) -> int:
        chunk_size = 80
        count = 0
        for start in range(0, len(lines), chunk_size):
            end = min(start + chunk_size, len(lines))
            chunk = "\n".join(lines[start:end]).strip()
            if not chunk:
                continue
            section_num = start // chunk_size + 1
            # Use first non-blank line as a meaningful chunk name
            first_line = ""
            for line in lines[start:end]:
                stripped = line.strip()
                if stripped:
                    first_line = stripped
                    break
            name = first_line[:80] if first_line else f"Section {section_num}"
            qualified = f"{stem} > {name}"

            sym = make_section(
                file_id,
                rel_path,
                name=name,
                qualified_name=qualified,
                content=chunk,
                start_line=start + 1,
                end_line=end,
            )
            db.insert_symbol(sym, rel_path)
            count += 1
        return count
