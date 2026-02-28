"""HTML extractor (requires beautifulsoup4)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from .base import make_document, make_section

if TYPE_CHECKING:
    from ..db import Database

# Tags to strip before extraction
_STRIP_TAGS = {"script", "style", "nav", "footer"}

# Heading tags
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class HtmlExtractor:
    language = "html"
    extensions = (".html", ".htm")

    def extract(
        self,
        file_id: int,
        rel_path: str,
        source: bytes,
        db: Database,
    ) -> int:
        stem = Path(rel_path).stem
        text = source.decode("utf-8", errors="replace")
        soup = BeautifulSoup(text, "html.parser")

        # Strip noise tags
        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()

        # Extract page title
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else None

        # Find headings
        headings = soup.find_all(_HEADING_TAGS)

        if headings:
            return self._emit_heading_sections(
                file_id, rel_path, stem, soup, headings, page_title, db,
            )

        # Fallback: all text as single document
        all_text = soup.get_text(separator="\n", strip=True)
        sym = make_document(
            file_id, rel_path, all_text,
            name=page_title or stem,
            doc_comment=f"Title: {page_title}" if page_title else None,
        )
        db.insert_symbol(sym, rel_path)
        return 1

    def _emit_heading_sections(
        self,
        file_id: int,
        rel_path: str,
        stem: str,
        soup: BeautifulSoup,
        headings: list,
        page_title: str | None,
        db: Database,
    ) -> int:
        count = 0
        # Parent stack: (level, sym_id, name)
        parent_stack: list[tuple[int, int, str]] = []

        for h_idx, heading in enumerate(headings):
            level = int(heading.name[1])  # "h2" → 2
            heading_text = heading.get_text(strip=True)
            if not heading_text:
                continue

            # Collect text between this heading and the next
            body_parts: list[str] = []
            sibling = heading.next_sibling
            while sibling:
                if hasattr(sibling, "name") and sibling.name in _HEADING_TAGS:
                    break
                text = sibling.get_text(strip=True) if hasattr(sibling, "get_text") else str(sibling).strip()
                if text:
                    body_parts.append(text)
                sibling = sibling.next_sibling

            body = "\n".join(body_parts)
            content = f"{heading_text}\n\n{body}".strip() if body else heading_text

            # Maintain parent stack by level
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()

            parent_id = parent_stack[-1][1] if parent_stack else None

            ancestry = [name for _, _, name in parent_stack] + [heading_text]
            qualified = f"{stem} > " + " > ".join(ancestry)

            if count == 0 and page_title:
                doc_comment = f"Title: {page_title}"
            elif body_parts:
                doc_comment = body_parts[0][:200]
            else:
                doc_comment = None

            sym = make_section(
                file_id, rel_path,
                name=heading_text,
                qualified_name=qualified,
                content=content,
                signature=f"<{heading.name}> {heading_text}",
                parent_symbol_id=parent_id,
                doc_comment=doc_comment,
            )
            sym_id = db.insert_symbol(sym, rel_path)
            parent_stack.append((level, sym_id, heading_text))
            count += 1

        return count
