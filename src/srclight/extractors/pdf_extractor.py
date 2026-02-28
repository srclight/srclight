"""PDF extractor (requires pdfplumber, optional PaddleOCR for scanned pages)."""

from __future__ import annotations

import io
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber

from .base import make_document, make_section

if TYPE_CHECKING:
    from ..db import Database

logger = logging.getLogger("srclight.extractors.pdf")


class PdfExtractor:
    language = "pdf"
    extensions = (".pdf",)

    def __init__(self) -> None:
        self._paddle = None
        self._paddle_attempted = False

    def extract(
        self,
        file_id: int,
        rel_path: str,
        source: bytes,
        db: Database,
    ) -> int:
        stem = Path(rel_path).stem

        with pdfplumber.open(io.BytesIO(source)) as pdf:
            page_count = len(pdf.pages)
            metadata = pdf.metadata or {}

            # Build metadata doc_comment
            meta_parts = [f"{page_count} pages"]
            author = metadata.get("Author")
            if author:
                meta_parts.append(f"Author: {author}")
            title = metadata.get("Title")
            if title:
                meta_parts.append(f"Title: {title}")
            file_doc_comment = ", ".join(meta_parts)

            # First pass: collect text spans with font sizes across all pages
            spans: list[tuple[str, float, int]] = []  # (text, font_size, page_num)
            for page_idx, page in enumerate(pdf.pages):
                words = page.extract_words(extra_attrs=["size"])
                for w in words:
                    spans.append((w["text"], w.get("size", 0), page_idx))

            # Try heading detection via font size clustering
            sections = self._detect_heading_sections(spans, stem)

            if sections:
                return self._emit_sections(
                    file_id, rel_path, stem, sections, file_doc_comment, db,
                )

            # Fallback: extract tables and chunk by page
            count = 0
            for page_idx, page in enumerate(pdf.pages):
                # Tables
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    table_content = self._serialize_table(table)
                    if table_content.strip():
                        name = f"Table {t_idx + 1} (page {page_idx + 1})"
                        sym = make_section(
                            file_id, rel_path,
                            name=name,
                            qualified_name=f"{stem} > {name}",
                            content=table_content,
                            kind="table",
                            start_line=page_idx + 1,
                            end_line=page_idx + 1,
                            doc_comment=file_doc_comment if count == 0 else None,
                        )
                        db.insert_symbol(sym, rel_path)
                        count += 1

                # Page text (fall back to OCR for scanned pages)
                text = page.extract_text()
                if not (text and text.strip()):
                    text = self._ocr_page(page_idx, page, source)
                if text and text.strip():
                    name = f"Page {page_idx + 1}"
                    sym = make_section(
                        file_id, rel_path,
                        name=name,
                        qualified_name=f"{stem} > {name}",
                        content=text.strip(),
                        kind="page",
                        start_line=page_idx + 1,
                        end_line=page_idx + 1,
                        doc_comment=file_doc_comment if count == 0 else None,
                    )
                    db.insert_symbol(sym, rel_path)
                    count += 1

            if count == 0:
                # Empty PDF
                sym = make_document(
                    file_id, rel_path, "",
                    name=stem, doc_comment=file_doc_comment,
                )
                db.insert_symbol(sym, rel_path)
                count = 1

            return count

    def _detect_heading_sections(
        self,
        spans: list[tuple[str, float, int]],
        stem: str,
    ) -> list[tuple[str, str, int]] | None:
        """Cluster font sizes to detect headings.

        Returns list of (heading_text, body_text, page_num) or None
        if no font-size variation detected.
        """
        if not spans:
            return None

        # Count font sizes
        size_counter: Counter[float] = Counter()
        for _, size, _ in spans:
            if size > 0:
                size_counter[size] += 1

        if len(size_counter) < 2:
            return None  # No variation → can't detect headings

        # Most common size is body text; larger sizes are headings
        body_size = size_counter.most_common(1)[0][0]
        heading_sizes = sorted(
            [s for s in size_counter if s > body_size],
            reverse=True,
        )

        if not heading_sizes:
            return None

        # Build sections from heading spans, accumulating consecutive
        # heading-size words into a single heading string.
        heading_threshold = min(heading_sizes)
        sections: list[tuple[str, list[str], int]] = []  # (heading, [body_words], page)
        current_heading_words: list[str] = []
        current_body: list[str] = []
        current_page = 0
        in_heading = False

        for text, size, page_num in spans:
            if size >= heading_threshold:
                if in_heading:
                    # Continue accumulating heading words
                    current_heading_words.append(text)
                else:
                    # Transition body→heading: finalize previous section
                    current_heading = " ".join(current_heading_words) if current_heading_words else ""
                    if current_heading or current_body:
                        sections.append((current_heading, current_body, current_page))
                    current_heading_words = [text]
                    current_body = []
                    current_page = page_num
                    in_heading = True
            else:
                if in_heading:
                    in_heading = False
                if current_heading_words:
                    current_body.append(text)
                else:
                    # Preamble text before first heading
                    current_body.append(text)

        # Final section
        current_heading = " ".join(current_heading_words) if current_heading_words else ""
        if current_heading or current_body:
            sections.append((current_heading, current_body, current_page))

        if not any(h for h, _, _ in sections):
            return None

        return [
            (heading or f"Section {i + 1}", " ".join(body), page)
            for i, (heading, body, page) in enumerate(sections)
        ]

    def _emit_sections(
        self,
        file_id: int,
        rel_path: str,
        stem: str,
        sections: list[tuple[str, str, int]],
        file_doc_comment: str,
        db: Database,
    ) -> int:
        count = 0
        for heading, body, page_num in sections:
            content = f"{heading}\n\n{body}".strip() if body else heading

            # First section gets file-level metadata; subsequent sections
            # get their first body sentence for porter-stemmed FTS3.
            if count == 0:
                doc_comment = file_doc_comment
            elif body:
                first_sentence = body.split(".")[0].strip()
                doc_comment = first_sentence[:200] if first_sentence else None
            else:
                doc_comment = None

            sym = make_section(
                file_id, rel_path,
                name=heading,
                qualified_name=f"{stem} > {heading}",
                content=content,
                start_line=page_num + 1,
                end_line=page_num + 1,
                doc_comment=doc_comment,
            )
            db.insert_symbol(sym, rel_path)
            count += 1
        return count

    def _serialize_table(self, table: list[list[str | None]]) -> str:
        """Serialize a table as tab-delimited text."""
        lines = []
        for row in table:
            cells = [cell or "" for cell in row]
            lines.append("\t".join(cells))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Optional PaddleOCR for scanned pages
    # ------------------------------------------------------------------

    def _init_paddle(self) -> bool:
        """Lazily initialize PaddleOCR engine. Returns True if available."""
        if self._paddle is not None:
            return True
        if self._paddle_attempted:
            return False
        self._paddle_attempted = True
        try:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(lang="en", use_angle_cls=True, device="cpu")
            logger.debug("PaddleOCR initialized for scanned-page fallback")
            return True
        except ImportError:
            logger.debug("paddleocr not installed — scanned-page OCR disabled")
            return False
        except Exception as exc:
            logger.debug("PaddleOCR init failed: %s", exc)
            return False

    def _ocr_page(
        self, page_idx: int, page: pdfplumber.page.Page, source: bytes,
    ) -> str | None:
        """OCR a single scanned page. Returns extracted text or None."""
        if not page.images:
            return None
        if not self._init_paddle():
            return None
        try:
            from pdf2image import convert_from_bytes
            import numpy as np

            images = convert_from_bytes(
                source, dpi=200,
                first_page=page_idx + 1, last_page=page_idx + 1,
            )
            if not images:
                return None
            img_np = np.array(images[0])
            return self._run_paddle(img_np)
        except ImportError:
            logger.debug("pdf2image not installed — scanned-page OCR disabled")
            return None
        except Exception as exc:
            logger.debug("OCR failed for page %d: %s", page_idx + 1, exc)
            return None

    def _run_paddle(self, img_np) -> str | None:
        """Run PaddleOCR on a numpy image array. Returns text or None."""
        assert self._paddle is not None
        try:
            results_gen = self._paddle.paddlex_pipeline.predict(img_np)
            if hasattr(results_gen, "__iter__") and not isinstance(
                results_gen, (list, tuple)
            ):
                results = list(results_gen)
            else:
                results = results_gen
        except (AttributeError, TypeError):
            results = self._paddle.predict(img_np)

        if not results:
            return None

        if hasattr(results, "__iter__") and not isinstance(
            results, (list, tuple, dict)
        ):
            results = list(results)

        record = None
        if isinstance(results, list) and len(results) > 0:
            record = results[0]
        elif isinstance(results, dict):
            record = results

        if not record:
            return None

        rec_texts = record.get("rec_texts", [])
        text = "\n".join(rec_texts)
        return text.strip() or None
