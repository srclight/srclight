"""Document extractor registry.

Auto-discovers available extractors at import time. Stdlib extractors are
always registered; optional ones (pdf, docx, etc.) are wrapped in
try/except ImportError so missing dependencies just skip that extractor.
"""

from __future__ import annotations

import logging

from .base import ExtractorProtocol

logger = logging.getLogger("srclight.extractors")

# language name → extractor instance
_REGISTRY: dict[str, ExtractorProtocol] = {}

# file extension → language name  (e.g. ".pdf" → "pdf")
DOCUMENT_EXTENSIONS: dict[str, str] = {}


def _register(extractor: ExtractorProtocol) -> None:
    """Register an extractor and its extensions."""
    _REGISTRY[extractor.language] = extractor
    for ext in extractor.extensions:
        DOCUMENT_EXTENSIONS[ext] = extractor.language


def _discover() -> None:
    """Import and register all available extractors."""
    # --- Stdlib extractors (always available) ---
    from .csv_extractor import CsvExtractor
    from .email_extractor import EmailExtractor
    from .text_extractor import TextExtractor

    _register(TextExtractor())
    _register(EmailExtractor())
    _register(CsvExtractor())

    # --- Optional extractors (skip if dependency missing) ---
    try:
        from .pdf_extractor import PdfExtractor
        _register(PdfExtractor())
    except ImportError:
        logger.debug("pdfplumber not installed — PDF extraction disabled")

    try:
        from .docx_extractor import DocxExtractor
        _register(DocxExtractor())
    except ImportError:
        logger.debug("python-docx not installed — DOCX extraction disabled")

    try:
        from .xlsx_extractor import XlsxExtractor
        _register(XlsxExtractor())
    except ImportError:
        logger.debug("openpyxl not installed — XLSX extraction disabled")

    try:
        from .html_extractor import HtmlExtractor
        _register(HtmlExtractor())
    except ImportError:
        logger.debug("beautifulsoup4 not installed — HTML extraction disabled")

    try:
        from .image_extractor import ImageExtractor
        _register(ImageExtractor())
    except ImportError:
        logger.debug("Pillow not installed — image extraction disabled")


def detect_document_language(suffix: str) -> str | None:
    """Return the language name for a document extension, or None."""
    return DOCUMENT_EXTENSIONS.get(suffix.lower())


def get_registry() -> dict[str, ExtractorProtocol]:
    """Return the extractor registry (language → extractor)."""
    return _REGISTRY


# Run discovery once at import time
_discover()
