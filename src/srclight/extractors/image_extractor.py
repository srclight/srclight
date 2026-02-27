"""Image extractor (requires Pillow, optional pytesseract for OCR)."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from PIL.ExifTags import TAGS

from .base import make_document

if TYPE_CHECKING:
    from ..db import Database


def _try_ocr(img: Image.Image) -> str | None:
    """Attempt OCR if pytesseract is available."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(img).strip()
        return text if text else None
    except ImportError:
        return None
    except Exception:
        return None


class ImageExtractor:
    language = "image"
    extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".svg", ".ico")

    def extract(
        self,
        file_id: int,
        rel_path: str,
        source: bytes,
        db: Database,
    ) -> int:
        stem = Path(rel_path).stem
        suffix = Path(rel_path).suffix.lower()

        if suffix == ".svg":
            return self._extract_svg(file_id, rel_path, stem, source, db)

        return self._extract_raster(file_id, rel_path, stem, source, db)

    def _extract_raster(
        self,
        file_id: int,
        rel_path: str,
        stem: str,
        source: bytes,
        db: Database,
    ) -> int:
        try:
            img = Image.open(io.BytesIO(source))
        except Exception:
            sym = make_document(file_id, rel_path, "", name=stem)
            db.insert_symbol(sym, rel_path)
            return 1

        width, height = img.size
        fmt = img.format or Path(rel_path).suffix.lstrip(".").upper()
        mode = img.mode
        signature = f"{width}x{height} {fmt}"

        # EXIF data
        meta_parts = [signature, f"Mode: {mode}"]
        try:
            exif = img.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    if tag_name in ("Make", "Model", "DateTime", "Software"):
                        meta_parts.append(f"{tag_name}: {value}")
        except Exception:
            pass
        doc_comment = "\n".join(meta_parts)

        # OCR
        content_parts = [signature]
        ocr_text = _try_ocr(img)
        if ocr_text:
            content_parts.append(f"\nOCR text:\n{ocr_text}")
        content = "\n".join(content_parts)

        sym = make_document(
            file_id, rel_path, content,
            name=stem, signature=signature, doc_comment=doc_comment,
        )
        db.insert_symbol(sym, rel_path)
        return 1

    def _extract_svg(
        self,
        file_id: int,
        rel_path: str,
        stem: str,
        source: bytes,
        db: Database,
    ) -> int:
        text = source.decode("utf-8", errors="replace")

        content_parts = []
        doc_parts = []
        sig_parts = ["SVG"]

        try:
            root = ET.fromstring(text)
            # Dimensions
            width = root.get("width", "")
            height = root.get("height", "")
            viewbox = root.get("viewBox", "")
            if width and height:
                sig_parts.append(f"{width}x{height}")
            elif viewbox:
                sig_parts.append(f"viewBox={viewbox}")

            # Namespace-aware search
            ns = {"svg": "http://www.w3.org/2000/svg"}

            # Title and desc
            title_el = root.find("svg:title", ns)
            if title_el is None:
                title_el = root.find("title")
            desc_el = root.find("svg:desc", ns)
            if desc_el is None:
                desc_el = root.find("desc")
            if title_el is not None and title_el.text:
                content_parts.append(f"Title: {title_el.text.strip()}")
                doc_parts.append(f"Title: {title_el.text.strip()}")
            if desc_el is not None and desc_el.text:
                content_parts.append(f"Description: {desc_el.text.strip()}")
                doc_parts.append(f"Description: {desc_el.text.strip()}")
        except ET.ParseError:
            content_parts.append("SVG (parse error)")

        signature = " ".join(sig_parts)
        content = "\n".join(content_parts) if content_parts else signature
        doc_comment = "\n".join(doc_parts) if doc_parts else None

        sym = make_document(
            file_id, rel_path, content,
            name=stem, signature=signature, doc_comment=doc_comment,
        )
        db.insert_symbol(sym, rel_path)
        return 1
