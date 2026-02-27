"""Email (.eml) extractor (stdlib only)."""

from __future__ import annotations

import email
import email.policy
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .base import make_document

if TYPE_CHECKING:
    from ..db import Database


class EmailExtractor:
    language = "email"
    extensions = (".eml",)

    def extract(
        self,
        file_id: int,
        rel_path: str,
        source: bytes,
        db: Database,
    ) -> int:
        msg = email.message_from_bytes(source, policy=email.policy.default)
        stem = Path(rel_path).stem

        subject = msg.get("Subject", "")
        from_addr = msg.get("From", "")
        date = msg.get("Date", "")
        to_addr = msg.get("To", "")

        # Build metadata doc_comment
        meta_parts = []
        if subject:
            meta_parts.append(f"Subject: {subject}")
        if from_addr:
            meta_parts.append(f"From: {from_addr}")
        if to_addr:
            meta_parts.append(f"To: {to_addr}")
        if date:
            meta_parts.append(f"Date: {date}")
        doc_comment = "\n".join(meta_parts) if meta_parts else None

        # Walk MIME tree for body text
        body = self._get_body(msg)
        name = subject if subject else stem

        sym = make_document(
            file_id,
            rel_path,
            body.strip() if body else "",
            name=name,
            signature=subject or None,
            doc_comment=doc_comment,
        )
        db.insert_symbol(sym, rel_path)
        return 1

    def _get_body(self, msg: email.message.Message) -> str:
        """Extract body text, preferring text/plain over text/html."""
        if msg.is_multipart():
            plain = ""
            html = ""
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain" and not plain:
                    payload = part.get_content()
                    if isinstance(payload, str):
                        plain = payload
                elif ct == "text/html" and not html:
                    payload = part.get_content()
                    if isinstance(payload, str):
                        html = payload
            return plain or self._strip_html(html)
        else:
            payload = msg.get_content()
            if isinstance(payload, str):
                ct = msg.get_content_type()
                return self._strip_html(payload) if ct == "text/html" else payload
            return ""

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags, returning plain text."""
        if not html:
            return ""
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
