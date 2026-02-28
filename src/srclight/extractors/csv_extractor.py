"""CSV extractor (stdlib only)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING

from .base import make_document

if TYPE_CHECKING:
    from ..db import Database

MAX_ROWS = 500


class CsvExtractor:
    language = "csv"
    extensions = (".csv", ".tsv")

    def extract(
        self,
        file_id: int,
        rel_path: str,
        source: bytes,
        db: Database,
    ) -> int:
        text = source.decode("utf-8", errors="replace")
        stem = Path(rel_path).stem

        # Detect delimiter
        try:
            sample = text[:8192]
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except csv.Error:
            # Default to comma, or tab for .tsv
            delimiter = "\t" if rel_path.endswith(".tsv") else ","

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows: list[list[str]] = []
        for row in reader:
            rows.append(row)
            if len(rows) > MAX_ROWS:
                break

        if not rows:
            sym = make_document(file_id, rel_path, "", name=stem)
            db.insert_symbol(sym, rel_path)
            return 1

        # Header row as signature
        header = rows[0]
        signature = "\t".join(header)

        # Content: all rows tab-delimited
        content_lines = ["\t".join(row) for row in rows]
        content = "\n".join(content_lines)

        # Count actual rows in full file
        total_rows = text.count("\n")
        col_count = len(header)
        truncated = len(rows) > MAX_ROWS

        col_names = ", ".join(h for h in header if h.strip())
        meta_parts = [f"{total_rows} rows", f"{col_count} columns"]
        if truncated:
            meta_parts.append(f"(truncated to {MAX_ROWS} rows)")
        stats = ", ".join(meta_parts)
        doc_comment = f"Columns: {col_names}. {stats}" if col_names else stats

        sym = make_document(
            file_id,
            rel_path,
            content,
            name=stem,
            signature=signature,
            doc_comment=doc_comment,
        )
        db.insert_symbol(sym, rel_path)
        return 1
