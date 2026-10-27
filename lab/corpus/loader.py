"""Load documents from files, folders, or URLs."""
from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

SUPPORTED = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".pdf"}


@dataclass
class RawDocument:
    doc_id: str
    text: str
    source: str


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in ("p", "div", "br", "li", "tr", "section", "article"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif tag in ("p", "li") or re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(html: str) -> str:
    p = _HTMLText()
    p.feed(html)
    return "".join(p.out)


def _read(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return html_to_text(raw) if ext in (".html", ".htm") else raw


def load_documents(path: str | Path) -> list[RawDocument]:
    s = str(path)
    if s.startswith(("http://", "https://")):
        with urllib.request.urlopen(s, timeout=30) as r:
            return [RawDocument(s, html_to_text(r.read().decode("utf-8", errors="ignore")), s)]
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
    base = p.parent if p.is_file() else p
    docs = []
    for f in files:
        text = _read(f)
        if text.strip():
            docs.append(RawDocument(f.relative_to(base).as_posix(), text, str(f)))
    return docs