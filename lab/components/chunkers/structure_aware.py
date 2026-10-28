"""Split by markdown headings; oversized sections fall back to recursive splitting."""
import re

from lab.components import Chunker, approx_tokens, make_chunks
from lab.components.chunkers.recursive import RecursiveChunker
from lab.components.registry import register

_HEAD = re.compile(r"^#{1,6}\s+\S.*$", re.M)


@register("chunker", "structure_aware")
class StructureAwareChunker(Chunker):
    def __init__(self, max_size: int = 512, min_size: int = 40, overlap: int = 0):
        self.max_size, self.min_size = max_size, min_size
        self._rec = RecursiveChunker(size=max_size, overlap=overlap)

    def chunk(self, doc_id, text):
        bounds = sorted({0, *(m.start() for m in _HEAD.finditer(text))}) + [len(text)]
        sections = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        merged = []
        for s, e in sections:  # fold tiny sections (e.g. heading-only) into what follows
            if merged and approx_tokens(text[merged[-1][0]:merged[-1][1]]) < self.min_size:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        spans = []
        for s, e in merged:
            if approx_tokens(text[s:e]) <= self.max_size:
                spans.append((s, e))
            else:
                spans += [(c.start + s, c.end + s) for c in self._rec.chunk(doc_id, text[s:e])]
        return make_chunks(doc_id, text, spans)