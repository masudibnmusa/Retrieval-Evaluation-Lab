import re

from lab.components import Chunker, make_chunks
from lab.components.chunkers.recursive import merge_spans
from lab.components.registry import register

_SENT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def sentence_spans(text):
    spans, pos = [], 0
    for m in _SENT.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


@register("chunker", "sentence")
class SentenceChunker(Chunker):
    def __init__(self, size: int = 256, overlap: int = 0):
        self.size, self.overlap = size, overlap

    def chunk(self, doc_id, text):
        return make_chunks(doc_id, text, merge_spans(text, sentence_spans(text), self.size, self.overlap))