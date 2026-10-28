from lab.components import Chunker, make_chunks, token_spans
from lab.components.registry import register


@register("chunker", "fixed")
class FixedChunker(Chunker):
    def __init__(self, size: int = 512, overlap: int = 0):
        assert 0 <= overlap < size, "overlap must be < size"
        self.size, self.overlap = size, overlap

    def chunk(self, doc_id, text):
        ts = token_spans(text)
        step, spans = self.size - self.overlap, []
        for i in range(0, len(ts), step):
            win = ts[i:i + self.size]
            spans.append((win[0][0], win[-1][1]))
            if i + self.size >= len(ts):
                break
        return make_chunks(doc_id, text, spans)