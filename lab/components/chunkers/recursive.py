from lab.components import Chunker, approx_tokens, make_chunks, token_spans
from lab.components.registry import register

SEPS = ["\n\n", "\n", ". ", " "]


def merge_spans(text, pieces, size, overlap=0):
    """Greedily merge contiguous pieces up to `size` tokens, carrying `overlap` tokens forward."""
    sizes = [approx_tokens(text[a:b]) for a, b in pieces]
    spans, i, n = [], 0, len(pieces)
    while i < n:
        j, tok = i, 0
        while j < n and (j == i or tok + sizes[j] <= size):
            tok += sizes[j]
            j += 1
        spans.append((pieces[i][0], pieces[j - 1][1]))
        if j >= n:
            break
        k, ov = j, 0
        while k - 1 > i and ov + sizes[k - 1] <= overlap:
            k -= 1
            ov += sizes[k]
        i = k
    return spans


@register("chunker", "recursive")
class RecursiveChunker(Chunker):
    def __init__(self, size: int = 512, overlap: int = 64, separators=None):
        self.size, self.overlap = size, overlap
        self.separators = list(separators or SEPS)

    def _split(self, text, s, e, seps):
        if approx_tokens(text[s:e]) <= self.size:
            return [(s, e)]
        if not seps:  # hard split on token windows
            ts = [(a + s, b + s) for a, b in token_spans(text[s:e])]
            return [(ts[i][0], ts[min(i + self.size, len(ts)) - 1][1]) for i in range(0, len(ts), self.size)]
        sep, pieces, pos = seps[0], [], s
        while pos < e:
            i = text.find(sep, pos, e)
            if i == -1:
                pieces.append((pos, e))
                break
            pieces.append((pos, i + len(sep)))
            pos = i + len(sep)
        if len(pieces) <= 1:
            return self._split(text, s, e, seps[1:])
        out = []
        for a, b in pieces:
            out.extend(self._split(text, a, b, seps[1:]))
        return out

    def chunk(self, doc_id, text):
        pieces = self._split(text, 0, len(text), self.separators)
        return make_chunks(doc_id, text, merge_spans(text, pieces, self.size, self.overlap))