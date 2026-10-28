"""Split where adjacent-sentence similarity drops (embedding-based breakpoints)."""
import numpy as np

from lab.components import Chunker, approx_tokens, make_chunks
from lab.components.chunkers.sentence import sentence_spans
from lab.components.registry import build, register


@register("chunker", "semantic")
class SemanticChunker(Chunker):
    def __init__(self, embedder=None, breakpoint_percentile: float = 90,
                 max_size: int = 512, min_size: int = 50):
        self.embedder = build("embedder", embedder or {"type": "hf", "model": "sentence-transformers/all-MiniLM-L6-v2"})
        self.pct, self.max_size, self.min_size = breakpoint_percentile, max_size, min_size

    def chunk(self, doc_id, text):
        sents = sentence_spans(text)
        if len(sents) < 2:
            return make_chunks(doc_id, text, [(0, len(text))])
        vecs = self.embedder.embed_documents([text[a:b] for a, b in sents])
        dist = 1 - (vecs[:-1] * vecs[1:]).sum(axis=1)
        thr = np.percentile(dist, self.pct)
        spans, cur_start, cur_tok = [], sents[0][0], 0
        for i, (a, b) in enumerate(sents):
            tok = approx_tokens(text[a:b])
            if i > 0:
                split = (dist[i - 1] > thr and cur_tok >= self.min_size) or cur_tok + tok > self.max_size
                if split:
                    spans.append((cur_start, sents[i - 1][1]))
                    cur_start, cur_tok = a, 0
            cur_tok += tok
        spans.append((cur_start, sents[-1][1]))
        return make_chunks(doc_id, text, spans)