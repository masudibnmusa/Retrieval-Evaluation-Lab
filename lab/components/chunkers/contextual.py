"""Prepend LLM-generated context to each chunk. Offsets still point at the ORIGINAL text;
only `embed_text` changes."""
from lab.components import Chunker
from lab.components.registry import build, register
from lab.llm import complete

PROMPT = """<document_excerpt>
{doc}
</document_excerpt>

Here is a chunk from that document:
<chunk>
{chunk}
</chunk>

Write 1-2 short sentences situating this chunk within the document, to improve search retrieval of the chunk.
Answer with only the context sentences."""


@register("chunker", "contextual")
class ContextualChunker(Chunker):
    def __init__(self, base=None, model: str | None = None, window_chars: int = 12000):
        self.base = build("chunker", base or {"type": "recursive", "size": 512, "overlap": 64})
        self.model, self.window = model, window_chars

    def chunk(self, doc_id, text):
        chunks = self.base.chunk(doc_id, text)
        for c in chunks:
            excerpt = text[max(0, c.start - self.window): c.end + self.window]
            ctx = complete(PROMPT.format(doc=excerpt, chunk=c.text), model=self.model,
                           max_tokens=150, category="contextualize").strip()
            c.embed_text = f"{ctx}\n\n{c.text}"
        return chunks