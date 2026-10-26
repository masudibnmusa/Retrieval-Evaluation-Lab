import pytest

from lab.components import approx_tokens
from lab.components.registry import build

TEXT = "\n\n".join(
    f"# Section {i}\n\n" + " ".join(f"word{i}_{j}" for j in range(120)) + "." for i in range(4))

SPECS = [
    {"type": "fixed", "size": 50, "overlap": 10},
    {"type": "recursive", "size": 60, "overlap": 10},
    {"type": "sentence", "size": 60},
    {"type": "structure_aware", "max_size": 60, "min_size": 5},
]


@pytest.mark.parametrize("spec", SPECS)
def test_offsets_roundtrip_to_original_text(spec):
    chunks = build("chunker", spec).chunk("doc", TEXT)
    assert chunks
    for c in chunks:
        assert TEXT[c.start:c.end] == c.text
        assert c.doc_id == "doc"


def test_fixed_size_and_overlap():
    chunks = build("chunker", {"type": "fixed", "size": 50, "overlap": 10}).chunk("d", TEXT)
    assert all(c.n_tokens <= 50 for c in chunks)
    assert chunks[1].start < chunks[0].end
    assert chunks[-1].end == len(TEXT.rstrip())


def test_recursive_respects_size():
    chunks = build("chunker", {"type": "recursive", "size": 60, "overlap": 0}).chunk("d", TEXT)
    assert all(approx_tokens(c.text) <= 60 for c in chunks)


def test_structure_aware_splits_on_headings():
    chunks = build("chunker", {"type": "structure_aware", "max_size": 1000, "min_size": 5}).chunk("d", TEXT)
    assert len(chunks) == 4
    assert all(c.text.startswith("# Section") for c in chunks)


def test_full_coverage_without_overlap():
    chunks = build("chunker", {"type": "fixed", "size": 40, "overlap": 0}).chunk("d", TEXT)
    assert sum(c.n_tokens for c in chunks) == approx_tokens(TEXT)