"""Clean text while keeping a map back to raw offsets."""
from __future__ import annotations

from dataclasses import dataclass

_DROP = {"\u200b", "\ufeff", "\u00ad"}
_SPACES = {"\t", "\u00a0", "\u3000", "\x0b", "\x0c"}


@dataclass
class Normalized:
    text: str
    offsets: list[int]  # offsets[i] = index in the raw text of normalized char i

    def to_raw_span(self, start: int, end: int) -> tuple[int, int]:
        return self.offsets[start], self.offsets[end - 1] + 1


def normalize(raw: str) -> Normalized:
    out: list[str] = []
    offs: list[int] = []
    n = len(raw)
    for i, ch in enumerate(raw):
        if ch in _DROP:
            continue
        if ch == "\r":
            if i + 1 < n and raw[i + 1] == "\n":
                continue
            ch = "\n"
        elif ch in _SPACES:
            ch = " "
        if ch == " ":
            if not out or out[-1] in " \n":
                continue
        elif ch == "\n":
            while out and out[-1] == " ":
                out.pop()
                offs.pop()
            if len(out) >= 2 and out[-1] == "\n" and out[-2] == "\n":
                continue
        out.append(ch)
        offs.append(i)
    while out and out[-1] in " \n":
        out.pop()
        offs.pop()
    return Normalized("".join(out), offs)