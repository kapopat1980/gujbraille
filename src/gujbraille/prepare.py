"""Prepare ground-truth pages for embossing/printing.

Gujarati source text -> Gujarati Braille, word-wrapped to the page format of
the embosser (default 40 cells x 25 lines).  For every page three files are
written, which become the ground truth of the scan of that page:

    <id>.guj.txt   Gujarati words on the page (reference for CER/WER)
    <id>.brl.txt   Unicode Braille, exactly as laid out on the page
    <id>.brf       Braille ASCII for the embosser driver

Words longer than one line are hyphen-free split (rare in Gujarati prose).
"""
from __future__ import annotations

import os
from typing import List, Tuple

from .braille_table import to_brf
from .transliterate import gujarati_to_braille

BLANK = "⠀"


def paginate(text: str, cells_per_line: int = 40, lines_per_page: int = 25
             ) -> List[Tuple[List[str], List[str]]]:
    words = text.split()
    pages: List[Tuple[List[str], List[str]]] = []
    brl_lines: List[str] = []
    guj_lines: List[str] = []
    cur_b: List[str] = []
    cur_g: List[str] = []

    def flush_line():
        nonlocal cur_b, cur_g
        if cur_b:
            brl_lines.append(BLANK.join(cur_b))
            guj_lines.append(" ".join(cur_g))
        cur_b, cur_g = [], []
        if len(brl_lines) == lines_per_page:
            flush_page()

    def flush_page():
        nonlocal brl_lines, guj_lines
        if brl_lines:
            pages.append((brl_lines, guj_lines))
        brl_lines, guj_lines = [], []

    for w in words:
        b = gujarati_to_braille(w)
        while len(b) > cells_per_line:  # over-long token
            flush_line()
            cur_b, cur_g = [b[:cells_per_line]], [w]
            flush_line()
            b, w = b[cells_per_line:], ""
        need = len(b) + (1 if cur_b else 0)
        if sum(len(x) for x in cur_b) + len(cur_b) - (1 if cur_b else 0) + need > cells_per_line:
            flush_line()
        cur_b.append(b)
        cur_g.append(w)
    flush_line()
    flush_page()
    return pages


def write_pages(text: str, out_dir: str, prefix: str, cells_per_line: int = 40,
                lines_per_page: int = 25) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    ids = []
    for n, (brl, guj) in enumerate(paginate(text, cells_per_line, lines_per_page), 1):
        pid = f"{prefix}_{n:03d}"
        with open(os.path.join(out_dir, pid + ".brl.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(brl) + "\n")
        with open(os.path.join(out_dir, pid + ".guj.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(guj) + "\n")
        with open(os.path.join(out_dir, pid + ".brf"), "w", encoding="ascii") as f:
            f.write(to_brf("\n".join(brl).replace(BLANK, " ")) + "\n\f")
        ids.append(pid)
    return ids
