"""Generate a SYNTHETIC page set in the same format as real scans.

For software testing only.  Synthetic results are not evidence about
recognition of real Braille material and must not be reported as such.
"""
from __future__ import annotations

import csv
import os
from typing import List

import cv2
import numpy as np

from . import braille_table as T
from .geometry import BrailleGeometry
from .render import render_page
from .transliterate import gujarati_to_braille

_MATRAS = [""] * 6 + [v[1] for v in T.VOWELS.values() if v[1]]


def random_word(rng: np.random.Generator) -> str:
    cons = list(T.CONSONANTS) + list(T.CONJUNCTS)
    w = ""
    if rng.random() < 0.15:
        w += rng.choice(list(T.VOWELS))
    for _ in range(int(rng.integers(1, 4))):
        w += rng.choice(cons)
        if rng.random() < 0.12:
            w += "્" + rng.choice(list(T.CONSONANTS))
        w += rng.choice(_MATRAS)
        if rng.random() < 0.1:
            w += rng.choice(["ં", "ઃ", "ઁ"])
    return w


def random_text(rng: np.random.Generator, n_lines: int, cells_per_line: int) -> List[str]:
    """Gujarati lines whose Braille fits into cells_per_line cells."""
    lines = []
    for _ in range(n_lines):
        words: List[str] = []
        while True:
            if rng.random() < 0.05:
                w = "".join(rng.choice(list(T.DIGITS)) for _ in range(int(rng.integers(1, 4))))
            else:
                w = random_word(rng)
            if rng.random() < 0.08:
                w += rng.choice(list(T.PUNCTUATION))
            cand = " ".join(words + [w])
            if len(gujarati_to_braille(cand)) > cells_per_line:
                break
            words.append(w)
        lines.append(" ".join(words))
    return lines


def make_synthetic(out_dir: str, n_pages: int = 20, n_docs: int = 4, dpi: float = 200,
                   n_lines: int = 12, cells_per_line: int = 30, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    geom = BrailleGeometry(dpi=dpi)
    for d in ("images", "gt"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    rows = []
    for i in range(n_pages):
        pid = f"syn{i:04d}"
        doc = f"doc{i % n_docs}"
        style = "emboss" if i % 2 else "ink"
        guj = random_text(rng, n_lines, cells_per_line)
        brl = [gujarati_to_braille(l) for l in guj]
        img, _ = render_page(brl, geom, style=style, skew_deg=float(rng.uniform(-2, 2)),
                             noise_sigma=float(rng.uniform(2, 8)), dropout=0.01, rng=rng)
        cv2.imwrite(os.path.join(out_dir, "images", pid + ".png"), img)
        open(os.path.join(out_dir, "gt", pid + ".brl.txt"), "w", encoding="utf-8").write("\n".join(brl))
        open(os.path.join(out_dir, "gt", pid + ".guj.txt"), "w", encoding="utf-8").write("\n".join(guj))
        rows.append({"page_id": pid, "doc_id": doc, "image": f"images/{pid}.png",
                     "gt_braille": f"gt/{pid}.brl.txt", "gt_gujarati": f"gt/{pid}.guj.txt",
                     "material": style, "dpi": dpi, "subject": "synthetic",
                     "notes": "SYNTHETIC - software test only"})
    path = os.path.join(out_dir, "pages.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path
