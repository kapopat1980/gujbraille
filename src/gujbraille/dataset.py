"""Dataset construction with documented provenance.

Input: a page manifest (CSV) with one row per scanned page::

    page_id,doc_id,image,gt_braille,gt_gujarati,material,dpi,subject,notes

* ``image``        path of the scan (PNG/TIFF, lossless; relative to the CSV)
* ``gt_braille``   Unicode-Braille text file that was sent to the
                   printer/embosser (exact cell sequence)  - preferred
* ``gt_gujarati``  Gujarati source text of the page (used for CER/WER; if
                   gt_braille is missing it is generated from this text)
* ``material``     ``ink`` or ``emboss`` (selects the dot detector)
* ``doc_id``       source document/book - used for leakage-free splitting

Every page is segmented (segment.py) and the detected cell sequence is aligned
to the ground-truth cell sequence by minimum edit distance.  Each aligned
non-blank cell is saved twice (grid-anchored crop and bounding-box crop) with
its ground-truth label.  Pages whose alignment is poor are reported in
``qa_pages.csv`` and excluded, so that labels never come from the recogniser.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import replace
from typing import Dict, List, Optional

import cv2

from . import braille_table as T
from .geometry import BrailleGeometry
from .metrics import align
from .segment import segment_page, crop_cell
from .transliterate import gujarati_to_braille

MANIFEST_FIELDS = ["page_id", "doc_id", "image", "gt_braille", "gt_gujarati",
                   "material", "dpi", "subject", "notes"]


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    base = os.path.dirname(os.path.abspath(path))
    for r in rows:
        for k in ("image", "gt_braille", "gt_gujarati"):
            if r.get(k):
                r[k] = os.path.join(base, r[k])
    return rows


def gt_cells(row: Dict[str, str]) -> List[int]:
    """Ground-truth cell sequence; a line break is represented by a blank cell."""
    if row.get("gt_braille") and os.path.exists(row["gt_braille"]):
        txt = open(row["gt_braille"], encoding="utf-8").read()
    else:
        txt = gujarati_to_braille(open(row["gt_gujarati"], encoding="utf-8").read())
    seq: List[int] = []
    for line in txt.splitlines():
        line = line.replace(" ", "⠀").strip("⠀")
        if not line:
            continue
        if seq:
            seq.append(0)
        seq.extend(T.unicode_to_mask(c) for c in line)
    return seq


def detected_cells(page) -> List[tuple]:
    seq: List[tuple] = []
    for li, row in enumerate(page.lines):
        if seq:
            seq.append((0, None))
        for c in row:
            seq.append((c.rule_mask, c))
    return seq


def build(manifest: str, out_dir: str, geom: BrailleGeometry, size: int = 28,
          min_match: float = 0.90) -> Dict:
    rows = read_manifest(manifest)
    os.makedirs(os.path.join(out_dir, "cells", "grid"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "cells", "bbox"), exist_ok=True)
    cells_rows, qa_rows = [], []
    for r in rows:
        g = replace(geom, dpi=float(r.get("dpi") or geom.dpi))
        img = cv2.imread(r["image"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(r["image"])
        mode = "relief" if r.get("material", "ink") == "emboss" else "dark"
        page = segment_page(img, g, dot_mode=mode, crop_mode="grid", size=size)
        gt = gt_cells(r)
        det = detected_cells(page)
        pairs = align(gt, [m for m, _ in det])
        n_gt_letters = sum(1 for m in gt if m)
        aligned = [(i, j) for i, j in pairs if gt[i] and det[j][1] is not None]
        agree = sum(1 for i, j in aligned if gt[i] == det[j][0])
        coverage = len(aligned) / max(1, n_gt_letters)
        ok = coverage >= min_match
        qa_rows.append({
            "page_id": r["page_id"], "doc_id": r["doc_id"], "material": r.get("material", ""),
            "dpi": g.dpi, "skew_deg": round(page.skew_deg, 3), "n_dots": page.n_dots,
            "rejected_dots": page.n_rejected_dots, "gt_cells": n_gt_letters,
            "aligned_cells": len(aligned), "coverage": round(coverage, 4),
            "rule_reading_agreement": round(agree / max(1, len(aligned)), 4),
            "used": int(ok), "grid": json.dumps({k: round(float(v), 3) for k, v in page.grid.items()}),
        })
        if not ok:
            continue
        for i, j in aligned:
            c = det[j][1]
            cid = f"{r['page_id']}_{c.line:03d}_{c.k:03d}"
            pg = os.path.join("cells", "grid", cid + ".png")
            pb = os.path.join("cells", "bbox", cid + ".png")
            cv2.imwrite(os.path.join(out_dir, pg), c.crop)
            bb, _ = crop_cell(page.image, page.grid, c.line, c.k, c.rule_mask, size, "bbox", g)
            cv2.imwrite(os.path.join(out_dir, pb), bb)
            cells_rows.append({"cell_id": cid, "page_id": r["page_id"], "doc_id": r["doc_id"],
                               "line": c.line, "k": c.k, "label": gt[i], "rule_mask": c.rule_mask,
                               "path_grid": pg, "path_bbox": pb})
    _write(os.path.join(out_dir, "cells.csv"), cells_rows)
    _write(os.path.join(out_dir, "qa_pages.csv"), qa_rows)
    summary = {"pages": len(rows), "pages_used": sum(q["used"] for q in qa_rows),
               "cells": len(cells_rows),
               "classes_present": len({c["label"] for c in cells_rows})}
    with open(os.path.join(out_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _write(path: str, rows: List[Dict]) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
