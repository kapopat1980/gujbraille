"""Ink-printed Braille: make a print-ready PDF and import the scans.

print_pdf     renders every ground-truth page (<id>.brl.txt) at true physical
              size (BrailleGeometry, millimetres) as black dots on an A4 page,
              one Braille page per sheet, in sorted page-id order.
import_scans  takes the scanned sheets (PDF files and/or image files, in any
              order), converts them to 8-bit grey PNG, identifies which
              ground-truth page each scan shows by comparing the cells read
              from the scan with every ground-truth page (also trying the
              sheet rotated by 180 degrees), and writes images/<id>.png and
              the page manifest pages.csv used by `gujbraille build`.
"""
from __future__ import annotations

import csv
import difflib
import glob
import os
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from .geometry import BrailleGeometry

A4_MM = (210.0, 297.0)


def _gt_pages(gt_dir: str) -> Dict[str, List[str]]:
    pages = {}
    for f in sorted(glob.glob(os.path.join(gt_dir, "*.brl.txt"))):
        pid = os.path.basename(f)[: -len(".brl.txt")]
        with open(f, encoding="utf-8") as fh:
            pages[pid] = [l.rstrip("\n").replace(" ", "⠀") for l in fh if l.strip("\n")]
    return pages


def select_pages(pages: Dict[str, List[str]], per_doc: int = 0) -> Dict[str, List[str]]:
    """Keep the `per_doc` fullest pages of every document (0 = keep all)."""
    if not per_doc:
        return pages
    by_doc: Dict[str, List[str]] = {}
    for pid in pages:
        by_doc.setdefault(pid.rsplit("_", 1)[0], []).append(pid)
    keep = []
    for doc, pids in by_doc.items():
        full = sorted(pids, key=lambda q: -sum(len(l.strip("\u2800")) for l in pages[q]))[:per_doc]
        keep += full
    return {pid: pages[pid] for pid in sorted(keep)}


def print_pdf(gt_dir: str, out_pdf: str, geom: BrailleGeometry, margin_mm: float = 15.0,
              render_dpi: int = 600, per_doc: int = 0) -> Tuple[int, List[str]]:
    from PIL import Image, ImageDraw

    pages = select_pages(_gt_pages(gt_dir), per_doc)
    if not pages:
        raise FileNotFoundError(f"no .brl.txt files in {gt_dir}")
    g = replace(geom, dpi=float(render_dpi))
    W, H = int(round(g.px(A4_MM[0]))), int(round(g.px(A4_MM[1])))
    m, d, p, L, r = g.px(margin_mm), g.d, g.p, g.L, g.r
    max_cells = max(len(l) for ls in pages.values() for l in ls)
    max_lines = max(len(ls) for ls in pages.values())
    if m + max_cells * p > W - g.px(5) or m + max_lines * L > H - g.px(5):
        raise ValueError(f"{max_cells} cells x {max_lines} lines does not fit on A4 at this geometry; "
                         f"re-run `gujbraille prepare` with --cells 30 --lines 25")
    images, order = [], []
    for pid, lines in pages.items():
        im = Image.new("1", (W, H), 1)
        dr = ImageDraw.Draw(im)
        for j, line in enumerate(lines):
            for k, ch in enumerate(line):
                mask = ord(ch) - 0x2800
                for bit in range(6):
                    if mask >> bit & 1:
                        x = m + k * p + (bit // 3) * d
                        y = m + j * L + (bit % 3) * d
                        dr.ellipse([x - r, y - r, x + r, y + r], fill=0)
        images.append(im)
        order.append(pid)
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    images[0].save(out_pdf, save_all=True, append_images=images[1:], resolution=render_dpi)
    with open(os.path.splitext(out_pdf)[0] + "_order.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(order) + "\n")
    return len(images), order


def _load_scans(inputs: List[str], dpi: int) -> List[Tuple[str, np.ndarray]]:
    import cv2

    files: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for ext in ("*.pdf", "*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"):
                files += sorted(glob.glob(os.path.join(item, ext)))
        else:
            files += sorted(glob.glob(item)) or [item]
    out = []
    for f in files:
        if f.lower().endswith(".pdf"):
            try:
                import pymupdf as fitz
            except ImportError:  # older PyMuPDF
                import fitz

            doc = fitz.open(f)
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                out.append((f"{os.path.basename(f)}#p{i + 1}", arr.copy()))
        else:
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise ValueError(f"cannot read {f}")
            out.append((os.path.basename(f), im))
    return out


def _signature(lines: List[str], n: int = 4) -> str:
    return "\n".join(l.strip("⠀") for l in lines[:n])


def _auto_dpi(img: np.ndarray, fallback: float) -> float:
    """Resolution of a scan cropped to an A4 sheet (e.g. phone scanning apps)."""
    h, w = img.shape[:2]
    short, long_ = min(h, w), max(h, w)
    if abs(long_ / short - A4_MM[1] / A4_MM[0]) < 0.08:
        return short / (A4_MM[0] / 25.4)
    return fallback


def import_scans(inputs: List[str], gt_dir: str, out_dir: str, geom: BrailleGeometry,
                 dpi: int = 300, material: str = "ink", min_score: float = 0.6,
                 only: str = "", auto_dpi: bool = False) -> Dict:
    import cv2
    from .segment import cells_to_braille, segment_page

    gt = _gt_pages(gt_dir)
    if only:
        with open(only, encoding="utf-8") as fh:
            wanted = {l.strip() for l in fh if l.strip()}
        gt = {k: v for k, v in gt.items() if k in wanted}
    sigs = {pid: _signature(lines) for pid, lines in gt.items()}
    scans = _load_scans(inputs, dpi)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    g = replace(geom, dpi=float(dpi))
    mode = "relief" if material == "emboss" else "dark"
    assigned: Dict[str, Tuple[str, float]] = {}
    report = []
    page_dpi: Dict[str, float] = {}
    for name, im in scans:
        if auto_dpi:
            g = replace(geom, dpi=float(_auto_dpi(im, dpi)))
        best = ("", 0.0, im)
        for rot in (0, 180):
            img = im if rot == 0 else cv2.rotate(im, cv2.ROTATE_180)
            page = segment_page(img, g, dot_mode=mode)
            sig = _signature(cells_to_braille(page.lines).split("\n"))
            if not sig:
                continue
            for pid, s in sigs.items():
                sm = difflib.SequenceMatcher(None, sig, s, autojunk=False)
                if sm.real_quick_ratio() < best[1] or sm.quick_ratio() < best[1]:
                    continue
                score = sm.ratio()
                if score > best[1]:
                    best = (pid, score, img)
            if best[1] > 0.9:
                break
        pid, score, img = best
        status = "ok"
        if score < min_score:
            status = "unmatched"
        elif pid in assigned and assigned[pid][1] >= score:
            status = f"duplicate of {assigned[pid][0]}"
        report.append({"scan": name, "page_id": pid if status == "ok" else "", "score": round(score, 3),
                       "status": status})
        if status == "ok":
            if pid in assigned:  # better match replaces an earlier one
                for r in report:
                    if r["page_id"] == pid and r["scan"] == assigned[pid][0]:
                        r["page_id"], r["status"] = "", f"duplicate of {name}"
            assigned[pid] = (name, score)
            page_dpi[pid] = round(g.dpi, 1)
            cv2.imwrite(os.path.join(img_dir, pid + ".png"), img)
    rel_gt = os.path.relpath(gt_dir, out_dir).replace("\\", "/")
    rows = []
    for pid in sorted(assigned):
        rows.append({"page_id": pid, "doc_id": pid.rsplit("_", 1)[0], "image": f"images/{pid}.png",
                     "gt_braille": f"{rel_gt}/{pid}.brl.txt", "gt_gujarati": f"{rel_gt}/{pid}.guj.txt",
                     "material": material, "dpi": page_dpi.get(pid, dpi), "subject": "social science",
                     "notes": f"scan {assigned[pid][0]}"})
    with open(os.path.join(out_dir, "pages.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page_id", "doc_id", "image", "gt_braille", "gt_gujarati",
                                          "material", "dpi", "subject", "notes"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "scan_import_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scan", "page_id", "score", "status"])
        w.writeheader()
        w.writerows(report)
    missing = sorted(set(gt) - set(assigned))
    return {"scans": len(scans), "matched": len(assigned), "unmatched_or_duplicate":
            sum(1 for r in report if r["status"] != "ok"), "gt_pages_without_scan": missing}
