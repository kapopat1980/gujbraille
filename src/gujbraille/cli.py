"""Command-line interface.

    gujbraille prepare  --text chapter.txt --out data/real/gt --prefix HIST-STD8-CH01
    gujbraille printpdf --gt data/real/gt --out print/braille_pages.pdf
    gujbraille import-scans --scans scans/ --gt data/real/gt --out data/real
    gujbraille synth    --out data/synthetic --pages 40
    gujbraille build    --manifest data/real/pages.csv --out work/real
    gujbraille run      --data work/real --manifest data/real/pages.csv --out results/real
    gujbraille convert  --image page.png --model results/real/cnn_grid/seed0/model.keras
    gujbraille table    > table1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from .geometry import BrailleGeometry


def _geom(a) -> BrailleGeometry:
    return BrailleGeometry(dpi=a.dpi, dot_pitch_mm=a.dot_pitch, cell_pitch_mm=a.cell_pitch,
                           line_pitch_mm=a.line_pitch, dot_diameter_mm=a.dot_diameter)


def _geom_args(p):
    p.add_argument("--dpi", type=float, default=300)
    p.add_argument("--dot-pitch", type=float, default=2.5)
    p.add_argument("--cell-pitch", type=float, default=6.0)
    p.add_argument("--line-pitch", type=float, default=10.0)
    p.add_argument("--dot-diameter", type=float, default=1.5)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gujbraille")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a synthetic test set (software testing only)")
    s.add_argument("--out", required=True)
    s.add_argument("--pages", type=int, default=40)
    s.add_argument("--docs", type=int, default=8)
    s.add_argument("--dpi", type=float, default=200)
    s.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("build", help="segment pages, align with ground truth, save labelled cells")
    b.add_argument("--manifest", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--min-match", type=float, default=0.90)
    _geom_args(b)

    r = sub.add_parser("run", help="baselines + CNN (5 seeds) + end-to-end CER/WER")
    r.add_argument("--data", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    r.add_argument("--epochs", type=int, default=100)
    r.add_argument("--split-seed", type=int, default=2026)
    r.add_argument("--group-key", default="doc_id", choices=["doc_id", "page_id"])
    _geom_args(r)

    c = sub.add_parser("convert", help="convert one page image to Gujarati text")
    c.add_argument("--image", required=True)
    c.add_argument("--model", default=None, help="Keras model; omit for rule-based reading")
    c.add_argument("--material", default="ink", choices=["ink", "emboss"])
    _geom_args(c)

    pr = sub.add_parser("prepare", help="Gujarati text -> paginated Braille ground truth (+ .brf)")
    pr.add_argument("--text", required=True, help="UTF-8 Gujarati source text")
    pr.add_argument("--out", required=True)
    pr.add_argument("--prefix", required=True, help="document id, e.g. HIST-STD8-CH01")
    pr.add_argument("--cells", type=int, default=40)
    pr.add_argument("--lines", type=int, default=25)

    pp = sub.add_parser("printpdf", help="print-ready A4 PDF of all ground-truth pages (true size)")
    pp.add_argument("--gt", required=True, help="folder with <id>.brl.txt files")
    pp.add_argument("--out", required=True, help="output PDF file")
    pp.add_argument("--margin", type=float, default=15.0, help="page margin in mm")
    pp.add_argument("--per-doc", type=int, default=0, help="print only the N fullest pages of each chapter")
    _geom_args(pp)

    im = sub.add_parser("import-scans", help="identify scanned pages and write images/ + pages.csv")
    im.add_argument("--scans", required=True, nargs="+", help="scanned PDF files, image files or folders")
    im.add_argument("--gt", required=True)
    im.add_argument("--out", required=True)
    im.add_argument("--material", default="ink", choices=["ink", "emboss"])
    im.add_argument("--only", default="", help="page list written by printpdf (<pdf>_order.txt)")
    im.add_argument("--auto-dpi", action="store_true",
                    help="derive the resolution from the A4 sheet size (phone scanning apps)")
    _geom_args(im)

    st = sub.add_parser("stress", help="robustness: degrade the real test-page scans and re-read them")
    st.add_argument("--manifest", required=True)
    st.add_argument("--results", required=True, help="output folder of `gujbraille run`")
    st.add_argument("--out", required=True)
    st.add_argument("--seed", type=int, default=0)
    _geom_args(st)

    sub.add_parser("table", help="print the Braille-Gujarati table as CSV")

    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if a.cmd == "synth":
        from .synthetic import make_synthetic
        print(make_synthetic(a.out, a.pages, a.docs, a.dpi, seed=a.seed))
    elif a.cmd == "build":
        from .dataset import build
        print(json.dumps(build(a.manifest, a.out, _geom(a), min_match=a.min_match), indent=2))
    elif a.cmd == "run":
        from .experiment import run_all
        res = run_all(a.data, a.manifest, a.out, a.seeds, a.epochs, a.split_seed, a.group_key, a.dpi)
        print(open(f"{a.out}/summary.md", encoding="utf-8").read())
    elif a.cmd == "convert":
        import cv2
        import numpy as np
        from .segment import segment_page
        from .transliterate import masks_to_gujarati
        page = segment_page(cv2.imread(a.image, cv2.IMREAD_GRAYSCALE), _geom(a),
                            dot_mode="relief" if a.material == "emboss" else "dark")
        model = None
        if a.model:
            import tensorflow as tf
            model = tf.keras.models.load_model(a.model)
        for line in page.lines:
            masks = [c.rule_mask for c in line]
            if model is not None:
                idx = [i for i, m in enumerate(masks) if m]
                if idx:
                    X = np.stack([line[i].crop for i in idx]).astype("float32")[..., None] / 255.0
                    for i, m in zip(idx, model.predict(X, verbose=0).argmax(1)):
                        masks[i] = int(m)
            print(masks_to_gujarati(masks))
    elif a.cmd == "prepare":
        from .prepare import write_pages
        text = open(a.text, encoding="utf-8").read()
        for pid in write_pages(text, a.out, a.prefix, a.cells, a.lines):
            print(pid)
    elif a.cmd == "printpdf":
        from .printing import print_pdf
        n, order = print_pdf(a.gt, a.out, _geom(a), a.margin, per_doc=a.per_doc)
        print(f"{n} pages written to {a.out} (order in {a.out[:-4]}_order.txt)")
        print("Print at 100 % / Actual size (NOT 'Fit to page'), single-sided, black only.")
    elif a.cmd == "import-scans":
        from .printing import import_scans
        res = import_scans(a.scans, a.gt, a.out, _geom(a), int(a.dpi), a.material,
                           only=a.only, auto_dpi=a.auto_dpi)
        print(json.dumps(res, indent=2))
    elif a.cmd == "stress":
        from .stress import run_stress
        res = run_stress(a.manifest, a.results, a.out, _geom(a), a.seed)
        for r in res["rows"]:
            print(f"{r['family']:12s} {r['level']:14s} {r['system']:5s} cell acc {100*r['cell_accuracy']:6.2f} %"
                  f"  CER {100*r['CER']:6.2f} %  WER {100*r['WER']:6.2f} %")
    elif a.cmd == "table":
        from .braille_table import table_rows
        w = csv.writer(sys.stdout)
        w.writerow(["class", "gujarati", "unicode", "braille", "dots"])
        w.writerows(table_rows())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
