"""Controlled degradation of real test-page scans (robustness experiment).

Each held-out page is degraded at increasing severity and read again by
(a) the rule-based grid reading and (b) the CNN trained on clean pages.
Reported per condition: cell accuracy over all ground-truth letter cells
(cells that were not found count as errors) and the end-to-end CER/WER.

The degradations are applied to the real scans, never to synthetic pages, and
are reported as such.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import replace
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

from .dataset import gt_cells, read_manifest
from .geometry import BrailleGeometry
from .metrics import align, cer, wer
from .segment import segment_page
from .transliterate import masks_to_gujarati


def _downscale(f: float) -> Callable:
    def fn(img, rng):
        h, w = img.shape
        return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), f
    return fn


def _blur(s: float) -> Callable:
    return lambda img, rng: (cv2.GaussianBlur(img, (0, 0), s), 1.0)


def _noise(s: float) -> Callable:
    def fn(img, rng):
        return np.clip(img.astype(np.float32) + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8), 1.0
    return fn


def _fade(k: float, noise: float = 8.0) -> Callable:
    """Faded toner: ink contrast reduced to a fraction k, plus mild noise."""
    def fn(img, rng):
        paper = float(np.percentile(img, 90))
        out = paper - (paper - img.astype(np.float32)) * k + rng.normal(0, noise, img.shape)
        return np.clip(out, 0, 255).astype(np.uint8), 1.0
    return fn


CONDITIONS: List[Tuple[str, str, Callable]] = [
    ("clean", "none", lambda img, rng: (img, 1.0)),
    ("resolution", "100 dpi", _downscale(100 / 150)),
    ("resolution", "75 dpi", _downscale(75 / 150)),
    ("resolution", "60 dpi", _downscale(60 / 150)),
    ("blur", "sigma 1.5 px", _blur(1.5)),
    ("blur", "sigma 2.5 px", _blur(2.5)),
    ("blur", "sigma 3.5 px", _blur(3.5)),
    ("noise", "sigma 25", _noise(25)),
    ("noise", "sigma 45", _noise(45)),
    ("noise", "sigma 65", _noise(65)),
    ("faded toner", "40 % contrast", _fade(0.40)),
    ("faded toner", "25 % contrast", _fade(0.25)),
    ("faded toner", "15 % contrast", _fade(0.15)),
]


def _read(page, model) -> Tuple[List[List[int]], List[List[int]]]:
    rule = [[c.rule_mask for c in line] for line in page.lines]
    if model is None:
        return rule, rule
    cnn = []
    for line in page.lines:
        masks = [c.rule_mask for c in line]
        idx = [i for i, m in enumerate(masks) if m]
        if idx:
            X = np.stack([line[i].crop for i in idx]).astype("float32")[..., None] / 255.0
            for i, m in zip(idx, model.predict(X, batch_size=512, verbose=0).argmax(1)):
                masks[i] = int(m)
        cnn.append(masks)
    return rule, cnn


def _seq(lines: List[List[int]]) -> List[int]:
    out: List[int] = []
    for i, l in enumerate(lines):
        if i:
            out.append(0)
        out.extend(l)
    # strip blanks at line ends for a fair comparison with the ground truth
    return out


def _cell_acc(gt: List[int], pred: List[int]) -> Tuple[int, int]:
    pairs = align(gt, pred)
    letters = sum(1 for m in gt if m)
    correct = sum(1 for i, j in pairs if gt[i] and gt[i] == pred[j])
    return correct, letters


def run_stress(manifest: str, results_dir: str, out_dir: str, geom: BrailleGeometry,
               seed: int = 0, model_path: str = "") -> Dict:
    import tensorflow as tf

    split = json.load(open(os.path.join(results_dir, "split.json"), encoding="utf-8"))
    rows = [r for r in read_manifest(manifest) if split.get(r["doc_id"]) == "test"]
    model_path = model_path or os.path.join(results_dir, "cnn_grid", f"seed{seed}", "model.keras")
    model = tf.keras.models.load_model(model_path)
    rng = np.random.default_rng(seed)
    table = []
    for family, level, fn in CONDITIONS:
        tot = {"rule": [0, 0, 0, 0, 0, 0], "cnn": [0, 0, 0, 0, 0, 0]}
        for r in rows:
            img = cv2.imread(r["image"], cv2.IMREAD_GRAYSCALE)
            deg, scale = fn(img, rng)
            g = replace(geom, dpi=float(r.get("dpi") or geom.dpi) * scale)
            page = segment_page(deg, g, dot_mode="dark")
            rule, cnn = _read(page, model)
            gt = gt_cells(r)
            ref = open(r["gt_gujarati"], encoding="utf-8").read()
            for name, lines in (("rule", rule), ("cnn", cnn)):
                c, n = _cell_acc(gt, _seq(lines))
                hyp = "\n".join(masks_to_gujarati(l) for l in lines)
                ce, cn = cer(ref, hyp)
                we, wn = wer(ref, hyp)
                t = tot[name]
                t[0] += c; t[1] += n; t[2] += ce; t[3] += cn; t[4] += we; t[5] += wn
        for name, t in tot.items():
            table.append({"family": family, "level": level, "system": name,
                          "cell_accuracy": t[0] / t[1], "CER": t[2] / t[3], "WER": t[4] / t[5],
                          "pages": len(rows), "letter_cells": t[1]})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stress.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    return {"rows": table}
