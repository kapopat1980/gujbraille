"""Training, evaluation, baselines and repeated runs.

All randomness is seeded.  The train/validation/test split is made at the
level of the source document (``doc_id``) so that cells from the same page or
book never appear in both training and test data.
"""
from __future__ import annotations

import csv
import json
import os
import platform
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .dataset import read_manifest
from .geometry import BrailleGeometry
from .metrics import cer, classification_metrics, confusion, mcnemar, mean_sd_ci, wer
from .segment import segment_page
from .transliterate import braille_to_gujarati, masks_to_gujarati


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_cells(data_dir: str) -> List[Dict]:
    with open(os.path.join(data_dir, "cells.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("line", "k", "label", "rule_mask"):
            r[k] = int(r[k])
    return [r for r in rows if r["label"] != 0]  # blank cells are found by segmentation


def make_split(rows: List[Dict], seed: int = 2026, frac=(0.7, 0.1, 0.2),
               group_key: str = "doc_id") -> Dict[str, str]:
    """Group-wise split; returns {group_id: 'train'|'val'|'test'}."""
    groups = sorted({r[group_key] for r in rows})
    if len(groups) < 3:
        raise ValueError(f"need >= 3 distinct {group_key} values for a group split")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(round(frac[2] * n)))
    n_val = max(1, int(round(frac[1] * n)))
    split = {}
    for i, g in enumerate(groups):
        split[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return split


def arrays(rows: List[Dict], data_dir: str, crop: str = "grid"):
    X = np.stack([cv2.imread(os.path.join(data_dir, r[f"path_{crop}"]), cv2.IMREAD_GRAYSCALE)
                  for r in rows]).astype("float32") / 255.0
    y = np.array([r["label"] for r in rows], dtype="int64")
    return X[..., None], y


def environment() -> Dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "numpy": np.__version__, "opencv": cv2.__version__}
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
    except Exception:
        pass
    try:
        import sklearn
        info["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------
def train_cnn(Xtr, ytr, Xva, yva, seed: int, out_dir: str, epochs: int = 100,
              batch: int = 32, patience: int = 10, augment: bool = True, lr: float = 1e-3):
    import tensorflow as tf
    from .model import augmentation, build_cnn

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model = build_cnn(lr=lr)
    ds = tf.data.Dataset.from_tensor_slices((Xtr, ytr)).shuffle(len(Xtr), seed=seed).batch(batch)
    if augment:
        aug = augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    val = tf.data.Dataset.from_tensor_slices((Xva, yva)).batch(256)
    os.makedirs(out_dir, exist_ok=True)
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, min_delta=1e-4,
                                            restore_best_weights=True),
           tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "history.csv"))]
    model.fit(ds, validation_data=val, epochs=epochs, callbacks=cbs, verbose=0)
    model.save(os.path.join(out_dir, "model.keras"))
    with open(os.path.join(out_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))
    return model


def predict(model, X) -> np.ndarray:
    return model.predict(X, batch_size=512, verbose=0).argmax(1)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def hog_features(X: np.ndarray) -> np.ndarray:
    from skimage.feature import hog
    return np.stack([hog(x[..., 0], orientations=9, pixels_per_cell=(7, 7),
                         cells_per_block=(2, 2)) for x in X])


def baseline_svm(Xtr, ytr, Xte, seed: int = 0):
    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=1.0, random_state=seed, max_iter=5000)
    clf.fit(hog_features(Xtr), ytr)
    return clf.predict(hog_features(Xte))


def baseline_knn(Xtr, ytr, Xte, k: int = 3):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
    return clf.predict(Xte.reshape(len(Xte), -1))


# ---------------------------------------------------------------------------
# end-to-end page evaluation
# ---------------------------------------------------------------------------
def page_eval(manifest: str, page_ids: Sequence[str], geom: BrailleGeometry,
              model=None, crop: str = "grid") -> Dict:
    rows = [r for r in read_manifest(manifest) if r["page_id"] in set(page_ids)]
    tot = {"rule": [0, 0, 0, 0], "cnn": [0, 0, 0, 0], "mapping_only": [0, 0, 0, 0]}
    per_page = []
    for r in rows:
        g = replace(geom, dpi=float(r.get("dpi") or geom.dpi))
        img = cv2.imread(r["image"], cv2.IMREAD_GRAYSCALE)
        mode = "relief" if r.get("material") == "emboss" else "dark"
        page = segment_page(img, g, dot_mode=mode, crop_mode=crop)
        ref = open(r["gt_gujarati"], encoding="utf-8").read()
        hyps = {"rule": braille_to_gujarati("\n".join(
            "".join(chr(0x2800 + c.rule_mask) for c in line) for line in page.lines))}
        if model is not None:
            out_lines = []
            for line in page.lines:
                idx = [i for i, c in enumerate(line) if c.rule_mask]
                masks = [0] * len(line)
                if idx:
                    X = np.stack([line[i].crop for i in idx]).astype("float32")[..., None] / 255.0
                    for i, m in zip(idx, predict(model, X)):
                        masks[i] = int(m)
                out_lines.append(masks_to_gujarati(masks))
            hyps["cnn"] = "\n".join(out_lines)
        if r.get("gt_braille") and os.path.exists(r["gt_braille"]):
            hyps["mapping_only"] = braille_to_gujarati(open(r["gt_braille"], encoding="utf-8").read())
        rec = {"page_id": r["page_id"]}
        for k, h in hyps.items():
            ce, cn = cer(ref, h)
            we, wn = wer(ref, h)
            t = tot[k]
            t[0] += ce; t[1] += cn; t[2] += we; t[3] += wn
            rec[f"{k}_cer"] = ce / max(1, cn)
            rec[f"{k}_wer"] = we / max(1, wn)
        per_page.append(rec)
    summary = {k: {"CER": v[0] / v[1], "WER": v[2] / v[3]} for k, v in tot.items() if v[1]}
    return {"summary": summary, "per_page": per_page}


# ---------------------------------------------------------------------------
# full protocol
# ---------------------------------------------------------------------------
def run_all(data_dir: str, manifest: str, out_dir: str, seeds: Sequence[int] = (0, 1, 2, 3, 4),
            epochs: int = 100, split_seed: int = 2026, group_key: str = "doc_id",
            dpi: Optional[float] = None) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    rows = load_cells(data_dir)
    split = make_split(rows, split_seed, group_key=group_key)
    for r in rows:
        r["split"] = split[r[group_key]]
    with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, sort_keys=True)
    part = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    test_pages = sorted({r["page_id"] for r in part["test"]})
    counts = {s: len(v) for s, v in part.items()}
    counts["test_pages"] = len(test_pages)
    geom = BrailleGeometry(dpi=dpi or 300)
    results: Dict = {"counts": counts, "environment": environment(), "runs": {}}

    y_te = np.array([r["label"] for r in part["test"]])
    # rule-based reading (no learning)
    preds = {"rule_based": np.array([r["rule_mask"] for r in part["test"]])}
    results["runs"]["rule_based"] = [classification_metrics(y_te, preds["rule_based"])]

    data = {}
    for crop in ("grid", "bbox"):
        data[crop] = {s: arrays(part[s], data_dir, crop) for s in ("train", "val", "test")}
    Xtr, ytr = data["grid"]["train"]
    Xte, _ = data["grid"]["test"]
    preds["hog_svm"] = baseline_svm(Xtr, ytr, Xte)
    preds["knn3"] = baseline_knn(Xtr, ytr, Xte)
    results["runs"]["hog_svm"] = [classification_metrics(y_te, preds["hog_svm"])]
    results["runs"]["knn3"] = [classification_metrics(y_te, preds["knn3"])]

    page_results = []
    for crop in ("grid", "bbox"):
        key = f"cnn_{crop}"
        results["runs"][key] = []
        for s in seeds:
            rd = os.path.join(out_dir, key, f"seed{s}")
            (Xtr, ytr), (Xva, yva), (Xte, _) = (data[crop][p] for p in ("train", "val", "test"))
            m = train_cnn(Xtr, ytr, Xva, yva, s, rd, epochs=epochs)
            yp = predict(m, Xte)
            met = classification_metrics(y_te, yp)
            met["seed"] = s
            results["runs"][key].append(met)
            np.save(os.path.join(rd, "y_pred.npy"), yp)
            if s == seeds[0]:
                preds[key] = yp
            if crop == "grid":
                cm, labels = confusion(y_te, yp)
                np.savetxt(os.path.join(rd, "confusion.csv"), cm, fmt="%d", delimiter=",",
                           header=",".join(map(str, labels)))
                pe = page_eval(manifest, test_pages, geom, m, crop)
                pe["seed"] = s
                page_results.append(pe)
                with open(os.path.join(rd, "page_eval.json"), "w", encoding="utf-8") as f:
                    json.dump(pe, f, indent=2, ensure_ascii=False)
    agg = {}
    for key, runs in results["runs"].items():
        agg[key] = {m: mean_sd_ci([r[m] for r in runs]) for m in
                    ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    if page_results:
        agg["end_to_end"] = {}
        for sysname in page_results[0]["summary"]:
            agg["end_to_end"][sysname] = {
                met: mean_sd_ci([p["summary"][sysname][met] for p in page_results])
                for met in ("CER", "WER")}
    results["aggregate"] = agg
    # paired significance tests on the same test cells (first seed of each CNN)
    results["mcnemar_vs_cnn_grid"] = {k: mcnemar(y_te, preds["cnn_grid"], v)
                                      for k, v in preds.items() if k != "cnn_grid"}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_markdown(results))
    return results


def summary_markdown(res: Dict) -> str:
    c = res["counts"]
    lines = ["# Results summary", "",
             f"Cells: train {c['train']}, validation {c['val']}, test {c['test']} "
             f"({c['test_pages']} test pages).", "",
             "| System | Accuracy | Macro precision | Macro recall | Macro F1 | Runs |",
             "|---|---|---|---|---|---|"]
    names = {"rule_based": "Grid + rule-based dot reading (no learning)",
             "hog_svm": "HOG + linear SVM", "knn3": "k-NN (k=3, raw pixels)",
             "cnn_bbox": "CNN, bounding-box crops (ablation)",
             "cnn_grid": "CNN, grid-anchored crops (proposed)"}
    for k in ("rule_based", "knn3", "hog_svm", "cnn_bbox", "cnn_grid"):
        if k not in res["aggregate"]:
            continue
        a = res["aggregate"][k]
        f = lambda m: (f"{100*a[m]['mean']:.2f} ± {100*a[m]['sd']:.2f}"
                       if a[m]["n_runs"] > 1 else f"{100*a[m]['mean']:.2f}")
        lines.append(f"| {names[k]} | {f('accuracy')} | {f('macro_precision')} | "
                     f"{f('macro_recall')} | {f('macro_f1')} | {a['accuracy']['n_runs']} |")
    if "end_to_end" in res["aggregate"]:
        lines += ["", "| End-to-end (test pages) | CER % | WER % |", "|---|---|---|"]
        for k, v in res["aggregate"]["end_to_end"].items():
            lines.append(f"| {k} | {100*v['CER']['mean']:.2f} ± {100*v['CER']['sd']:.2f} | "
                         f"{100*v['WER']['mean']:.2f} ± {100*v['WER']['sd']:.2f} |")
    if "mcnemar_vs_cnn_grid" in res:
        lines += ["", "| McNemar: CNN-grid (seed 0) vs | CNN right / other wrong | CNN wrong / other right | p |",
                  "|---|---|---|---|"]
        for k, v in res["mcnemar_vs_cnn_grid"].items():
            lines.append(f"| {names.get(k, k)} | {v['a_right_b_wrong']} | {v['a_wrong_b_right']} | {v['p_value']:.3g} |")
    lines += ["", "Values are mean ± SD over runs (percent). Macro averages are means of "
              "per-class values over classes present in the test set."]
    return "\n".join(lines) + "\n"
