"""Pre-processing and cell segmentation of a scanned Braille page.

Pipeline (all pixel thresholds derived from BrailleGeometry):

1. grey-scale conversion and median denoising;
2. dot-candidate detection
     - ``dark``   : Otsu binarisation (inverse) - ink-printed Braille,
     - ``relief`` : signed deviation from the local background; the shadow of
                    each embossed dot (oblique scanner light) is the dot marker;
3. connected components filtered by area/aspect derived from the dot diameter
   (noise removal);
4. skew estimation by maximising the sharpness of the projection profile of
   dot centroids over -max_skew..+max_skew degrees, then rotation of the image;
5. estimation of the dot pitch from nearest-neighbour distances;
6. fitting of a regular Braille grid (line pitch, cell pitch, offsets) to the
   dot centroids by exhaustive search;
7. assignment of every dot to (line, cell, dot position) - this directly gives
   the rule-based reading of each cell and the word boundaries (blank cells);
8. extraction of a fixed-size, grid-anchored crop per cell (position of the
   dots inside the cell is preserved), or - for the ablation study - a tight
   bounding-box crop around the dots of the cell (``bbox`` mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import BrailleGeometry


@dataclass
class Cell:
    line: int
    k: int
    rule_mask: int                 # dot pattern read directly from the grid
    box: Tuple[int, int, int, int]  # x, y, w, h of the grid-anchored window
    crop: Optional[np.ndarray] = None


@dataclass
class PageResult:
    image: np.ndarray              # deskewed grey image
    skew_deg: float
    grid: Dict[str, float]
    lines: List[List[Cell]] = field(default_factory=list)
    n_dots: int = 0
    n_rejected_dots: int = 0


# ---------------------------------------------------------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _components(b: np.ndarray, geom: BrailleGeometry, lo: float = 0.25, hi: float = 4.0,
                max_side: float = 3.4) -> np.ndarray:
    r = geom.r
    n, _, st, cen = cv2.connectedComponentsWithStats(b, 8)
    area_ref = np.pi * r * r
    keep = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (lo * area_ref <= a <= hi * area_ref):
            continue
        if max(w, h) > max_side * r or min(w, h) < 0.4 * r:
            continue
        keep.append(cen[i])
    return np.array(keep, dtype=np.float64).reshape(-1, 2)


def detect_dots(gray: np.ndarray, geom: BrailleGeometry, mode: str = "dark") -> np.ndarray:
    """Return Nx2 array of dot centroids (x, y).

    ``dark``  : ink dots darker than the paper (Otsu, inverse binarisation).
    ``relief``: embossed dots under oblique light.  Each dot produces a bright
                highlight and a dark shadow.  Shadows (one component per dot)
                are used as dot markers and shifted by the median
                shadow->highlight half-vector so that the marker sits on the
                dot centre.
    """
    r = geom.r
    k = max(3, int(round(r / 2)) | 1)
    g = cv2.medianBlur(gray, k)
    if mode == "dark":
        t_otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # On nearly empty pages Otsu splits the paper noise; never threshold closer
        # to the paper level than 6 robust SDs (and at least 40 grey levels).
        med = float(np.median(g))
        sigma = 1.4826 * float(np.median(np.abs(g.astype(np.float32) - med)))
        t = min(t_otsu, med - max(40.0, 6.0 * sigma))
        b = (g < t).astype(np.uint8) * 255
        return _components(b, geom)
    if mode == "relief":
        bgk = int(round(geom.d * 2.5)) | 1
        bg = cv2.medianBlur(g, bgk).astype(np.int16)
        diff = g.astype(np.int16) - bg
        t, _ = cv2.threshold(np.abs(diff).astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        opn = np.ones((3, 3), np.uint8)
        dark = cv2.morphologyEx(((-diff) > t).astype(np.uint8) * 255, cv2.MORPH_OPEN, opn)
        bright = cv2.morphologyEx((diff > t).astype(np.uint8) * 255, cv2.MORPH_OPEN, opn)
        D = _components(dark, geom, lo=0.1)
        B = _components(bright, geom, lo=0.1)
        if len(D) and len(B):
            dd = np.linalg.norm(D[:, None] - B[None], axis=2)
            j = dd.argmin(1)
            ok = dd[np.arange(len(D)), j] < 0.9 * geom.d
            if ok.any():
                off = np.median(B[j[ok]] - D[ok], axis=0) / 2.0
                return D + off
        return D
    raise ValueError(mode)


def estimate_skew(pts: np.ndarray, geom: BrailleGeometry, max_deg: float = 5.0,
                  step: float = 0.05) -> float:
    if len(pts) < 10:
        return 0.0
    c = pts.mean(0)
    q = pts - c
    best, best_s = 0.0, -1.0
    bins = max(4.0, geom.r)  # histogram bin ~ dot radius
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        t = np.deg2rad(a)
        y = -q[:, 0] * np.sin(t) + q[:, 1] * np.cos(t)
        h, _ = np.histogram(y, bins=np.arange(y.min() - bins, y.max() + 2 * bins, bins))
        s = float((h.astype(np.float64) ** 2).sum())
        if s > best_s:
            best, best_s = a, s
    return float(best)


def rotate(gray: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-6:
        return gray
    H, W = gray.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    return cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderValue=int(np.median(gray)))


def estimate_dot_pitch(pts: np.ndarray, geom: BrailleGeometry) -> float:
    """Median distance between vertically neighbouring dots (within a cell column).

    Vertical neighbours are only ever d or 2d apart, whereas horizontal
    neighbours can also be p - d apart (about 1.4 d), which biases the estimate
    on pages with few dots.  Horizontal pairs are used only as a fallback.
    """
    d0 = geom.d
    if len(pts) < 4:
        return d0
    diff = pts[None, :, :] - pts[:, None, :]
    for axis in (1, 0):
        other = 1 - axis
        da, do = np.abs(diff[..., axis]), np.abs(diff[..., other])
        sel = (do < 0.25 * d0) & (da > 0.75 * d0) & (da < 1.25 * d0)
        v = da[sel]
        if len(v) >= 6:
            return float(np.median(v))
    return d0


def _fit_1d(vals: np.ndarray, period0: float, offsets: np.ndarray, rel: float = 0.06,
            n_period: int = 25, res: float = 0.5) -> Tuple[float, float]:
    """Find (period, origin) so that vals ~ origin + n*period + one of offsets."""
    best = (period0, 0.0, np.inf)
    cap = offsets[1] / 2 if len(offsets) > 1 else period0 / 4
    for P in np.linspace(period0 * (1 - rel), period0 * (1 + rel), n_period):
        o = np.arange(0, P, res)
        rmod = np.mod(vals[None, :] - o[:, None], P)          # (O, N)
        dist = np.full(rmod.shape, np.inf)
        for off in list(offsets) + [P + offsets[0]]:
            dist = np.minimum(dist, np.abs(rmod - off))
        cost = np.minimum(dist, cap).sum(1)
        i = int(np.argmin(cost))
        if cost[i] < best[2]:
            best = (float(P), float(o[i]), float(cost[i]))
    return best[0], best[1]


def fit_grid(pts: np.ndarray, geom: BrailleGeometry) -> Dict[str, float]:
    d = estimate_dot_pitch(pts, geom)
    scale = d / geom.d
    L, y0 = _fit_1d(pts[:, 1], geom.L * scale, np.array([0.0, d, 2 * d]))
    p, x0 = _fit_1d(pts[:, 0], geom.p * scale, np.array([0.0, d]))
    # move the origins to the first line / first cell that contains a dot
    y0 = y0 + np.floor((pts[:, 1].min() - y0 + 0.5 * d) / L) * L
    x0 = x0 + np.floor((pts[:, 0].min() - x0 + 0.5 * d) / p) * p
    return {"d": d, "L": L, "p": p, "x0": x0, "y0": y0}


def assign(pts: np.ndarray, g: Dict[str, float], tol: float = 0.45):
    """Map dots to (line j, cell k, bit) ; returns dict and number of rejects."""
    d, L, p, x0, y0 = g["d"], g["L"], g["p"], g["x0"], g["y0"]
    cellmap: Dict[Tuple[int, int], int] = {}
    rejected = 0
    for x, y in pts:
        j = int(np.floor((y - y0 + (L - 2 * d) / 2) / L))
        row = int(round((y - y0 - j * L) / d))
        k = int(np.floor((x - x0 + (p - d) / 2) / p))
        col = int(round((x - x0 - k * p) / d))
        if not (0 <= row <= 2 and 0 <= col <= 1):
            rejected += 1
            continue
        ex, ey = x0 + k * p + col * d, y0 + j * L + row * d
        if abs(x - ex) > tol * d or abs(y - ey) > tol * d:
            rejected += 1
            continue
        bit = col * 3 + row
        cellmap[(j, k)] = cellmap.get((j, k), 0) | (1 << bit)
    return cellmap, rejected


def crop_cell(gray: np.ndarray, g: Dict[str, float], j: int, k: int, mask: int,
              size: int = 28, mode: str = "grid", geom: Optional[BrailleGeometry] = None):
    d, L, p, x0, y0 = g["d"], g["L"], g["p"], g["x0"], g["y0"]
    cx, cy = x0 + k * p + d / 2, y0 + j * L + d          # centre of the cell
    w, h = p, p + d
    if mode == "grid":
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
    elif mode == "bbox":  # legacy tight crop: loses dot position information
        r = (geom.r if geom else d * 0.3) * 1.3
        xs = [x0 + k * p + (b // 3) * d for b in range(6) if mask >> b & 1]
        ys = [y0 + j * L + (b % 3) * d for b in range(6) if mask >> b & 1]
        if not xs:
            xs, ys = [cx], [cy]
        x1, x2 = min(xs) - r, max(xs) + r
        y1, y2 = min(ys) - r, max(ys) + r
        s = max(x2 - x1, y2 - y1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        x1, x2, y1, y2 = mx - s / 2, mx + s / 2, my - s / 2, my + s / 2
    else:
        raise ValueError(mode)
    box = (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))
    H, W = gray.shape
    pad = int(max(w, h))
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    c = padded[box[1] + pad: box[1] + pad + box[3], box[0] + pad: box[0] + pad + box[2]]
    c = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
    return c, box


def segment_page(img: np.ndarray, geom: BrailleGeometry, dot_mode: str = "dark",
                 crop_mode: str = "grid", size: int = 28, max_skew: float = 5.0) -> PageResult:
    gray = to_gray(img)
    pts = detect_dots(gray, geom, dot_mode)
    skew = estimate_skew(pts, geom, max_skew)
    if abs(skew) > 0.05:
        gray = rotate(gray, skew)
        pts = detect_dots(gray, geom, dot_mode)
    if len(pts) == 0:
        return PageResult(gray, skew, {}, [], 0, 0)
    g = fit_grid(pts, geom)
    cellmap, rej = assign(pts, g)
    res = PageResult(gray, skew, g, [], len(pts), rej)
    if not cellmap:
        return res
    lines_idx = sorted({j for j, _ in cellmap})
    for j in lines_idx:
        ks = [k for jj, k in cellmap if jj == j]
        row: List[Cell] = []
        for k in range(min(ks), max(ks) + 1):
            m = cellmap.get((j, k), 0)
            crop, box = crop_cell(gray, g, j, k, m, size, crop_mode, geom)
            row.append(Cell(j, k, m, box, crop))
        res.lines.append(row)
    return res


def cells_to_braille(lines: List[List[Cell]], masks: Optional[List[List[int]]] = None) -> str:
    """Join cells into a Unicode Braille string (one text line per Braille line)."""
    out = []
    for li, row in enumerate(lines):
        ms = masks[li] if masks is not None else [c.rule_mask for c in row]
        out.append("".join(chr(0x2800 + m) for m in ms))
    return "\n".join(out)
