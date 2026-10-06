"""Synthetic Braille page renderer.

USED ONLY FOR UNIT TESTS, SMOKE TESTS AND (OPTIONALLY) PRE-TRAINING.
Results obtained on synthetic pages must never be reported as results on
scanned material.  Two styles are provided:

* ``ink``     - dark dots on white paper (Braille printed with an ordinary
                printer, e.g. for sighted teachers),
* ``emboss``  - raised dots on light paper lit obliquely, giving a bright
                highlight and a dark shadow per dot, as on a flatbed scan of an
                embossed sheet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .braille_table import unicode_to_mask
from .geometry import BrailleGeometry


@dataclass
class RenderedCell:
    line: int
    k: int
    mask: int
    cx: float  # centre of dot column 1, before skew (pixels)
    cy: float  # centre of dot row 1, before skew


def render_page(
    braille_lines: List[str],
    geom: BrailleGeometry,
    style: str = "ink",
    margin_mm: float = 15.0,
    skew_deg: float = 0.0,
    noise_sigma: float = 4.0,
    blur: float = 0.6,
    dot_jitter_mm: float = 0.08,
    dropout: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[RenderedCell]]:
    rng = rng or np.random.default_rng(0)
    d, p, L, r = geom.d, geom.p, geom.L, geom.r
    m = geom.px(margin_mm)
    ncols = max((len(l) for l in braille_lines), default=1)
    W = int(2 * m + ncols * p)
    H = int(2 * m + len(braille_lines) * L)
    ss = 3  # supersampling for anti-aliased dots
    canvas = np.full((H * ss, W * ss), 0.0, np.float32)  # dot "height" map
    cells: List[RenderedCell] = []
    for j, line in enumerate(braille_lines):
        for k, ch in enumerate(line):
            mask = unicode_to_mask(ch)
            cx, cy = m + k * p + (p - d) / 2.0, m + j * L + (L - 2 * d) / 2.0
            cells.append(RenderedCell(j, k, mask, cx, cy))
            for bit in range(6):
                if not mask >> bit & 1:
                    continue
                if dropout and rng.random() < dropout:
                    continue
                col, row = bit // 3, bit % 3
                x = cx + col * d + rng.normal(0, geom.px(dot_jitter_mm))
                y = cy + row * d + rng.normal(0, geom.px(dot_jitter_mm))
                cv2.circle(canvas, (int(x * ss), int(y * ss)), int(r * ss), 1.0, -1, cv2.LINE_AA)
    canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    if style == "ink":
        img = 245.0 - 215.0 * canvas
    elif style == "emboss":
        height = cv2.GaussianBlur(canvas, (0, 0), r * 0.6)
        gy, gx = np.gradient(height)
        shade = -(gx + gy) * r * 2.2          # light from the top-left
        img = 205.0 + 60.0 * np.clip(shade, -1, 1)
        img += rng.normal(0, 6, (H, W)).astype(np.float32)  # paper texture
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
    else:
        raise ValueError(style)
    # uneven illumination
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img += 10.0 * (xx / W - 0.5) + 6.0 * (yy / H - 0.5)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img += rng.normal(0, noise_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if skew_deg:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), skew_deg, 1.0)
        img = cv2.warpAffine(img, M, (W, H), borderValue=int(np.median(img)))
    return img, cells
