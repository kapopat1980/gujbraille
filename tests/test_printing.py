import os

import numpy as np
import pytest

from gujbraille.geometry import BrailleGeometry
from gujbraille.prepare import write_pages
from gujbraille.printing import import_scans, print_pdf

pymupdf = pytest.importorskip("pymupdf")

from gujbraille.synthetic import random_text

# non-repeating pseudo-Gujarati text (repeated sentences would make pages identical)
TEXT = " ".join(random_text(np.random.default_rng(0), 90, 30))


def test_print_and_import_roundtrip(tmp_path):
    gt = tmp_path / "gt"
    ids = write_pages(TEXT, str(gt), "DOC-A", cells_per_line=30, lines_per_page=25)
    pdf = tmp_path / "print.pdf"
    n, order = print_pdf(str(gt), str(pdf), BrailleGeometry())
    assert n == len(ids) and order == sorted(ids)
    doc = pymupdf.open(str(pdf))
    assert abs(doc[0].rect.width - 595.3) < 1 and abs(doc[0].rect.height - 841.9) < 1   # A4
    # "scan": rasterise at 300 dpi in reverse order, last sheet upside down
    import cv2
    scans = tmp_path / "scans"
    scans.mkdir()
    for i, page in enumerate(reversed(list(doc))):
        pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        a = cv2.GaussianBlur(a, (0, 0), 1.0)
        if i == 0:
            a = cv2.rotate(a, cv2.ROTATE_180)
        cv2.imwrite(str(scans / f"scan{i:03d}.png"), a)
    res = import_scans([str(scans)], str(gt), str(tmp_path / "data"), BrailleGeometry(dpi=300))
    assert res["matched"] == len(ids) and not res["gt_pages_without_scan"]
    assert os.path.exists(tmp_path / "data" / "pages.csv")


def test_too_wide_for_a4_is_refused(tmp_path):
    gt = tmp_path / "gt"
    write_pages(TEXT, str(gt), "DOC-B", cells_per_line=40, lines_per_page=25)
    with pytest.raises(ValueError):
        print_pdf(str(gt), str(tmp_path / "x.pdf"), BrailleGeometry())


def test_select_pages_keeps_fullest_per_doc():
    from gujbraille.printing import select_pages
    pages = {"A_001": ["⠁⠁⠁"], "A_002": ["⠁"], "A_003": ["⠁⠁"], "B_001": ["⠃"]}
    assert list(select_pages(pages, 2)) == ["A_001", "A_003", "B_001"]
    assert select_pages(pages, 0) == pages


def test_auto_dpi_from_a4_sheet():
    from gujbraille.printing import _auto_dpi
    a4_200dpi = np.zeros((2339, 1654), np.uint8)
    assert abs(_auto_dpi(a4_200dpi, 300) - 200) < 1
    assert _auto_dpi(np.zeros((1000, 1000), np.uint8), 300) == 300   # not A4: fallback
