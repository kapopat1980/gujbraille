import numpy as np
import pytest
from gujbraille.geometry import BrailleGeometry
from gujbraille.render import render_page
from gujbraille.segment import cells_to_braille, segment_page
from gujbraille.transliterate import gujarati_to_braille as g2b

TEXT = ["સોનાચાંદી હીરા કુરિયરના ધંધા ઠપ.", "⠂⠤⠂ ⠤⠂", "પરીક્ષા ૨૦૨૬ જ્ઞાન સ્કૂલ કૃષ્ણ",
        "ઠ ડ ઢ ણ ત થ દ ધ ન પ ફ"]


def lines():
    return [t if t[0] >= "⠀" and t[0] <= "⣿" else g2b(t) for t in TEXT]


@pytest.mark.parametrize("style,mode", [("ink", "dark"), ("emboss", "relief")])
@pytest.mark.parametrize("dpi", [200, 300])
@pytest.mark.parametrize("skew", [0.0, 2.2, -3.5])
def test_exact_recovery(style, mode, dpi, skew):
    bl = [l.replace(" ", "⠀") for l in lines()]
    geom = BrailleGeometry(dpi=dpi)
    img, _ = render_page(bl, geom, style=style, skew_deg=skew, rng=np.random.default_rng(3))
    page = segment_page(img, geom, dot_mode=mode)
    assert cells_to_braille(page.lines).split("\n") == [l.strip("⠀") for l in bl]
    assert abs(page.skew_deg + skew) < 0.3
    for line in page.lines:
        for c in line:
            assert c.crop.shape == (28, 28)


def test_grid_crop_preserves_dot_position_bbox_does_not():
    geom = BrailleGeometry(dpi=200)
    img, _ = render_page(["⠅⠀⠨⠀⠿⠇⠁⠃⠸⠗⠽"], geom, rng=np.random.default_rng(0), noise_sigma=0)
    g = segment_page(img, geom, crop_mode="grid").lines[0]
    b = segment_page(img, geom, crop_mode="bbox").lines[0]
    diff = lambda a, c: float(np.abs(a.astype(int) - c.astype(int)).mean())
    assert diff(g[0].crop, g[2].crop) > 20       # dots 1-3 vs 4-6 look different
    assert diff(b[0].crop, b[2].crop) < diff(g[0].crop, g[2].crop) / 3
