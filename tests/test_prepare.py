from gujbraille.prepare import paginate
from gujbraille.transliterate import braille_to_gujarati

TEXT = ("ભારતનું બંધારણ ૨૬ જાન્યુઆરી ૧૯૫૦ના રોજ અમલમાં આવ્યું. " * 30).strip()


def test_paginate_limits_and_content():
    pages = paginate(TEXT, 40, 25)
    words = []
    for brl, guj in pages:
        assert len(brl) <= 25
        assert all(len(l) <= 40 for l in brl)
        for b, g in zip(brl, guj):
            assert braille_to_gujarati(b) == g
        words += " ".join(guj).split()
    assert words == TEXT.split()
