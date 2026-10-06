from gujbraille import braille_table as T
from gujbraille.transliterate import braille_to_gujarati


def test_mask_unicode_roundtrip():
    for m in range(64):
        assert T.unicode_to_mask(T.mask_to_unicode(m)) == m
        assert T.dots_to_mask(T.mask_to_dots(m)) == m


def test_no_shared_cells_within_letter_classes():
    T.check_table()
    letters = list(T.CONSONANTS.values()) + list(T.CONJUNCTS.values())
    assert len(letters) == len(set(letters))
    vowels = [v[0] for v in T.VOWELS.values()]
    assert len(vowels) == len(set(vowels))
    assert not set(vowels) & set(letters)
    assert not set(T.PUNCTUATION.values()) & (set(letters) | set(vowels) | set(T.SIGNS.values()))


def test_figure14_sample_decodes_to_alphabet():
    # Cells decoded from the sample image of the original manuscript (Fig. 14)
    rows = {
        "⠅⠨⠛⠣⠬⠉⠡⠚⠴⠒⠾": "કખગઘઙચછજઝઞટ",
        "⠺⠫⠿⠼⠞⠹⠙⠮⠝⠏⠖": "ઠડઢણતથદધનપફ",
        "⠃⠘⠍⠽⠗⠇⠧⠸⠩⠯⠎": "બભમયરલવળશષસ",
    }
    for cells, guj in rows.items():
        # letters are separated by blank cells in the sample, decode one by one
        assert "".join(braille_to_gujarati(c) for c in cells) == guj
    assert "".join(braille_to_gujarati(c) for c in "⠁⠜⠊⠔⠥⠳⠑⠌⠕⠪") == "અઆઇઈઉઊએઐઓઔ"


def test_brf_roundtrip():
    s = "".join(chr(0x2800 + i) for i in range(64))
    assert T.from_brf(T.to_brf(s)) == s
    assert T.to_brf("⠁⠃⠉⠿") == "ABC="
