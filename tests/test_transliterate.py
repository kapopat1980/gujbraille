import pytest
from gujbraille.transliterate import braille_to_gujarati, gujarati_to_braille

SENTENCES = [
    "સોનાચાંદી હીરા કુરિયરના ધંધા ઠપ.",
    "પરીક્ષા ૨૦૨૬ જ્ઞાન સ્કૂલ કૃષ્ણ",
    "ભારતનું બંધારણ, ગુજરાતનો ઇતિહાસ?",
    "ઋષિ દુઃખ ઔષધ એક ઓરડો",
]


@pytest.mark.parametrize("s", SENTENCES)
def test_roundtrip(s):
    assert braille_to_gujarati(gujarati_to_braille(s)) == s


def test_specific_cells():
    assert gujarati_to_braille("ક્ષ") == "⠟"
    assert gujarati_to_braille("જ્ઞ") == "⠱"
    assert gujarati_to_braille("કૃ") == "⠅⠐⠗"
    assert gujarati_to_braille("સ્ક") == "⠎⠈⠅"
    assert gujarati_to_braille("૧૨") == "⠼⠁⠃"
    assert gujarati_to_braille("12") == "⠼⠁⠃"
    assert gujarati_to_braille("કં") == "⠅⠰"


def test_number_sign_vs_nna():
    assert braille_to_gujarati("⠼⠁⠃") == "૧૨"
    assert braille_to_gujarati("⠅⠜⠼") == "કાણ"   # word-final ણ, not a number


def test_known_limitation_consonant_plus_independent_vowel():
    # થઈ and થી are written with the same cells; the mapper chooses the matra
    assert gujarati_to_braille("થઈ") == gujarati_to_braille("થી")
    assert braille_to_gujarati(gujarati_to_braille("થઈ")) == "થી"
