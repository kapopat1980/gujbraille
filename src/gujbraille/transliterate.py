"""Rule-based conversion between Gujarati Unicode text and Gujarati Braille.

* ``gujarati_to_braille`` is used to generate ground truth: the exact Braille
  sent to the printer/embosser is derived from known Gujarati source text, so
  every cell label is known before the page is scanned.
* ``braille_to_gujarati`` is the context-sensitive mapping stage of the OBR
  pipeline: it turns a sequence of recognised cells (dot patterns predicted by
  the CNN) into Gujarati Unicode text.

Rules implemented (Bharati / Gujarati Braille conventions):

R1  A consonant cell followed by a vowel cell -> consonant + dependent vowel
    sign (matra).  A vowel cell anywhere else -> independent vowel.
R2  The inherent vowel 'a' is not written; a consonant followed by another
    consonant (no virama) keeps its inherent 'a'.
R3  Conjuncts are written consonant + virama cell (dots 4) + consonant;
    ક્ષ and જ્ઞ have their own single cells.
R4  ઋ / ૃ is the two-cell sequence dots 5 + dots 1-2-3-5.
R5  Number sign (dots 3-4-5-6) at the start of a word followed by a digit cell
    (letters a-j) switches to digit mode until the end of the word.  The same
    cell is ણ in any other position.
R6  Anusvara, visarga and chandrabindu are written after the syllable they
    modify, as in print order.
R7  A blank cell separates words.

Known limitation (documented in the paper): a consonant followed by an
*independent* vowel (e.g. થઈ) is written with the same cells as consonant +
matra (થી) unless the transcriber inserts an explicit 'a' cell; R1 resolves the
pair to the matra reading.
"""
from __future__ import annotations

from typing import List

from . import braille_table as T

BLANK = "⠀"
VIRAMA = "્"
NUKTA = "઼"
UNKNOWN = "�"

_ASCII_TO_GUJ_DIGIT = {str(i): d for i, d in enumerate("૦૧૨૩૪૫૬૭૮૯")}


def gujarati_to_braille(text: str, strict: bool = True) -> str:
    """Gujarati Unicode text -> Unicode Braille string (blank cell for space).

    Newlines are preserved so that line layout of the source can be kept.
    """
    out: List[str] = []
    i, n = 0, len(text)
    in_number = False
    while i < n:
        ch = text[i]
        if ch in _ASCII_TO_GUJ_DIGIT:
            ch = _ASCII_TO_GUJ_DIGIT[ch]
        if ch in T.DIGITS:
            if not in_number:
                out.append(T.NUMBER_SIGN)
                in_number = True
            out.append(T.DIGITS[ch])
            i += 1
            continue
        in_number = False
        if ch == NUKTA:
            i += 1
            continue
        if ch in (" ", "\t"):
            out.append(BLANK)
            i += 1
            continue
        if ch == "\n":
            out.append("\n")
            i += 1
            continue
        # consonant (possibly a pre-formed conjunct)
        if ch in T.CONSONANTS:
            tri = text[i:i + 3]
            if tri in T.CONJUNCTS:
                out.append(T.CONJUNCTS[tri])
                i += 3
            else:
                out.append(T.CONSONANTS[ch])
                i += 1
            while i < n and text[i] == NUKTA:
                i += 1
            if i < n and text[i] == VIRAMA:
                out.append(T.SIGNS[VIRAMA])
                i += 1
            elif i < n and text[i] in T.MATRA_TO_VOWEL_CELLS:
                out.append(T.MATRA_TO_VOWEL_CELLS[text[i]])
                i += 1
            continue
        if ch in T.VOWELS:
            out.append(T.VOWELS[ch][0])
            i += 1
            continue
        if ch in T.SIGNS:
            out.append(T.SIGNS[ch])
            i += 1
            continue
        if ch == "।":
            ch = "."
        if ch in T.PUNCTUATION:
            out.append(T.PUNCTUATION[ch])
            i += 1
            continue
        if strict:
            raise ValueError(f"unsupported character {ch!r} (U+{ord(ch):04X}) at {i}")
        i += 1
    return "".join(out)


def _word_to_gujarati(cells: str) -> str:
    out: List[str] = []
    i, n = 0, len(cells)
    prev_cons = False
    # R5: number mode
    if n >= 2 and cells[0] == T.NUMBER_SIGN and cells[1] in T.DIGIT_BY_CELL:
        i = 1
        while i < n and cells[i] in T.DIGIT_BY_CELL:
            out.append(T.DIGIT_BY_CELL[cells[i]])
            i += 1
    while i < n:
        c = cells[i]
        # R4: two-cell vowel ઋ
        if c == "⠐" and i + 1 < n and cells[i + 1] == "⠗":
            out.append("ૃ" if prev_cons else "ઋ")
            prev_cons = False
            i += 2
            continue
        if c in T.CONJ_BY_CELL:
            out.append(T.CONJ_BY_CELL[c])
            prev_cons = True
        elif c in T.CONS_BY_CELL:
            out.append(T.CONS_BY_CELL[c])
            prev_cons = True
        elif c in T.VOWEL_BY_CELLS:  # R1
            out.append(T.MATRA_BY_CELLS[c] if prev_cons else T.VOWEL_BY_CELLS[c])
            prev_cons = False
        elif c in T.SIGN_BY_CELL:
            out.append(T.SIGN_BY_CELL[c])
            prev_cons = False
        elif c in T.PUNCT_BY_CELL:
            out.append(T.PUNCT_BY_CELL[c])
            prev_cons = False
        else:
            out.append(UNKNOWN)
            prev_cons = False
        i += 1
    return "".join(out)


def braille_to_gujarati(braille: str) -> str:
    """Unicode Braille (blank cell or space = word gap, newline kept) -> Gujarati."""
    lines = braille.split("\n")
    res = []
    for line in lines:
        words = line.replace(" ", BLANK).split(BLANK)
        res.append(" ".join(_word_to_gujarati(w) for w in words))
    return "\n".join(res)


def masks_to_gujarati(masks: List[int]) -> str:
    """Convenience wrapper for predicted dot-pattern ids (0 = blank)."""
    return braille_to_gujarati(T.to_unicode(masks))
