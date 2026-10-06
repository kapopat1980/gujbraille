"""Gujarati Braille (Bharati Braille family) code table.

A Braille cell is represented as an integer bit-mask 0..63 where bit (k-1)
is set when dot k (1..6) is raised.  Dots are numbered in the standard way:

    1 4
    2 5
    3 6

The Unicode Braille Patterns block encodes the same mask: chr(0x2800 + mask).

Source of the letter assignments: Gujarati Braille as standardised under
Bharati Braille (see the README for references).  Two vowels differ from
Hindi (Devanagari) Bharati Braille: in Gujarati, એ = dots 1-5 (⠑) and
ઓ = dots 1-3-5 (⠕), while ઍ = ⠢ and ઑ = ⠭.

The punctuation subset (PUNCTUATION) and the inherent-vowel convention must be
confirmed by the Braille experts who prepared the ground-truth material; they
are kept in a separate dictionary so they can be changed without touching the
letter table.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def dots_to_mask(dots: Iterable[int]) -> int:
    """(1, 3) -> 0b000101"""
    m = 0
    for d in dots:
        if not 1 <= d <= 6:
            raise ValueError(f"dot number out of range: {d}")
        m |= 1 << (d - 1)
    return m


def mask_to_dots(mask: int) -> Tuple[int, ...]:
    return tuple(k + 1 for k in range(6) if mask >> k & 1)


def mask_to_unicode(mask: int) -> str:
    return chr(0x2800 + mask)


def unicode_to_mask(ch: str) -> int:
    cp = ord(ch) - 0x2800
    if not 0 <= cp < 64:
        raise ValueError(f"not a 6-dot Unicode Braille character: {ch!r}")
    return cp


def cells(s: str) -> List[int]:
    """Unicode Braille string -> list of masks (U+2800 is the blank cell)."""
    return [unicode_to_mask(c) for c in s]


def to_unicode(masks: Iterable[int]) -> str:
    return "".join(mask_to_unicode(m) for m in masks)


def dots_label(mask: int) -> str:
    """Human-readable label, e.g. 'dots 1-3' or 'blank'."""
    d = mask_to_dots(mask)
    return "blank" if not d else "dots " + "-".join(map(str, d))


U = mask_to_unicode  # short alias

# ---------------------------------------------------------------------------
# Gujarati Braille table
# ---------------------------------------------------------------------------

# independent vowel -> (braille cells, dependent vowel sign or "" for inherent a)
VOWELS: Dict[str, Tuple[str, str]] = {
    "અ": ("⠁", ""),
    "આ": ("⠜", "ા"),
    "ઇ": ("⠊", "િ"),
    "ઈ": ("⠔", "ી"),
    "ઉ": ("⠥", "ુ"),
    "ઊ": ("⠳", "ૂ"),
    "ઋ": ("⠐⠗", "ૃ"),
    "ઍ": ("⠢", "ૅ"),
    "એ": ("⠑", "ે"),
    "ઐ": ("⠌", "ૈ"),
    "ઑ": ("⠭", "ૉ"),
    "ઓ": ("⠕", "ો"),
    "ઔ": ("⠪", "ૌ"),
}

CONSONANTS: Dict[str, str] = {
    "ક": "⠅", "ખ": "⠨", "ગ": "⠛", "ઘ": "⠣", "ઙ": "⠬",
    "ચ": "⠉", "છ": "⠡", "જ": "⠚", "ઝ": "⠴", "ઞ": "⠒",
    "ટ": "⠾", "ઠ": "⠺", "ડ": "⠫", "ઢ": "⠿", "ણ": "⠼",
    "ત": "⠞", "થ": "⠹", "દ": "⠙", "ધ": "⠮", "ન": "⠝",
    "પ": "⠏", "ફ": "⠖", "બ": "⠃", "ભ": "⠘", "મ": "⠍",
    "ય": "⠽", "ર": "⠗", "લ": "⠇", "વ": "⠧", "શ": "⠩",
    "ષ": "⠯", "સ": "⠎", "હ": "⠓", "ળ": "⠸",
}

# pre-formed conjuncts written with a single cell
CONJUNCTS: Dict[str, str] = {
    "ક્ષ": "⠟",
    "જ્ઞ": "⠱",
}

SIGNS: Dict[str, str] = {
    "્": "⠈",   # virama / halant
    "ં": "⠰",   # anusvara
    "ઃ": "⠠",   # visarga
    "ઁ": "⠄",   # chandrabindu
}

NUMBER_SIGN = "⠼"  # dots 3-4-5-6 (same cell as ણ; resolved by context)
DIGITS: Dict[str, str] = {
    "૧": "⠁", "૨": "⠃", "૩": "⠉", "૪": "⠙", "૫": "⠑",
    "૬": "⠋", "૭": "⠛", "૮": "⠓", "૯": "⠊", "૦": "⠚",
}

# Punctuation subset used in the ground-truth material.  CONFIRM with the
# Braille experts; cells that collide with letters are only interpreted as
# punctuation in word-final position (see transliterate.py).
PUNCTUATION: Dict[str, str] = {
    ".": "⠲",   # full stop
    ",": "⠂",   # comma
    "?": "⠦",   # question mark
    ";": "⠆",   # semicolon
    "-": "⠤",   # hyphen
}

# ---------------------------------------------------------------------------
# derived reverse maps
# ---------------------------------------------------------------------------
CONS_BY_CELL = {v: k for k, v in CONSONANTS.items()}
CONJ_BY_CELL = {v: k for k, v in CONJUNCTS.items()}
VOWEL_BY_CELLS = {v[0]: k for k, v in VOWELS.items()}
MATRA_BY_CELLS = {v[0]: v[1] for v in VOWELS.values()}
SIGN_BY_CELL = {v: k for k, v in SIGNS.items()}
DIGIT_BY_CELL = {v: k for k, v in DIGITS.items()}
PUNCT_BY_CELL = {v: k for k, v in PUNCTUATION.items()}
MATRA_TO_VOWEL_CELLS = {v[1]: v[0] for v in VOWELS.values() if v[1]}


# North American Braille ASCII (BRF), indexed by dot mask 0..63
_BRF = " A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)="


def to_brf(braille_text: str) -> str:
    """Unicode Braille -> Braille ASCII (.brf) for sending to an embosser."""
    out = []
    for ch in braille_text:
        if ch in "\n\f":
            out.append(ch)
        elif ch == " ":
            out.append(" ")
        else:
            out.append(_BRF[unicode_to_mask(ch)])
    return "".join(out)


def from_brf(brf_text: str) -> str:
    rev = {c: i for i, c in enumerate(_BRF)}
    return "".join(ch if ch in "\n\f" else mask_to_unicode(rev[ch.upper()]) for ch in brf_text)


def check_table() -> None:
    """Sanity checks: no two letters share a cell inside the same class."""
    seen: Dict[str, str] = {}
    for table in (CONSONANTS, CONJUNCTS):
        for ch, cell in table.items():
            if cell in seen:
                raise AssertionError(f"{ch} and {seen[cell]} share {cell}")
            seen[cell] = ch
    vseen: Dict[str, str] = {}
    for ch, (cell, _) in VOWELS.items():
        if cell in vseen:
            raise AssertionError(f"{ch} and {vseen[cell]} share {cell}")
        vseen[cell] = ch


def table_rows() -> List[Tuple[str, str, str, str, str]]:
    """Rows (class, Gujarati, Unicode code point(s), Braille, dots) for the paper."""
    rows = []
    for ch, (cell, matra) in VOWELS.items():
        rows.append(("Vowel", ch + (f" / {matra}" if matra else ""),
                     " ".join(f"U+{ord(c):04X}" for c in ch + matra), cell,
                     " + ".join(dots_label(unicode_to_mask(c)) for c in cell)))
    for ch, cell in CONSONANTS.items():
        rows.append(("Consonant", ch, f"U+{ord(ch):04X}", cell,
                     dots_label(unicode_to_mask(cell))))
    for ch, cell in CONJUNCTS.items():
        rows.append(("Conjunct", ch, " ".join(f"U+{ord(c):04X}" for c in ch), cell,
                     dots_label(unicode_to_mask(cell))))
    for ch, cell in SIGNS.items():
        rows.append(("Sign", ch, f"U+{ord(ch):04X}", cell, dots_label(unicode_to_mask(cell))))
    rows.append(("Number sign", "—", "—", NUMBER_SIGN, dots_label(unicode_to_mask(NUMBER_SIGN))))
    for ch, cell in DIGITS.items():
        rows.append(("Digit", ch, f"U+{ord(ch):04X}", NUMBER_SIGN + cell,
                     "(num) " + dots_label(unicode_to_mask(cell))))
    for ch, cell in PUNCTUATION.items():
        rows.append(("Punctuation", ch, f"U+{ord(ch):04X}", cell, dots_label(unicode_to_mask(cell))))
    return rows


check_table()
