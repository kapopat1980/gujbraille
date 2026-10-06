"""Physical Braille geometry, converted to pixels for a given scan resolution.

Defaults follow common embosser specifications (dot pitch 2.5 mm, cell pitch
6.0 mm, line pitch 10.0 mm, dot base diameter 1.5 mm).  Measure the real
material and override these values in the config file: every pixel threshold
in the pipeline is derived from them, so nothing is tuned to one image size.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

MM_PER_INCH = 25.4


@dataclass
class BrailleGeometry:
    dpi: float = 300.0
    dot_pitch_mm: float = 2.5     # centre-to-centre, dots 1->4 and 1->2
    cell_pitch_mm: float = 6.0    # centre-to-centre of adjacent cells
    line_pitch_mm: float = 10.0   # centre-to-centre of adjacent lines
    dot_diameter_mm: float = 1.5

    def px(self, mm: float) -> float:
        return mm * self.dpi / MM_PER_INCH

    @property
    def d(self) -> float:
        return self.px(self.dot_pitch_mm)

    @property
    def p(self) -> float:
        return self.px(self.cell_pitch_mm)

    @property
    def L(self) -> float:
        return self.px(self.line_pitch_mm)

    @property
    def r(self) -> float:
        return self.px(self.dot_diameter_mm) / 2.0

    def to_dict(self) -> dict:
        return asdict(self)
