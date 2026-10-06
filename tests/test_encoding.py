"""Every text file opened by the package must state its encoding (Windows uses cp1252 by default)."""
import pathlib
import re


def test_all_text_opens_declare_encoding():
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "gujbraille"
    bad = []
    for f in src.glob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"(?<![.\w])open\(", line) and "encoding=" not in line and "imread" not in line:
                bad.append(f"{f.name}:{i}: {line.strip()}")
    assert not bad, "\n".join(bad)
