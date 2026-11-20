"""Kopier-Tests: `wmr_smo.py` ist die Datei `hmg_smo.py` aus `harte-marge-demo` (nicht geändert, auch nicht im Docstring), `wmr_rng.py` die Vorlage `rng_splitmix64_exact.py`.

Der Hash der Funktion `smo` steht in `harte-marge-demo/tests/test_smo.py` und in `kernmethoden-planung/BAU_2_SMO.md`; liegt der Nachbarordner neben diesem (im Projects-Ordner, nicht in der CI),
werden zusätzlich die Dateien bytegleich verglichen."""

import hashlib
import inspect
from pathlib import Path

import wmr_smo as S

ROOT = Path(__file__).resolve().parent.parent
SMO_SHA256 = "7ef2a5211d6ae6acff96f78d33bcdbeb74560bef2a6131ed64a23e01cf6a835b"  # Funktion smo (Quelltext, Zeilenenden normalisiert)
RNG_SHA256 = "d69b2b2fd67c1234cc0d79249676a662a2189d02803eea7d2edfb04b7a505a3f"  # Datei, Bytegleichheit mit tools/templates/rng_splitmix64_exact.py


def _funktionstext(src: str) -> str:
    """Text der Funktion `smo` (ab `def smo(` bis zum Dateiende) aus einer Quelldatei."""
    return src[src.index("def smo("):]


def test_quelltext_von_smo_ist_unveraendert():
    src = inspect.getsource(S.smo).replace("\r\n", "\n")
    assert hashlib.sha256(src.encode("utf-8")).hexdigest() == SMO_SHA256


def test_smo_datei_ist_die_kopie_aus_stueck_2_falls_vorhanden():
    quelle = ROOT.parent / "harte-marge-demo" / "hmg_smo.py"
    if quelle.exists():  # im Projects-Ordner vorhanden, in der CI nicht
        assert (ROOT / "wmr_smo.py").read_bytes() == quelle.read_bytes()
        assert _funktionstext((ROOT / "wmr_smo.py").read_text(encoding="utf-8")) == _funktionstext(quelle.read_text(encoding="utf-8"))


def test_wmr_rng_ist_die_vorlage_bytegleich():
    mine = ROOT / "wmr_rng.py"
    assert hashlib.sha256(mine.read_bytes()).hexdigest() == RNG_SHA256
    vorlage = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if vorlage.exists():
        assert mine.read_bytes() == vorlage.read_bytes()


def test_keine_andere_zufallsquelle_in_der_app():
    """Kein `numpy.random` und kein Python-`random` außer dem Zufalls-Seed-Knopf der Seitenleiste (`random.randint` in wmr_presets.py)."""
    for p in sorted(ROOT.glob("wmr_*.py")) + [ROOT / "app.py"]:
        text = p.read_text(encoding="utf-8")
        assert "numpy.random" not in text and "np.random" not in text, p.name
        if p.name not in ("wmr_presets.py",):
            assert "import random" not in text and "from random" not in text, p.name
