"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4, §11a Punkt 13): die Zahl-Fragmente der README (`tests/readme_facts.py`) stehen alle in der README, die wichtigsten Zahlen werden aus den Rohwerten
von `wmr_frozen.json` unabhängig von `wmr_studies.py` nachgerechnet, dazu Bänder für die Aussagen („kaum“, „schadet“, „meldet nichts“, „Schranke gilt“) und die Testzahl. Zahlen, die nur als Abweichung von der
Vormessung genannt sind, haben keinen Test (Zeile „Zahl → Test“ der README)."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wmr_texts as T
from readme_facts import D, facts
from wmr_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
F = facts()


def _flat(v):
    return [v] if isinstance(v, str) else list(v)


def test_annahmentabelle_der_app_steht_wortgleich_in_der_readme():
    assert T.assumptions_md().strip() in README


def test_testzahl_der_readme_gleich_der_suite():
    m = re.search(r"(\d+) Tests \(`pytest --collect-only -q tests`\)", README)
    assert m, "Testzahl nicht in der README"
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT, capture_output=True, text=True, timeout=600).stdout
    n = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert int(m.group(1)) == n


def test_jedes_zahlfragment_der_readme_steht_in_der_readme():
    fehlt = [(k, frag) for k, v in F.items() for frag in _flat(v) if frag not in README]
    assert not fehlt, fehlt


# ---- unabhängige Nachrechnung aus den Rohwerten (ohne wmr_studies) ---------------------------------------------------------------------------------------------------------------------------------
def _de(x, nd=4):
    return f"{x:.{nd}f}".replace(".", ",")


def test_c_wirkung_spannen_und_randwerte_aus_den_rohwerten():
    for key, g_span, g_ab in (("gauss_lin", (0.0085, 0.0151), (0.0024, 0.0050)), ("gauss_rbf", (0.0907, 0.0822), None)):
        for gi, g in enumerate(("1", "2")):
            means = np.array(D["c_wirkung"]["faelle"][key]["gruppen"][g]["test"]).mean(axis=1)
            assert means.max() - means.min() == pytest.approx(g_span[gi], abs=6e-5)
            if g_ab:
                assert means[1:].max() - means[1:].min() == pytest.approx(g_ab[gi], abs=6e-5)
    assert f"{_de(0.0085)} und {_de(0.0151)}" in README
    # lineare Kerne ohne Rauschen: Spanne unter 0,02 in beiden Gruppen (Zählschwelle für „C ändert den Testfehler wenig“), mit Rauschen über 0,05
    sp = lambda k: [np.ptp(np.array(D["c_wirkung"]["faelle"][k]["gruppen"][g]["test"]).mean(axis=1)) for g in ("1", "2")]
    assert max(sp("gauss_lin")) <= 0.02 and min(sp("gauss_lin_flip")) > 0.05 and min(sp("gauss_rbf")) > 0.05
    # die benannte Ausnahme: mit Rauschen ist der lineare Kern ab C = 0,1 flach (Spanne höchstens 0,01), bei C = 0,001 nicht
    for g in ("1", "2"):
        m = np.array(D["c_wirkung"]["faelle"]["gauss_lin_flip"]["gruppen"][g]["test"]).mean(axis=1)
        assert np.ptp(m[2:]) <= 0.01 and m[0] - m[2] > 0.15


def test_gauss_kern_grosse_c_schaden_aus_den_rohwerten():
    for g in ("1", "2"):
        m = np.array(D["c_wirkung"]["faelle"]["gauss_rbf"]["gruppen"][g]["test"]).mean(axis=1)
        assert m[-1] - m[0] > 0.07 and np.all(np.diff(m) >= -0.003)  # Fehler wächst mit C (bis auf Rauschen 0,003 zwischen benachbarten C)
        assert D["bayes"]["gauss"] < m.min()  # kein Mittel unter dem Bayes-Fehler


def test_loeser_zaehlung_aus_den_rohwerten():
    for g in ("1", "2"):
        warn_unb = gap_over = cells = 0
        gmax = 0.0
        warn_cap_cells = []
        for key, case in D["c_wirkung"]["faelle"].items():
            gr = case["gruppen"][g]
            cells += np.array(gr["test"]).size
            warn_unb += int(np.sum(gr["warn_unbegrenzt"]))
            gmax = max(gmax, float(np.max(gr["gap"])))
            gap_over += int((np.array(gr["gap"]) > 1e-3 + 1e-9).sum())
            warn_cap_cells += [(key, c) for ci, c in enumerate(D["c_wirkung"]["C"]) if np.sum(gr["warn100k"][ci])]
        assert cells == 1260 and warn_unb == 0 and gmax < 1.7e-3 and gmax > 1e-3  # die Lücke liegt nur knapp über der Toleranz (libsvm schrumpft)
        assert all(k.startswith("gauss_lin") and c >= 100.0 for k, c in warn_cap_cells)  # Warnungen nur beim linearen Kern und großem C
    assert "strukturell" in README and "keine Messung" in README


def test_gegenprobe_warnung_nimmt_mit_c_zu_und_ist_bei_c_1e2_null():
    w = np.array(D["warn_harte"]["warn"])
    assert w.shape == (6, 4) and w[:, 0].sum() == 0 and w[:, 1].sum() <= 4 and w[:, 2].min() >= 5 and w[:, 3].min() >= 5
    assert "Die Falle führt `harte-marge-demo`" in README


def test_wahl_aus_den_rohwerten_gepaarte_differenz_und_optimismus():
    for g in ("1", "2"):
        gd = D["wahl60"]["gruppen"][g]["flip0"]
        te, cv, dflt = np.array(gd["te"]), np.array(gd["cv"]), np.array(gd["default"])
        pick = te[np.arange(30), cv.argmin(axis=1)]
        diff = pick - dflt
        se = diff.std(ddof=1) / np.sqrt(30)
        assert diff.mean() > 2.5 * se  # ohne Rauschen ist die Kreuzvalidierung hier im Mittel schlechter (Band: über 2,5 Standardfehler in beiden Gruppen)
        gn = D["wahl60"]["gruppen"][g]["flip10"]
        dn = np.array(gn["te"])[np.arange(30), np.array(gn["cv"]).argmin(axis=1)] - np.array(gn["default"])
        assert abs(dn.mean()) < 1.0 * dn.std(ddof=1) / np.sqrt(30)  # mit Rauschen kein Unterschied (unter einem Standardfehler)
        opt = {}
        for m in (30, 100, 500):  # Optimismus der Gewinnerauswahl nimmt mit der Größe der Validierungsmenge ab
            va = np.array(gd["va"][str(m)])
            k = va.argmin(axis=1)
            opt[m] = float((te[np.arange(30), k] - va[np.arange(30), k]).mean())
        assert opt[30] > 0.03 and opt[100] > 0.01 and opt[500] < 0.01 and opt[30] > opt[100] > opt[500]
    p = [np.array(D["wahl200"]["gruppen"][g]["flip0"]["te"])[np.arange(20), np.array(D["wahl200"]["gruppen"][g]["flip0"]["cv"]).argmin(axis=1)] - np.array(D["wahl200"]["gruppen"][g]["flip0"]["default"]) for g in ("1", "2", "3")]
    pooled = np.concatenate(p)
    assert pooled.mean() > 0 and all(v.mean() > 0 for v in p) and pooled.mean() / (pooled.std(ddof=1) / np.sqrt(60)) > 3.0


def test_leave_one_out_aus_den_rohwerten():
    e1, e5 = D["loo"]["e1"], D["loo"]["e5"]
    assert len(e1) == 300 and sum(r["loo_solver"] > r["nsv"] for r in e1) == 0  # mit dem b des Lösers 0 Verletzungen in 300 Instanzen (Zählung, kein Satz)
    assert len(e5) == 600 and sum(r["loo_solver"] > r["nsv"] for r in e5) == 0 and sum(r["loo_worst"] > r["nsv"] for r in e5) >= 10  # mit ungünstigstem b Gegenbeispiele
    assert all(r["one_class"] == 0 for r in e5) and all(r["ties"] == 0 for r in e1)
    assert "nicht für jedes optimale b" not in README  # Absolutaussage nur als Zählung mit Voraussetzung (siehe Befund und Grenzen)
    assert "Die Zählung gilt für das b des Lösers" in README


def test_schritte_aus_den_rohwerten():
    own = np.array(D["schritte_c"]["faelle"]["gauss"]["own"])
    assert np.all(np.diff(own.mean(axis=1)) > 0) and own.mean(axis=1)[-1] > 100 * own.mean(axis=1)[0]  # Schritte wachsen mit C
    n = np.array(D["schritte_n"]["n"], float)
    steps = np.array(D["schritte_n"]["own"], float)
    slope = np.polyfit(np.log(n), np.log(steps.mean(axis=1)), 1)[0]
    assert 1.1 < slope < 1.5 and f"{slope:.2f}".replace(".", ",") in README  # „nicht etwa linear“ und nicht über 1,5
    for s_ in (1.1, 1.5):
        assert s_ != slope
    assert "Das Band der Steigung schließt 1 nicht ein" in README
    boot = []
    for s_ in range(300):  # Bootstrap über die 20 Zufalls-Seeds, hier mit einem anderen Strom (SplitMix64, Zufalls-Seed 31) als in `wmr_studies.steps_n` nachgerechnet
        rng = SplitMix64(31 + s_)
        idx = [rng.below(20) for _ in range(20)]
        boot.append(np.polyfit(np.log(n), np.log(steps[:, idx].mean(axis=1)), 1)[0])
    assert np.percentile(boot, 5) > 1.0  # das 90-%-Band der Steigung schließt 1 nicht ein


def test_barriere_und_live_vs_studie_aus_den_rohwerten():
    nt = np.array(D["barriere"]["newton"], float)
    assert nt.min() >= 50 and nt.max() <= 300 and nt.mean(axis=1)[-1] > nt.mean(axis=1)[0]
    ab = [np.mean(v) for v in D["barriere"]["abhaengigkeit_n80"].values()]
    assert max(ab) / min(ab) > 3  # die Zahl hängt an den Einstellungen
    lv = D["live_vs_studie"]
    assert lv["dual_rel_max"] < 1e-8 and 0.5 < min(lv["verhaeltnis"]) and max(lv["verhaeltnis"]) < 2.5 and lv["gleich"] < 0.2 * lv["instanzen"]


def test_absolutaussagen_stehen_nur_mit_ihrer_voraussetzung():
    """Gegenproben (DEMO-PLAYBOOK §11a): keine bloßen Absolutwörter über Löser, Schranke, Wirkung von C oder Schritte in der README."""
    for verboten in ("Der Löser warnt nie", "Der Löser meldet nie", "gilt immer", "C wirkt kaum.", "etwa linear in n gilt", "0 Warnungen in 840", "keine einzige"):
        assert verboten not in README, verboten
    assert "nicht trennbar" in README and "strukturell" in README and "mit dem b des Lösers" in README
    assert not re.search(r"(trägt|tragen|tragend)", README)  # Wortwahl-Regel (nicht „beträgt“)


def test_zu_kleines_c_ist_eine_mischung_mehrheitsklasse_bei_unausgewogenen_klassen():
    """Das Mittel bei C = 0,001 mit vertauschten Labels (Gauß σ = 1) ist eine Mischung: Testfehler genau 0,5 (alle Testpunkte einer Klasse) in den Zufalls-Seeds mit unausgewogenen Klassen, sonst nahe dem besten C."""
    from readme_facts import majority_check

    mj = majority_check()
    assert mj["n_unausgewogen"] + mj["n_ausgewogen"] == 30 and mj["n_alle_mehrheit"] == mj["n_unausgewogen"] > 0  # jede unausgewogene Instanz: alle 2 000 Testpunkte in der Mehrheitsklasse
    assert mj["err_max"] < 0.3 and mj["n_ausgewogen"] > 0  # ausgewogene Instanzen: kein Umkippen
    for key in ("gauss_rbf_flip", "gauss_lin_flip"):
        for g in ("1", "2"):
            t = np.array(D["c_wirkung"]["faelle"][key]["gruppen"][g]["test"])[0]  # C = 0,001 (libsvm)
            halb = np.isclose(t, 0.5, atol=1e-12)
            assert 8 <= halb.sum() <= 25 and t[~halb].max() < 0.26 and t[~halb].min() > 0.14  # Mischung: genau 0,5 oder um 0,15 bis 0,25
    assert "Mischung" in README
