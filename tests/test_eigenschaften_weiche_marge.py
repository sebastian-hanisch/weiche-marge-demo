"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Invarianten des SMO und der weichen Marge auf je 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed (Bedingungen im Test, unter 10 s).

Σ yα = 0 und 0 ≤ α ≤ C nach jedem Schritt; Dualwert steigt je Schritt; Labels vertauschen; Punkte permutieren; Verschiebung (linear mit Bias); Skalierung von Daten und σ;
Daten verdoppeln und C halbieren; Dualwert, ‖w‖ und Schlupfsumme monoton in C; harte Marge ab C ≥ max α; gleicher Zufalls-Seed gleiche Ausgabe."""

import numpy as np
import pytest

import wmr_evaluation as ev
import wmr_scenario as Sc
from wmr_rng import SplitMix64
from wmr_smo import smo


def _inst(seed, n=20, flip=0.1, kind=0):
    return Sc.make_instance("Gauß-Klassen" if kind == 0 else "Ringe mit Streuung", n, seed, flip)


@pytest.mark.parametrize("kind,sigma", (("linear", 1.0), ("gauss", 1.0)))
def test_nach_jedem_schritt_gleichung_schranken_und_steigender_dualwert(kind, sigma):
    for seed in range(8):
        inst = _inst(1000 + seed, 14, 0.1, seed % 2)
        c = (0.1, 1.0, 10.0)[seed % 3]
        K = ev.gram(inst.X, inst.X, kind, sigma)
        last = 0.0
        final = smo(K, inst.y, c, eps=1e-8, max_steps=100000)
        for k in range(0, final.steps + 1):
            r = smo(K, inst.y, c, eps=1e-8, max_steps=k)
            assert abs(np.sum(inst.y * r.alpha)) < 1e-10 and (r.alpha >= 0).all() and (r.alpha <= c + 1e-12).all()
            assert r.dual >= last - 1e-12  # Dualwert steigt in jedem Schritt (zu maximieren)
            last = r.dual
            assert r.steps == min(k, final.steps)


def test_etiketten_vertauschen_gleiche_alpha_b_wechselt_das_vorzeichen():
    for seed in range(20):
        inst = _inst(1100 + seed)
        neg = Sc.Instance(inst.X, -inst.y)
        a = ev.fit(inst, "gauss", 1.0, 1.0, 1e-10, 100000)
        b = ev.fit(neg, "gauss", 1.0, 1.0, 1e-10, 100000)
        assert np.abs(a.alpha - b.alpha).max() < 1e-6 and a.b == pytest.approx(-b.b, abs=1e-6)
        z = Sc.test_set("Gauß-Klassen", 50, 3).X
        assert np.abs(a.decision(z) + b.decision(z)).max() < 1e-5  # f → −f


def test_punkte_permutieren_gleiche_loesung():
    for seed in range(20):
        inst = _inst(1200 + seed)
        p = np.arange(inst.n)
        SplitMix64(seed).shuffle(list(p))
        idx = list(range(inst.n))
        SplitMix64(seed + 1).shuffle(idx)
        perm = Sc.Instance(inst.X[idx], inst.y[idx])
        a = ev.fit(inst, "gauss", 1.0, 1.0, 1e-10, 100000)
        b = ev.fit(perm, "gauss", 1.0, 1.0, 1e-10, 100000)
        assert a.dual == pytest.approx(b.dual, rel=1e-8) and np.abs(a.alpha[idx] - b.alpha).max() < 1e-5


def test_verschiebung_aller_punkte_linearer_kern_mit_bias_gleiche_alpha_und_w():
    for seed in range(20):
        inst = _inst(1300 + seed, kind=seed % 2)
        shift = np.array([3.0, -2.0])
        moved = Sc.Instance(inst.X + shift, inst.y)
        a = ev.fit(inst, "linear", 1.0, 1.0, 1e-10, 200000)
        b = ev.fit(moved, "linear", 1.0, 1.0, 1e-10, 200000)
        wa, wb = (a.alpha * inst.y) @ inst.X, (b.alpha * inst.y) @ moved.X
        # α ist beim linearen Kern in 2D nicht immer eindeutig; w ist es (und die Marge)
        assert np.abs(wa - wb).max() < 1e-4
        if np.linalg.norm(wa) > 1e-6:  # bei w = 0 (lineares Modell auf den Ringen, alle α an der Schranke) ist die Marge nicht definiert
            assert a.margin() == pytest.approx(b.margin(), rel=1e-4)


def test_daten_und_sigma_mit_demselben_faktor_skalieren_gleiche_alpha():
    for seed in range(20):
        inst = _inst(1400 + seed)
        for f in (0.1, 7.0):
            scaled = Sc.Instance(inst.X * f, inst.y)
            a = ev.fit(inst, "gauss", 1.0, 1.0, 1e-10, 100000)
            b = ev.fit(scaled, "gauss", f, 1.0, 1e-10, 100000)
            assert np.abs(a.alpha - b.alpha).max() < 1e-6 and a.b == pytest.approx(b.b, abs=1e-6)


def test_daten_verdoppeln_und_c_halbieren_gleiches_w_und_b():
    """Jeder Punkt doppelt mit halbem C (je Punkt zusammen derselbe Preis): gleiche Entscheidungsfunktion (gemessen 1,1e-12 bei n = 30)."""
    for seed in range(20):
        inst = _inst(1500 + seed, 24)
        dbl = Sc.Instance(np.vstack([inst.X, inst.X]), np.r_[inst.y, inst.y])
        a = ev.fit(inst, "gauss", 1.0, 2.0, 1e-10, 200000)
        b = ev.fit(dbl, "gauss", 1.0, 1.0, 1e-10, 200000)
        z = Sc.test_set("Gauß-Klassen", 60, 9).X
        assert np.abs(a.decision(z) - b.decision(z)).max() < 1e-6


@pytest.mark.parametrize("kind,sigma", (("linear", 1.0), ("gauss", 1.0)))
def test_dualwert_norm_von_w_und_schlupfsumme_sind_monoton_in_c(kind, sigma):
    for seed in range(20):
        inst = _inst(1600 + seed, 20, 0.1, seed % 2)
        rows = []
        for c in (0.01, 0.1, 1.0, 10.0):
            m = ev.fit(inst, kind, sigma, c, 1e-10, 300000)
            assert m.converged
            rows.append((m.dual, m.w_norm(), float(m.slack().sum())))
        for (d0, w0, x0), (d1, w1, x1) in zip(rows, rows[1:]):
            assert d1 >= d0 - 1e-7 and w1 >= w0 - 1e-6 and x1 <= x0 + 1e-6  # Dualwert und ‖w‖ nicht fallend, Σξ nicht steigend


def test_ab_c_gleich_groesstes_alpha_der_harten_marge_sind_marge_und_alpha_konstant():
    for seed in range(10):
        inst = Sc.gauss_classes(20, 1700 + seed, 0.0, sep=6.0)
        hart = ev.fit(inst, "linear", 1.0, np.inf, 1e-10, 100000)
        amax = float(hart.alpha.max())
        for fac in (1.0, 3.0, 100.0):
            m = ev.fit(inst, "linear", 1.0, fac * amax, 1e-10, 100000)
            assert m.margin() == pytest.approx(hart.margin(), rel=1e-6) and np.abs(m.alpha - hart.alpha).max() < 1e-6


def test_gleicher_zufalls_seed_gleiche_ausgabe_und_eingabe_bleibt_unveraendert():
    inst = _inst(1800, 30)
    K = ev.gram(inst.X, inst.X, "gauss", 1.0)
    K0, y0 = K.copy(), inst.y.copy()
    for wss in ("second", "first", "random"):
        a = smo(K, inst.y, 1.0, eps=1e-8, wss=wss, rng=SplitMix64(5) if wss == "random" else None)
        b = smo(K, inst.y, 1.0, eps=1e-8, wss=wss, rng=SplitMix64(5) if wss == "random" else None)
        assert np.array_equal(a.alpha, b.alpha) and a.steps == b.steps
    assert np.array_equal(K, K0) and np.array_equal(inst.y, y0)
    assert ev.exp1("Gauß-Klassen", "gauss", 1.0, 1.0, 40, 5, 0.1, 1e-3, 30000)["test_err"] == ev.exp1("Gauß-Klassen", "gauss", 1.0, 1.0, 40, 5, 0.1, 1e-3, 30000)["test_err"]


def test_die_drei_arbeitsmengen_geben_denselben_dualwert_aber_verschiedene_schrittzahlen():
    n_verschieden = 0
    for seed in range(10):
        inst = _inst(1900 + seed, 40, 0.05)
        K = ev.gram(inst.X, inst.X, "gauss", 1.0)
        res = {w: smo(K, inst.y, 10.0, eps=1e-8, wss=w, rng=SplitMix64(seed)) for w in ("second", "first", "random")}
        d = [r.dual for r in res.values()]
        assert max(d) - min(d) < 1e-6
        n_verschieden += int(len({r.steps for r in res.values()}) > 1)
    assert n_verschieden >= 8  # Zweig-Test: jede Variante läuft wirklich anders
