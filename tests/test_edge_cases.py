"""Randfälle: n = 2, nur eine Klasse, alle Punkte gleich, extrem kleines und großes C (Meldung statt Rechnen), doppelte Punkte mit verschiedenen Etiketten, unausgewogene Klassen,
Gleichstand bei der Arbeitsmenge (kleinster Index), Regler an den Grenzen."""

import numpy as np
import pytest

import wmr_barrier as Bar
import wmr_constants as C
import wmr_evaluation as ev
import wmr_scenario as Sc
from wmr_smo import smo


def test_zwei_punkte_von_hand(zwei_punkte):
    """Q = [[1, 1], [1, 1]], Dual 2a − 2a²: α = (1/2, 1/2) für C ≥ 1/2, sonst (C, C); w = 1, b = 0, Marge 2 und Dualwert 1/2 (C ≥ 1/2)."""
    m = ev.fit(zwei_punkte, "linear", 1.0, 5.0, 1e-12, 1000)
    assert np.allclose(m.alpha, [0.5, 0.5]) and m.b == pytest.approx(0.0, abs=1e-12) and m.margin() == pytest.approx(2.0) and m.dual == pytest.approx(0.5)
    k = ev.fit(zwei_punkte, "linear", 1.0, 0.25, 1e-12, 1000)
    assert np.allclose(k.alpha, [0.25, 0.25]) and k.margin() == pytest.approx(4.0) and k.dual == pytest.approx(2 * 0.25 - 2 * 0.25 ** 2)  # w = 2C = 1/2, Marge 2/w = 4


def test_nur_eine_klasse_meldet_statt_zu_rechnen():
    inst = Sc.Instance(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]), np.ones(3))
    m = ev.fit(inst, "linear", 1.0, 1.0, 1e-3, 1000)
    assert m.stopped_by == "no_pair" and m.steps == 0 and (m.alpha == 0).all()
    with pytest.raises(ValueError):
        Bar.barrier(m.K, inst.y, 1.0)


def test_alle_punkte_gleich_singulaere_gram_matrix_laeuft_ohne_absturz():
    X = np.zeros((6, 2))
    y = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    for kind in ("linear", "gauss"):
        m = ev.fit(Sc.Instance(X, y), kind, 1.0, 1.0, 1e-6, 10000)
        assert np.isfinite(m.alpha).all() and np.isfinite(m.b) and m.margin() == float("inf")  # w = 0: die Funktion ist konstant, die Marge ist nicht definiert


def test_extrem_kleines_c_alle_alpha_an_der_schranke_und_b_nicht_eindeutig():
    inst = Sc.gauss_classes(20, 3, 0.0)  # ausgewogene Klassen (10 und 10): nur dann liegen bei sehr kleinem C alle α an der Schranke
    m = ev.fit(inst, "gauss", 1.0, 1e-9, 1e-12, 100000)
    assert (m.alpha >= 1e-9 * (1 - 1e-6)).all() and m.groups()["schranke"] == 20 and m.groups()["marge"] == 0
    lo, hi = ev.b_interval(m.K, inst.y, m.alpha, 1e-9)
    assert hi - lo > 1e-3 or not np.isfinite(hi - lo)  # kein freier Stützvektor: b liegt nur in einem Intervall
    flipped = Sc.gauss_classes(20, 3, 0.1)  # unausgewogen (zwei Etiketten vertauscht): Σ yα = 0 verbietet α = C bei allen
    k = ev.fit(flipped, "gauss", 1.0, 1e-9, 1e-12, 100000)
    assert (flipped.y > 0).sum() != (flipped.y < 0).sum() and k.groups()["schranke"] < 20


def test_extrem_grosses_c_schrittgrenze_meldet_nicht_konvergiert():
    inst = Sc.gauss_classes(60, 3, 0.1)
    m = ev.fit(inst, "linear", 1.0, 1e3, 1e-3, 500)
    assert not m.converged and m.stopped_by == "max_steps" and "nicht konvergiert" in m.message and m.steps == 500
    r = ev.exp2("Gauß-Klassen", "linear", 1.0, 60, 3, 0.1, 1e-3, 500)
    assert any(not row["converged"] and row["message"] for row in r["rows"])


def test_doppelte_punkte_mit_verschiedenen_etiketten_sind_in_jedem_kern_nicht_trennbar():
    base = Sc.gauss_classes(11, 3, 0.0, sep=6.0)
    X = np.vstack([base.X, base.X[:1]])
    y = np.r_[base.y, -base.y[0]]
    for kind in ("linear", "gauss"):
        m = ev.fit(Sc.Instance(X, y), kind, 1.0, 10.0, 1e-8, 100000)
        assert m.converged and (m.slack() > 1e-6).any()  # mindestens ein Punkt verletzt die Marge: nicht trennbar, und der Löser meldet nichts


def test_unausgewogene_klassen_barriere_startpunkt_und_smo():
    y = np.array([1.0] * 3 + [-1.0] * 9)
    a = Bar.start_point(y, 1.0)
    assert abs(np.sum(y * a)) < 1e-12 and (a > 0).all() and (a < 1.0).all()
    X = np.array([(i % 3, i // 3) for i in range(12)], float) + 0.0
    X[:3] += 3.0
    inst = Sc.Instance(X, y)
    K = ev.gram(X, X, "gauss", 1.0)
    b = Bar.barrier(K, y, 1.0)
    s = smo(K, y, 1.0, eps=1e-9)
    assert b.converged and abs(s.dual - float(0.5 * (b.alpha.sum() * 2 - b.alpha @ ((y[:, None] * y[None, :] * K) @ b.alpha)))) < 1e-5
    assert ev.fit(inst, "gauss", 1.0, 1.0, 1e-9, 10000).converged


def test_gleichstand_bei_der_arbeitsmenge_nimmt_den_kleinsten_index():
    """Vier Punkte in symmetrischer Lage: die beiden Partner haben gleiche Werte; das Verfahren nimmt (wie die Handrechnung) das erste Maximum, also den kleinsten Index."""
    X = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    K = X @ X.T
    r = smo(K, y, 10.0, eps=1e-12, max_steps=1)
    assert r.steps == 1 and r.alpha[0] > 0 and r.alpha[1] == 0 and r.alpha[2] > 0 and r.alpha[3] == 0  # erster Schritt: Paar (0, 2), nicht (1, 3)


def test_regler_an_den_grenzen_rechnen_ohne_fehler():
    for data in C.DATASETS:
        for kern in C.KERN_OPTIONS:
            for c in (C.C_OPTIONS[0], C.C_OPTIONS[-1]):
                r = ev.exp1(data, kern, C.SIGMA_OPTIONS[0], c, C.N_MIN, 0, 0.1, 1e-3, C.STEPS_OPTIONS[0])
                assert np.isfinite(r["gap"]) and r["margin"] > 0
    r = ev.exp1("Gauß-Klassen", "gauss", C.SIGMA_OPTIONS[-1], 1000.0, C.N_MAX, C.SEED_MAX, 0.1, 1e-6, C.STEPS_OPTIONS[0])
    assert not r["model"].converged and r["model"].steps == C.STEPS_OPTIONS[0]


def test_loo_randfaelle_einklassiger_rest_zaehlt_als_fehler_und_wird_ausgewiesen():
    inst = Sc.Instance(np.array([[-1.0], [1.0], [3.0]]), np.array([-1.0, 1.0, 1.0]))
    r = ev.loo(inst, "linear", 1.0, 0.1)
    assert r["one_class"].tolist() == [True, False, False] and r["n_sv"] == 2 and r["loo_worst"] == 3 and r["loo_solver"] == 2  # das 3-Punkte-Gegenbeispiel der Planung (Teil 4b)
    kind, text = ev.verdict_loo(r)
    assert kind == "warning" and "nur eine Klasse" in text
