"""Barriere (Innere Punkte, nur zum Zählen): Mini-Instanz von Hand, Startpunkt, Einstellungen als Zählung, Abgrenzung der Einheit (Newton-Schritte)."""

import numpy as np
import pytest

import wmr_barrier as Bar
import wmr_evaluation as ev
import wmr_scenario as Sc
from wmr_smo import dual_value


def test_zwei_punkte_von_hand(zwei_punkte):
    """Q = [[1, 1], [1, 1]]; Dual 2a − 2a² unter 0 < a < C: a = 1/2 bei C ≥ 1/2 (Dualwert 1/2), a = C darunter (Dualwert 2C − 2C²; die Barriere nähert sich der Schranke von innen)."""
    K = ev.gram(zwei_punkte.X, zwei_punkte.X, "linear")
    r = Bar.barrier(K, zwei_punkte.y, 5.0)
    assert r.converged and np.allclose(r.alpha, [0.5, 0.5], atol=1e-5) and dual_value(K, zwei_punkte.y, r.alpha) == pytest.approx(0.5, abs=1e-8)
    r2 = Bar.barrier(K, zwei_punkte.y, 0.25)
    assert np.allclose(r2.alpha, [0.25, 0.25], atol=1e-5) and (r2.alpha < 0.25).all()  # strikt innen


def test_startpunkt_strikt_innen_und_gleichung_erfuellt():
    for npos, nneg in ((5, 5), (3, 9), (9, 3), (1, 7)):
        y = np.r_[np.ones(npos), -np.ones(nneg)]
        a = Bar.start_point(y, 2.0)
        assert abs(np.sum(y * a)) < 1e-12 and (a > 0).all() and (a < 2.0).all()
    with pytest.raises(ValueError):
        Bar.start_point(np.ones(4), 1.0)
    with pytest.raises(ValueError):
        Bar.barrier(np.eye(4), np.array([1.0, -1.0, 1.0, -1.0]), float("inf"))


def test_ergebnis_ist_zulaessig_und_die_luecke_stimmt():
    inst = Sc.gauss_classes(30, 5, 0.0)
    K = ev.gram(inst.X, inst.X, "gauss", 1.0)
    r = Bar.barrier(K, inst.y, 1.0)
    assert r.converged and abs(np.sum(inst.y * r.alpha)) < 1e-9 and (r.alpha > 0).all() and (r.alpha < 1.0).all()
    assert r.gap == pytest.approx(2 * 30 / (1e1 ** (r.outer - 1)), rel=1e-9) and r.gap <= 1e-6  # Lücke 2n/t mit t = mu^(outer − 1)


def test_einstellungen_aendern_die_newton_schritte_als_zaehlung():
    inst = Sc.gauss_classes(40, 6, 0.0)
    K = ev.gram(inst.X, inst.X, "gauss", 1.0)
    zahlen = {}
    for mu in (2.0, 10.0, 50.0):
        for gt in (1e-4, 1e-8):
            r = Bar.barrier(K, inst.y, 1.0, Bar.BarrierSettings(mu=mu, gap_tol=gt))
            assert r.converged
            zahlen[(mu, gt)] = r.newton
    assert len(set(zahlen.values())) > 3  # die Zahl hängt an den Einstellungen (Zweig-Test: jede Einstellung wirkt)
    assert all(zahlen[(m, 1e-4)] < zahlen[(m, 1e-8)] for m in (2.0, 10.0, 50.0))  # engere Abbruchlücke braucht mehr Newton-Schritte


def test_barriere_und_smo_zaehlen_verschiedene_einheiten():
    """Newton-Schritte (jeder löst ein (n+1)×(n+1)-System) gegen Aktualisierungsschritte des SMO (zwei Multiplikatoren): die Zahlen sind nicht dieselbe Einheit, nur die Modellzählung ohne Konstanten vergleicht sie."""
    inst = Sc.gauss_classes(60, 7, 0.0)
    r = ev.exp5("Gauß-Klassen", "gauss", 1.0, 1.0, 60, 7, 0.0, 1e-6, 30000, "second")
    assert r["barrier"].newton >= 20 and r["model"].steps >= 1  # Zählungen, keine Rangfolge: bei n = 40 sind beide Zahlen in der Studie fast gleich (111 und 109)
    assert r["modell_newton"] == pytest.approx(r["barrier"].newton * 60 ** 3 / 3.0) and r["modell_smo"] == r["model"].steps * 60
    assert r["model"].trace is not None and r["model"].trace["step"][0] == 0 and r["model"].trace["gap"][0] == pytest.approx(2.0)  # Index 0 ist der Startzustand: Lücke 2


def test_spur_hat_je_aktualisierungsschritt_einen_punkt_und_endet_im_endzustand():
    import wmr_visualization as V

    r = ev.exp5("Gauß-Klassen", "gauss", 1.0, 10.0, 60, 7, 0.1, 1e-3, 30000, "second")
    tr = r["model"].trace
    assert len(tr["step"]) == r["model"].steps + 1 and tr["step"][-1] == r["model"].steps and tr["gap"][-1] == pytest.approx(r["model"].gap)
    assert np.all(np.diff(tr["dual"]) >= -1e-12)  # Dualwert steigt je Schritt
    fig = V.fig_trace(tr, 1e-3)
    assert len(fig.data[0].x) <= len(tr["step"]) and fig.data[0].x[-1] == r["model"].steps
