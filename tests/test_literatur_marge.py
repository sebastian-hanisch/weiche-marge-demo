"""Literaturwerte (nur aus gelesenen Quellen, Stufe W im Docstring; „Quelle nicht eingesehen heißt: kein Test“).

* Cortes/Vapnik 1995 [W, Zeichenverluste im Text]: Marge 2/‖w‖ als Abstand der Ebenen f = ±1 (Gl. 13) und weiche Marge mit Schlupf (Gl. 22 bis 29).
* Platt 1998 [W]: KKT-Bedingungen (Gl. 12) als Definition am Ergebnis; Toleranz 1e-3 typisch; Vorzeichen von b dort verschieden (u = Σ y α K − b).
* Nicht als Test: Cortes/Vapnik Tab. 2 (Zahlen der USPS-Ziffern, nicht reproduzierbar), Boser/Guyon/Vapnik 1992 (nur Abstract), Platt 1999, Steinwart-Sätze (Quelle nicht eingesehen).
* Die Zählung LOO-Fehler ≤ Zahl der Stützvektoren ist **eigene Herleitung** (bei Cortes/Vapnik Gl. 5 steht eine Erwartungswert-Aussage für trennbare Daten) und hat darum hier keinen Test."""

import numpy as np
import pytest

import wmr_evaluation as ev
import wmr_scenario as Sc


def test_marge_ist_der_abstand_der_ebenen_f_plus_minus_eins_gleich_zwei_durch_norm_w(zwei_punkte):
    """Cortes/Vapnik Gl. 13: die Ebenen f = ±1 haben den Abstand 2/‖w‖. Zwei Punkte x = ±1: w = 1, f = x, die Ebenen liegen bei x = ±1, Abstand 2."""
    m = ev.fit(zwei_punkte, "linear", 1.0, 10.0, 1e-12, 1000)
    w = (m.alpha * zwei_punkte.y) @ zwei_punkte.X
    assert m.margin() == pytest.approx(2.0 / np.linalg.norm(w)) == pytest.approx(2.0)
    xs = np.array([[-1.0], [0.0], [1.0]])
    f = m.decision(xs)
    assert np.allclose(f, [-1.0, 0.0, 1.0], atol=1e-9)  # f = ±1 bei x = ±1: Abstand der Ebenen 2


def test_marge_ist_zwei_durch_norm_w_und_nicht_eins_durch_norm_w_namens_test():
    inst = Sc.gauss_classes(30, 4, 0.0, sep=6.0)
    m = ev.fit(inst, "linear", 1.0, np.inf, 1e-10, 100000)
    w = (m.alpha * inst.y) @ inst.X
    assert m.margin() == pytest.approx(2.0 / np.linalg.norm(w)) and m.margin() != pytest.approx(1.0 / np.linalg.norm(w))
    yf = inst.y * (inst.X @ w + m.b)
    assert yf.min() == pytest.approx(1.0, abs=1e-8)  # die nächsten Punkte liegen genau auf f = ±1


def test_platt_kkt_bedingungen_gl_12_am_ergebnis_als_definition():
    """α = 0 ⇒ y f ≥ 1, 0 < α < C ⇒ y f = 1, α = C ⇒ y f ≤ 1 (mit Toleranz der Abbruchschwelle); das Vorzeichen von b ist hier f = Σ α y K + b (Platt: u = Σ y α K − b)."""
    inst = Sc.gauss_classes(40, 8, 0.1)
    for c in (0.1, 1.0):
        m = ev.fit(inst, "gauss", 1.0, c, 1e-6, 200000)
        yf = inst.y * (m.K @ (m.alpha * inst.y) + m.b)
        t = ev._tol_alpha(c)
        assert (yf[m.alpha <= t] >= 1 - 1e-5).all() and (np.abs(yf[(m.alpha > t) & (m.alpha < c - t)] - 1) <= 1e-5).all() and (yf[m.alpha >= c - t] <= 1 + 1e-5).all()
        u = m.K @ (m.alpha * inst.y) - (-m.b)  # Platt: u = Σ y α K − b mit b_Platt = −b
        assert np.allclose(u, m.K @ (m.alpha * inst.y) + m.b)
