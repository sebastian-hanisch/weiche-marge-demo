"""Handbeispiel „Fünf Punkte auf der Geraden“ in Brüchen (Aufzählung aller 3⁵ = 243 Zuordnungen, kein Optimierer): die Prüfungen aus `handbeispiele_03.py` der Planung als echte Tests,
mit denselben engen Toleranzen (Brüche exakt). Der Rechenweg der Tests unterscheidet sich von dem der App (`wmr_exact.py`): geschlossene Formeln von Hand, Minimierung des Primals über b
an den Knickstellen, SMO in Brüchen (zwei Implementierungen), Primal mit Schlupfvariablen per SLSQP, libsvm (`SVC`)."""

from fractions import Fraction as F

import numpy as np
import pytest

import wmr_exact as Ex
import wmr_evaluation as ev
import wmr_scenario as Sc
from exact_helpers import HX, HY, b_intervall_primal, dual_wert, primal_wert, smo_exakt, smo_exakt_paare

C7 = (F(1, 8), F(1, 4), F(1, 2), F(1), F(2), F(4), F(100))


def _formel(C):
    """Von Hand: C >= 1/3: α = (0, (3C+2)/9, C, C, (3C+2)/9), w = 2/3, b = −5/3, Dualwert 2/9 + 8C/3; C <= 1/3: alle vier bei C, w = 2C, Dualwert 4C − 2C², b in [−1−2C, min(−1, 1−8C)]."""
    if C >= F(1, 3):
        return (F(0), (3 * C + 2) / 9, C, C, (3 * C + 2) / 9), F(2, 3), F(-5, 3), F(-5, 3), F(2, 9) + F(8, 3) * C
    return (F(0), C, C, C, C), 2 * C, -1 - 2 * C, min(F(-1), 1 - 8 * C), 4 * C - 2 * C * C


@pytest.mark.parametrize("C", C7 + (F(3, 10), F(1, 3)))
def test_aufzaehlung_gibt_genau_eine_loesung_gleich_der_handformel(C):
    z = Ex.knick_zeile(C)
    alpha, w, blo, bhi, dual = _formel(C)
    assert z["alpha"] == alpha and z["w"] == w and (z["b_lo"], z["b_hi"]) == (blo, bhi) and z["dual"] == dual
    assert sum(HY[i] * z["alpha"][i] for i in range(5)) == 0


@pytest.mark.parametrize("C", C7 + (F(3, 10),))
def test_primal_gleich_dual_exakt_auch_an_beiden_enden_des_b_intervalls(C):
    z = Ex.knick_zeile(C)
    assert z["primal"] == z["primal_hi"] == z["dual"] == dual_wert(z["alpha"], HX, HY)
    assert primal_wert(z["alpha"], z["b_lo"], HX, HY, C)[0] == z["dual"]


def test_knick_bei_c_gleich_ein_drittel_beide_formeln_geben_zehn_neuntel():
    assert F(2, 9) + F(8, 3) * F(1, 3) == F(10, 9) == 4 * F(1, 3) - 2 * F(1, 9)
    assert Ex.knick_zeile(F(1, 3))["dual"] == F(10, 9)


def test_b_intervall_gitter_zwei_rechenwege_und_die_alte_obergrenze_ist_falsch():
    """C = k/120 (Auswahl von k bis 60, dazu 3/10 und 7/20): Aufzählung gegen Formel und gegen Minimierung des Primals über b an den Knickstellen (anderer Rechenweg)."""
    for C in sorted({F(k, 120) for k in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 31, 35, 40, 45, 50, 60)} | {F(3, 10), F(7, 20)}):
        z = Ex.knick_zeile(C)
        plo, phi_, pmin = b_intervall_primal(z["alpha"], HX, HY, C)
        assert pmin == z["dual"] and (plo, phi_) == (z["b_lo"], z["b_hi"]), C
        _, _, blo, bhi, _ = _formel(C)
        assert (z["b_lo"], z["b_hi"]) == (blo, bhi)
    z = Ex.knick_zeile(F(3, 10))
    assert (z["b_lo"], z["b_hi"]) == (F(-8, 5), F(-7, 5)) and z["dual"] == F(51, 50)
    assert primal_wert(z["alpha"], F(-1), HX, HY, F(3, 10))[0] > z["dual"]  # b = −1 (Obergrenze der ersten Fassung) ist bei C = 3/10 nicht optimal


def test_monotonie_in_c_im_handbeispiel_exakt():
    rows = Ex.knick_tabelle([F(1, 8), F(1, 4), F(1, 3), F(1, 2), 1, 2, 4, 100])
    assert [r["w"] for r in rows] == sorted(r["w"] for r in rows)  # ‖w‖ nicht fallend in C
    assert [r["dual"] for r in rows] == sorted(r["dual"] for r in rows)  # Dualwert nicht fallend in C
    xi = [r["xi_sum"] for r in rows]
    assert xi == sorted(xi, reverse=True)  # Schlupfsumme nicht steigend in C


def test_schlupfsumme_von_hand():
    """Von Hand: ab C ≥ 1/3 ist f = (−5/3, −1, −1/3, 1/3, 1), y·f = (5/3, 1, −1/3, −1/3, 1), ξ = (0, 0, 4/3, 4/3, 0), Σ ξ = 8/3; darunter Σ ξ = 4 − 4C."""
    assert Ex.knick_zeile(F(1))["xi_sum"] == F(8, 3)
    for C in (F(1, 8), F(1, 4), F(3, 10), F(1, 3)):
        assert Ex.knick_zeile(C)["xi_sum"] == 4 - 4 * C
    for C in (F(1, 2), F(2), F(100)):
        assert Ex.knick_zeile(C)["xi_sum"] == F(8, 3)


@pytest.mark.parametrize("C", C7)
def test_smo_exakt_erreicht_das_alpha_der_aufzaehlung_und_zaehlt_die_schritte(C):
    alpha, schritte = smo_exakt(HX, HY, C)
    assert tuple(alpha) == Ex.knick_zeile(C)["alpha"]
    assert len(schritte) == {F(1, 8): 3, F(1, 4): 6, F(1, 2): 7, F(1): 11, F(2): 21, F(4): 39, F(100): 935}[C]
    t1 = min(F(1, 2), C)  # erster Schritt von Hand: Paar (x = 2, x = 0), a = 4, t* = 2/4, geklemmt auf C
    assert schritte[0][:2] == (2, 0) and schritte[0][2] == (t1, 0, t1, 0, 0)
    assert smo_exakt_paare(HX, HY, C)[1] == [(i, j) for i, j, _ in schritte]  # zweite Implementierung derselben Regel gibt dieselbe Schrittfolge


@pytest.mark.parametrize("C", C7)
def test_slsqp_primal_mit_schlupf_gleich_exaktem_dualwert(C):
    optimize = pytest.importorskip("scipy.optimize")
    xs, ys, Cf = np.array([float(x) for x in HX]), np.array(HY, float), float(C)
    cons = [{"type": "ineq", "fun": lambda z: ys * (z[0] * xs + z[1]) - 1 + z[2:]}]
    best = None
    for start in ([0.0, 0.0] + [1.0] * 5, [1.0, -2.0] + [0.5] * 5, [-1.0, 1.0] + [2.0] * 5):
        r = optimize.minimize(lambda z: 0.5 * z[0] ** 2 + Cf * z[2:].sum(), np.array(start), constraints=cons, bounds=[(None, None)] * 2 + [(0, None)] * 5, method="SLSQP",
                              options={"ftol": 1e-14, "maxiter": 500})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    z = Ex.knick_zeile(C)
    assert best is not None and abs(best.fun - float(z["dual"])) < 1e-8 * max(1.0, float(z["dual"])) and abs(best.x[0] - float(z["w"])) < 1e-6


@pytest.mark.parametrize("C", C7)
def test_eigenes_smo_zweite_ordnung_gleich_aufzaehlung(hand, C):
    m = ev.fit(hand, "linear", 1.0, float(C), 1e-10, 200000)
    z = Ex.knick_zeile(C)
    assert m.converged and np.allclose(m.alpha, [float(a) for a in z["alpha"]], atol=1e-7) and m.dual == pytest.approx(float(z["dual"]), abs=1e-8)
    lo, hi = float(z["b_lo"]), float(z["b_hi"])
    assert lo - 1e-7 <= m.b <= hi + 1e-7  # b des Lösers liegt im optimalen Intervall
    assert ev.b_interval(m.K, m.y, m.alpha, float(C))[0] == pytest.approx(lo, abs=1e-6) and ev.b_interval(m.K, m.y, m.alpha, float(C))[1] == pytest.approx(hi, abs=1e-6)


@pytest.mark.parametrize("C", C7)
def test_libsvm_alpha_und_b_intervall(C):
    SVC = pytest.importorskip("sklearn.svm").SVC
    Xm, ys = np.array([float(x) for x in HX]).reshape(-1, 1), np.array(HY, float)
    m = SVC(kernel="linear", C=float(C), tol=1e-10).fit(Xm, ys)
    a = np.zeros(5)
    a[m.support_] = np.abs(m.dual_coef_[0])
    z = Ex.knick_zeile(C)
    assert np.abs(a - np.array([float(v) for v in z["alpha"]])).max() < 1e-7
    assert float(z["b_lo"]) - 1e-7 <= float(m.intercept_[0]) <= float(z["b_hi"]) + 1e-7


def test_leave_one_out_handbeispiel_vier_gegen_drei_und_stuetzvektoren():
    for C in C7:
        r = Ex.loo_exakt(HX, HY, C)
        assert r["n_sv"] == 4 and r["worst"] == (4 if C == F(1, 8) else 3) and r["mitte"] == 3 and r["einklassig"] == 0
        assert r["worst"] <= r["n_sv"]


def test_leave_one_out_gegenbeispiel_drei_punkte_nur_mit_der_zaehlung_des_einklassigen_rests():
    g = Ex.GEGENBEISPIELE[0]
    mit = Ex.loo_exakt(g["x"], g["y"], g["C"], True)
    ohne = Ex.loo_exakt(g["x"], g["y"], g["C"], False)
    assert (mit["n_sv"], mit["worst"], mit["mitte"], mit["einklassig"]) == (2, 3, 2, 1)  # 3 > 2 nur mit der Zählung
    assert (ohne["worst"], ohne["mitte"]) == (2, 1) and ohne["worst"] <= ohne["n_sv"]  # ohne sie: keine Verletzung
    al, lo, hi = Ex.kkt_aufzaehlung(g["x"], g["y"], g["C"])[0]
    assert al == (g["C"], g["C"], 0) and (lo, hi) == (F(2, 5), F(4, 5))
    al2, lo2, hi2 = Ex.kkt_aufzaehlung(g["x"][:2], g["y"][:2], g["C"])[0]
    assert (lo2, hi2) == (F(-4, 5), F(4, 5)) and 1 * (2 * g["C"] * 3 + lo2) == F(-1, 5)  # bei b = −4/5 ist y f(3) = −1/5


def test_leave_one_out_gegenbeispiel_fuenf_punkte_ohne_konvention():
    g = Ex.GEGENBEISPIELE[1]
    r = Ex.loo_exakt(g["x"], g["y"], g["C"])
    assert (r["n_sv"], r["worst"], r["mitte"], r["einklassig"]) == (4, 5, 3, 0)  # 5 > 4, kein einklassiger Rest
    assert Ex.loo_exakt(g["x"], g["y"], g["C"], False) == r  # die Konvention ändert nichts
    al, lo, hi = Ex.kkt_aufzaehlung(g["x"], g["y"], g["C"])[0]
    assert al == (0, g["C"], g["C"], g["C"], g["C"]) and (lo, hi) == (F(-3, 5), F(-2, 5))


def test_leave_one_out_gegenbeispiel_fuenf_punkte_libsvm_neufit_haelt_die_schranke():
    SVC = pytest.importorskip("sklearn.svm").SVC
    g = Ex.GEGENBEISPIELE[1]
    X = np.array([float(x) for x in g["x"]]).reshape(-1, 1)
    y = np.array(g["y"], float)
    loo = 0
    for i in range(5):
        keep = [k for k in range(5) if k != i]
        mi = SVC(kernel="linear", C=float(g["C"]), tol=1e-10).fit(X[keep], y[keep])
        loo += int(y[i] * float(mi.decision_function(X[i:i + 1])[0]) <= 0)
    assert loo == 3 <= 4  # für das b, das libsvm im Neufit wählt (Intervallmitte), hält die Schranke


def test_app_tabelle_hand_table_stimmt_mit_der_aufzaehlung(hand):
    rows = ev.hand_table()
    assert [r["C"] for r in rows] == [F(a, b) for a, b in ((1, 8), (1, 4), (3, 10), (1, 3), (1, 2), (1, 1), (2, 1), (4, 1), (100, 1))]
    for r in rows:
        assert r["smo_converged"] and r["smo_dual"] == pytest.approx(float(r["dual"]), abs=1e-8)
    assert [r["smo_steps"] > 0 for r in rows] == [True] * 9
    assert Sc.fuenf_punkte().n == 5
