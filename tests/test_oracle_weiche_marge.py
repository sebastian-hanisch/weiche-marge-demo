"""Orakel-Tests (ORACLE-PLAYBOOK §2, DEMO-PLAYBOOK §12): das eigene SMO, das Dual, die KKT-Lücke, Leave-one-out und die Barriere gegen einen anderen Rechenweg.

Rechenwege: libsvm (`sklearn.svm.SVC`, nur als Fremdreferenz im Test), das Primal mit Schlupfvariablen per SLSQP, die KKT-Bedingungen nach ihrer Definition, die harte Marge (C unendlich),
das Herausnehmen durch Neuberechnung mit libsvm und die Abkürzung „nur Stützvektoren neu rechnen“, die Aufzählung in Brüchen (siehe `test_exact.py`). Instanzen: wenige (20 bis 50), kleine n."""

import numpy as np
import pytest

import wmr_barrier as Bar
import wmr_evaluation as ev
import wmr_scenario as Sc
from wmr_smo import dual_value, kkt_gap, smo

KERNELS = (("linear", 1.0), ("gauss", 1.0), ("gauss", 0.5))


def _inst(kind_i, n, seed, flip):
    data = "Gauß-Klassen" if kind_i != 2 else "Ringe mit Streuung"
    return Sc.make_instance(data, n, seed, flip)


def _svc(inst, kind, sigma, c, tol=1e-10):
    SVC = pytest.importorskip("sklearn.svm").SVC
    if kind == "linear":
        m = SVC(kernel="linear", C=c, tol=tol)
    else:
        m = SVC(kernel="rbf", gamma=ev.gamma_of(sigma), C=c, tol=tol)
    m.fit(inst.X, inst.y)
    a = np.zeros(inst.n)
    a[m.support_] = np.abs(m.dual_coef_[0])
    return m, a


@pytest.mark.parametrize("kind,sigma", KERNELS)
@pytest.mark.parametrize("c", (0.1, 1.0, 10.0))
def test_dualwert_und_w_gleich_libsvm_und_schrittzahl_nur_als_faktor(kind, sigma, c):
    steps_ratio = []
    for seed in range(4):
        inst = _inst(0 if kind == "linear" else seed % 3, 30, 100 + seed, 0.1)
        m = ev.fit(inst, kind, sigma, c, 1e-9, 500000)
        assert m.converged
        _, a = _svc(inst, kind, sigma, c)
        K = m.K
        d_lib = dual_value(K, inst.y, a)
        assert abs(m.dual - d_lib) <= 1e-8 * max(1.0, abs(d_lib))
        w2 = (m.alpha * inst.y) @ K @ (m.alpha * inst.y)
        w2_lib = (a * inst.y) @ K @ (a * inst.y)
        assert w2 == pytest.approx(w2_lib, rel=1e-5, abs=1e-8)  # ‖w‖² ist eindeutig (α und b nicht immer)
        # Schrittzahl: nur als Faktor 4 gegen n_iter_ von libsvm bei gleicher Toleranz ohne Schrumpfen (Studie, Gauß-Kern: 0,82 bis 1,32; linear 0,33 bis 1,85 in diesen Instanzen)
        SVC = pytest.importorskip("sklearn.svm").SVC
        mk = SVC(kernel="linear", C=c, tol=1e-6, shrinking=False) if kind == "linear" else SVC(kernel="rbf", gamma=ev.gamma_of(sigma), C=c, tol=1e-6, shrinking=False)
        mk.fit(inst.X, inst.y)
        mine = ev.fit(inst, kind, sigma, c, 1e-6, 500000)
        steps_ratio.append(mine.steps / max(int(np.ravel(mk.n_iter_)[0]), 1))
    assert all(1 / 4 <= r <= 4 for r in steps_ratio), steps_ratio


def test_alpha_gleich_libsvm_bei_streng_konvexem_gauss_kern():
    """Beim Gauß-Kern ist α eindeutig (die Gram-Matrix ist positiv definit); beim linearen Kern in 2D nur ‖w‖² (siehe vorheriger Test)."""
    kind, sigma = "gauss", 1.0
    for seed in range(3):
        inst = _inst(0, 25, 200 + seed, 0.1)
        m = ev.fit(inst, kind, sigma, 1.0, 1e-10, 500000)
        _, a = _svc(inst, kind, sigma, 1.0)
        assert np.abs(m.alpha - a).max() < 1e-4


def test_primal_mit_schlupfvariablen_slsqp_gleich_dualwert():
    optimize = pytest.importorskip("scipy.optimize")
    inst = _inst(0, 12, 301, 0.1)
    X, y = inst.X, inst.y
    for c in (0.1, 1.0):
        m = ev.fit(inst, "linear", 1.0, c, 1e-10, 100000)
        n = len(y)

        def ziel(z):
            return 0.5 * (z[0] ** 2 + z[1] ** 2) + c * z[3:].sum()

        cons = [{"type": "ineq", "fun": lambda z: y * (X @ z[:2] + z[2]) - 1 + z[3:]}]
        best = None
        for start in (np.r_[0.0, 0.0, 0.0, np.ones(n)], np.r_[1.0, -1.0, 0.0, 2 * np.ones(n)], np.r_[-1.0, 1.0, 0.1, np.ones(n)]):
            r = optimize.minimize(ziel, start, constraints=cons, bounds=[(None, None)] * 3 + [(0, None)] * n, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
            if r.success and (best is None or r.fun < best.fun):
                best = r
        assert best is not None and abs(best.fun - m.dual) < 1e-6 * max(1.0, m.dual)


@pytest.mark.parametrize("kind,sigma", KERNELS)
def test_kkt_bedingungen_nach_definition_am_ergebnis(kind, sigma):
    for seed in range(3):
        inst = _inst(seed % 3, 40, 400 + seed, 0.1)
        for c in (0.1, 10.0):
            tol = 1e-6
            m = ev.fit(inst, kind, sigma, c, tol, 500000)
            assert m.converged and abs(np.sum(m.alpha * inst.y)) < 1e-9 and (m.alpha >= 0).all() and (m.alpha <= c + 1e-12).all()
            yf = inst.y * (m.K @ (m.alpha * inst.y) + m.b)
            t = ev._tol_alpha(c)
            assert (yf[m.alpha <= t] >= 1 - 10 * tol).all()  # α = 0: außerhalb oder auf der Marge
            free = (m.alpha > t) & (m.alpha < c - t)
            assert (np.abs(yf[free] - 1) <= 10 * tol).all()  # 0 < α < C: genau auf der Marge
            assert (yf[m.alpha >= c - t] <= 1 + 10 * tol).all()  # α = C: in oder jenseits der Marge
            assert kkt_gap(m.K, inst.y, m.alpha, c) <= tol + 1e-12  # und die Lücke selbst, unabhängig vom Verfahren neu gerechnet


def test_weich_gleich_hart_ab_c_gleich_groesstes_alpha_der_harten_marge():
    for seed in range(5):
        inst = Sc.gauss_classes(20, 500 + seed, 0.0, sep=6.0)
        hart = ev.fit(inst, "linear", 1.0, np.inf, 1e-10, 100000)
        assert hart.converged
        amax = float(hart.alpha.max())
        weich = ev.fit(inst, "linear", 1.0, 1.5 * amax, 1e-10, 100000)
        assert np.abs(weich.alpha - hart.alpha).max() < 1e-6 and weich.margin() == pytest.approx(hart.margin(), rel=1e-6)
        zu_klein = ev.fit(inst, "linear", 1.0, 0.5 * amax, 1e-10, 100000)
        assert abs(zu_klein.margin() - hart.margin()) > 1e-6  # bei C = 0,5 · max α ist die Lösung eine andere


def _loo_libsvm(inst, kind, sigma, c):
    """Leave-one-out mit libsvm-Neufit je Punkt (tol 1e-10), b von libsvm; ein Rest mit einer Klasse zählt als Fehler."""
    SVC = pytest.importorskip("sklearn.svm").SVC
    n = inst.n
    err = np.zeros(n, bool)
    for i in range(n):
        keep = np.delete(np.arange(n), i)
        if len(np.unique(inst.y[keep])) < 2:
            err[i] = True
            continue
        m = SVC(kernel="linear", C=c, tol=1e-10) if kind == "linear" else SVC(kernel="rbf", gamma=ev.gamma_of(sigma), C=c, tol=1e-10)
        m.fit(inst.X[keep], inst.y[keep])
        err[i] = inst.y[i] * float(m.decision_function(inst.X[i:i + 1])[0]) <= 0
    return err


@pytest.mark.parametrize("kind,sigma", KERNELS)
def test_leave_one_out_durch_echtes_herausnehmen_gleich_libsvm_neufit(kind, sigma):
    for seed in range(3):
        inst = _inst(0 if kind == "linear" else seed % 3, 14, 600 + seed, 0.1)
        for c in (0.1, 10.0):
            r = ev.loo(inst, kind, sigma, c, max_steps=200000)
            assert not r["nonconv"].any()
            assert np.array_equal(r["err_solver"], _loo_libsvm(inst, kind, sigma, c))


def test_leave_one_out_abkuerzung_nur_stuetzvektoren_neu_rechnen_gibt_dieselbe_zahl():
    """Ein Punkt mit α = 0 wird beim Herausnehmen richtig eingeordnet (die Lösung ohne ihn hat dieselben α): nur Stützvektoren können fehlen. Gilt für das b des Lösers, wenn die volle Lösung
    einen freien Stützvektor hat (dann ist b eindeutig); gemessen an 20 Instanzen."""
    geprueft = 0
    for seed in range(20):
        inst = _inst(seed % 3, 16, 700 + seed, 0.1)
        for c in (1.0, 10.0):
            r = ev.loo(inst, "linear" if seed % 3 == 0 else "gauss", 1.0, c, max_steps=200000)
            full = r["full"]
            t = ev._tol_alpha(c)
            if not ((full.alpha > t) & (full.alpha < c - t)).any():
                continue
            assert not (r["err_solver"] & ~r["was_sv"]).any()  # kein Punkt mit α = 0 wird falsch eingeordnet
            assert r["loo_solver"] == int((r["err_solver"] & r["was_sv"]).sum())
            geprueft += 1
    assert geprueft >= 20


def test_leave_one_out_schranke_mit_dem_b_des_loesers_in_30_instanzen():
    for seed in range(30):
        inst = _inst(seed % 3, 12 + seed % 8, 800 + seed, 0.1 * (seed % 2))
        r = ev.loo(inst, "linear" if seed % 3 == 0 else "gauss", 1.0, (0.1, 1.0, 10.0)[seed % 3], max_steps=200000)
        assert r["loo_solver"] <= r["n_sv"], (seed, r["loo_solver"], r["n_sv"])


def test_barriere_dualwert_gleich_libsvm_und_newton_schritte_als_zaehlung():
    for seed in range(3):
        inst = Sc.gauss_classes(40, 900 + seed, 0.0)
        K = ev.gram(inst.X, inst.X, "gauss", 1.0)
        b = Bar.barrier(K, inst.y, 1.0)
        assert b.converged
        _, a = _svc(inst, "gauss", 1.0, 1.0, tol=1e-9)
        assert abs(dual_value(K, inst.y, b.alpha) - dual_value(K, inst.y, a)) < 1e-6 * max(1.0, dual_value(K, inst.y, a))
        assert 20 <= b.newton <= 400 and b.outer >= 3  # Zählung, keine Gleichheit: hängt an den Einstellungen
        s = smo(K, inst.y, 1.0, eps=1e-9)
        assert abs(dual_value(K, inst.y, b.alpha) - s.dual) < 1e-5
