"""Messwerkzeug: rechnet die eingefrorenen Studien der Demo (`wmr_frozen.json`) mit der Zufallsquelle der App (SplitMix64) neu.

Nur im Werkzeug und in den Tests, nie in der App: nutzt scikit-learn (libsvm, `SVC`) und scipy als Fremdreferenz und für die Messungen mit libsvm. Das Live-Modell der App
(eigenes SMO) ist nicht dasselbe Programm wie libsvm: Schrittzahlen und Meldungen der Studie stammen von libsvm (siehe Abschnitt `live_vs_studie`).

Aufruf (aus dem Demo-Ordner, mit dem Test-venv, `OMP_NUM_THREADS=1`):
  python tools/wmr_make_frozen.py                      alle Abschnitte rechnen und wmr_frozen.json schreiben
  python tools/wmr_make_frozen.py --section c_wirkung  nur diesen Abschnitt rechnen, in den vorhandenen Stand einsetzen
  python tools/wmr_make_frozen.py --parts DIR --section X   Teilergebnis nach DIR/X.json (zum parallelen Rechnen), danach --merge DIR
  python tools/wmr_make_frozen.py --check              Stichproben neu rechnen und mit der Datei vergleichen (schnell, ohne zu schreiben)

Abschnitte: bayes, c_wirkung, warn_harte, wahl60, wahl200, loo, schritte_c, schritte_n, barriere, live_vs_studie.
Zufalls-Seeds: Gruppe g (1, 2, 3) nimmt die Zufalls-Seeds 100000·g + s; Testmengen und Validierungsmengen haben getrennte Zufalls-Seeds (siehe wmr_scenario.test_set).
Zähleinheiten stehen in der Datei je Abschnitt (Zufalls-Seeds, Zellen, Instanzen, Aktualisierungsschritte, Newton-Schritte)."""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import wmr_barrier as Bar  # noqa: E402
import wmr_evaluation as ev  # noqa: E402
import wmr_scenario as Sc  # noqa: E402
from wmr_rng import SplitMix64  # noqa: E402
from wmr_smo import kkt_gap, smo  # noqa: E402

OUT = os.path.join(ROOT, "wmr_frozen.json")
CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GRID_C = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GRID_S = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
N_TRAIN, N_TEST = 60, 2000

CASES = {  # Schlüssel -> (Datensatz, Kern, σ, Anteil vertauscht)
    "gauss_lin": ("Gauß-Klassen", "linear", None, 0.0),
    "gauss_lin_flip": ("Gauß-Klassen", "linear", None, 0.1),
    "gauss_rbf": ("Gauß-Klassen", "gauss", 1.0, 0.0),
    "gauss_rbf_flip": ("Gauß-Klassen", "gauss", 1.0, 0.1),
    "ringe_rbf": ("Ringe mit Streuung", "gauss", 0.5, 0.0),
    "ringe_rbf_flip": ("Ringe mit Streuung", "gauss", 0.5, 0.1),
}


def seed_of(group: int, s: int) -> int:
    return 100000 * group + s


def make(data, n, seed, flip):
    return Sc.make_instance(data, n, seed, flip)


def svc(X, y, kind, sigma, c, tol=1e-3, shrinking=True, max_iter=-1):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import SVC

    if kind == "linear":
        m = SVC(kernel="linear", C=c, tol=tol, shrinking=shrinking, max_iter=max_iter, cache_size=200)
    else:
        m = SVC(kernel="rbf", gamma=ev.gamma_of(sigma), C=c, tol=tol, shrinking=shrinking, max_iter=max_iter, cache_size=200)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(X, y)
        warned = any(issubclass(x.category, ConvergenceWarning) for x in w)
    return m, warned


def alpha_of(m, n):
    a = np.zeros(n)
    a[m.support_] = np.abs(m.dual_coef_[0])
    return a


def err_of(m, X, y):
    f = m.decision_function(X)
    return float((np.where(f >= 0, 1.0, -1.0) != y).mean())


def n_iter(m) -> int:
    return int(np.ravel(m.n_iter_)[0])


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- bayes
def sec_bayes():
    from scipy import integrate
    from scipy.stats import norm

    out = {"gauss": Sc.bayes_gauss(), "ringe": {}, "stichprobe_punkte": 200000}
    for s in (0.25, 0.35, 0.45):
        def f(r, mu):
            return norm.pdf(r, mu, s) / (1 - norm.cdf(0, mu, s))
        val, _ = integrate.quad(lambda r: min(f(r, 1.0), f(r, 2.0)), 0, 6, points=[1.5], limit=200)
        inst = Sc.ringe_streuung(200000, 900000 + int(100 * s), s=s, fast=True)
        r = np.linalg.norm(inst.X, axis=1)
        e = float((np.where(r < 1.5, -1.0, 1.0) != inst.y).mean())
        out["ringe"][str(s)] = {"integral": 0.5 * val, "trapez": Sc.bayes_ringe(s), "stichprobe": e}
    g = Sc.gauss_classes(200000, 900007, 0.0, fast=True)
    out["gauss_stichprobe"] = float((np.where(g.X[:, 0] >= 0, 1.0, -1.0) != g.y).mean())
    return out


# ---------------------------------------------------------------- c_wirkung
def sec_c_wirkung(n_seeds=30):
    res = {"zufalls_seeds_je_gruppe": n_seeds, "n_train": N_TRAIN, "n_test": N_TEST, "C": CS, "faelle": {}}
    for key, (data, kind, sigma, flip) in CASES.items():
        case = {"data": data, "kern": kind, "sigma": sigma, "flip": flip, "gruppen": {}}
        for g in (1, 2):
            arr = {k: [[] for _ in CS] for k in ("test", "train", "n_iter", "gap", "sv", "warn100k", "warn_unbegrenzt")}
            for s in range(n_seeds):
                sd = seed_of(g, s)
                tr = make(data, N_TRAIN, sd, flip)
                te = Sc.test_set(data, N_TEST, sd)
                K = ev.gram(tr.X, tr.X, kind, sigma or 1.0)
                for ci, c in enumerate(CS):
                    m, w_unb = svc(tr.X, tr.y, kind, sigma, c)
                    a = alpha_of(m, N_TRAIN)
                    _, w_cap = svc(tr.X, tr.y, kind, sigma, c, max_iter=100000)
                    arr["test"][ci].append(err_of(m, te.X, te.y))
                    arr["train"][ci].append(err_of(m, tr.X, tr.y))
                    arr["n_iter"][ci].append(n_iter(m))
                    arr["gap"][ci].append(kkt_gap(K, tr.y, a, c))
                    arr["sv"][ci].append(float((a > 1e-8 * max(1.0, c)).mean()))
                    arr["warn100k"][ci].append(int(w_cap))
                    arr["warn_unbegrenzt"][ci].append(int(w_unb))
            case["gruppen"][str(g)] = arr
            log(f"c_wirkung {key} Gruppe {g} fertig")
        res["faelle"][key] = case
    return res


# ---------------------------------------------------------------- warn_harte (ein Satz mit Verweis auf `harte-marge-demo`; hier nur als Gegenprobe der Warnzählung)
def sec_warn_harte():
    cs = [1e2, 1e4, 1e6, 1e8]
    sets = 6
    warn = [[0] * len(cs) for _ in range(sets)]
    test = [[0.0] * len(cs) for _ in range(sets)]
    for k in range(sets):
        for s in range(30):
            sd = 400000 + 100 * k + s
            tr = make("Gauß-Klassen", N_TRAIN, sd, 0.1)
            te = Sc.test_set("Gauß-Klassen", N_TEST, sd)
            for ci, c in enumerate(cs):
                m, w = svc(tr.X, tr.y, "gauss", 1.0, c, max_iter=100000)
                warn[k][ci] += int(w)
                test[k][ci] += err_of(m, te.X, te.y) / 30.0
    return {"C": cs, "saetze": sets, "je_satz": 30, "n_train": N_TRAIN, "grenze": 100000, "warn": warn, "test_mittel": test}


# ---------------------------------------------------------------- wahl (Ringe mit Streuung, Raster 6 C × 6 σ)
def _grid(tr, te, fits_ref=None):
    te_err = np.zeros((len(GRID_C), len(GRID_S)))
    for ci, c in enumerate(GRID_C):
        for si, sg in enumerate(GRID_S):
            m, _ = svc(tr.X, tr.y, "gauss", sg, c)
            te_err[ci, si] = err_of(m, te.X, te.y)
    return te_err


def _cv_grid(tr, nfold):
    n = tr.n
    folds = np.arange(n) % nfold
    cv = np.zeros((len(GRID_C), len(GRID_S)))
    for ci, c in enumerate(GRID_C):
        for si, sg in enumerate(GRID_S):
            e = 0.0
            for f in range(nfold):
                m_tr = folds != f
                if len(np.unique(tr.y[m_tr])) < 2:
                    e += float((~m_tr).sum())
                    continue
                m, _ = svc(tr.X[m_tr], tr.y[m_tr], "gauss", sg, c)
                e += err_of(m, tr.X[~m_tr], tr.y[~m_tr]) * float((~m_tr).sum())
            cv[ci, si] = e / n
    return cv


def _default_fit(tr, te):
    from sklearn.svm import SVC

    m = SVC(kernel="rbf", gamma="scale", C=1.0).fit(tr.X, tr.y)
    sg0 = float(np.sqrt(tr.X.shape[1] * tr.X.var() / 2.0))  # gamma = 1/(p Var(X)) -> σ = sqrt(1/(2 γ))
    return err_of(m, te.X, te.y), sg0


def sec_wahl(n_train, groups, n_seeds, with_val, key_off):
    res = {"n_train": n_train, "zufalls_seeds_je_gruppe": n_seeds, "C": GRID_C, "sigma": GRID_S, "gruppen": {}}
    for g in groups:
        gd = {}
        for flip_key, flip in (("flip0", 0.0), ("flip10", 0.1)):
            te_all, cv_all, va_all, dflt, sg0 = [], [], {"30": [], "100": [], "500": []}, [], []
            for s in range(n_seeds):
                sd = seed_of(g, s) + key_off
                tr = make("Ringe mit Streuung", n_train, sd, flip)
                te = Sc.test_set("Ringe mit Streuung", N_TEST, sd)
                te_err = _grid(tr, te)
                te_all.append(te_err.ravel().tolist())
                cv_all.append(_cv_grid(tr, 5).ravel().tolist())
                d, sg = _default_fit(tr, te)
                dflt.append(d)
                sg0.append(sg)
                if with_val and flip == 0.0:
                    va = Sc.make_instance("Ringe mit Streuung", 500, sd + 13_000_019, 0.0, fast=True)
                    pred = np.zeros((len(GRID_C), len(GRID_S), 500))
                    for ci, c in enumerate(GRID_C):
                        for si, sg_ in enumerate(GRID_S):
                            m, _ = svc(tr.X, tr.y, "gauss", sg_, c)
                            pred[ci, si] = np.where(m.decision_function(va.X) >= 0, 1.0, -1.0)
                    for mm in (30, 100, 500):
                        va_all[str(mm)].append((pred[:, :, :mm] != va.y[:mm]).mean(axis=2).ravel().tolist())
            gd[flip_key] = {"te": te_all, "cv": cv_all, "default": dflt, "sigma0": sg0}
            if with_val and flip == 0.0:
                gd[flip_key]["va"] = va_all
            log(f"wahl{n_train} Gruppe {g} {flip_key} fertig")
        res["gruppen"][str(g)] = gd
    return res


def sec_wahl60():
    return sec_wahl(60, (1, 2), 30, True, 0)


def sec_wahl200():
    return sec_wahl(200, (1, 2, 3), 20, False, 50_000)


# ---------------------------------------------------------------- loo
LOO_KINDS = (("Gauß-Klassen", "linear", None, 0.0, 2.0), ("Gauß-Klassen", "gauss", 1.0, 0.1, 2.0), ("Ringe mit Streuung", "gauss", 0.5, 0.0, None), ("Gauß-Klassen", "linear", None, 0.0, 6.0))


def _loo_svc(tr, kind, sigma, c, tol, strict_too=False):
    """LOO mit libsvm-Neufit: mit dem b von libsvm und mit dem ungünstigsten b des optimalen Intervalls; Zahl der Stützvektoren der vollen Lösung."""
    n = tr.n
    X, y = tr.X, tr.y
    K = ev.gram(X, X, kind, sigma or 1.0)
    m, _ = svc(X, y, kind, sigma, c, tol=tol)
    a = alpha_of(m, n)
    t = 1e-8 * max(1.0, c)
    nsv = int((a > t).sum())
    free = int(((a > t) & (a < c - t)).sum())
    loo_s = loo_w = loo_w_strict = ties = one_class = 0
    for i in range(n):
        keep = np.delete(np.arange(n), i)
        if len(np.unique(y[keep])) < 2:
            loo_s += 1
            loo_w += 1
            loo_w_strict += 1
            one_class += 1
            continue
        mi, _ = svc(X[keep], y[keep], kind, sigma, c, tol=tol)
        f = float(mi.decision_function(X[i:i + 1])[0])
        if abs(f) < 1e-7:
            ties += 1
        loo_s += int(y[i] * f <= 0)
        ai = alpha_of(mi, n - 1)
        Kk = K[np.ix_(keep, keep)]
        lo, hi = ev.b_interval(Kk, y[keep], ai, c, tol=1e-7 * max(1.0, c))
        f0i = float(K[i, keep] @ (ai * y[keep]))
        ends = [b for b in (lo, hi) if np.isfinite(b)] or [float(mi.intercept_[0])]
        worst = min(y[i] * (f0i + b) for b in ends)
        loo_w += int(worst <= 1e-9)
        loo_w_strict += int(worst < -1e-9)
    return {"n": n, "nsv": nsv, "free": free, "loo_solver": loo_s, "loo_worst": loo_w, "loo_worst_strict": loo_w_strict, "ties": ties, "one_class": one_class}


def sec_loo():
    rng = SplitMix64(2024)
    e1 = []
    cs = [0.01, 0.1, 1.0, 10.0, 1000.0]
    for t in range(300):
        n = 12 + rng.below(19)
        kind_i = rng.below(4)
        c = cs[rng.below(len(cs))]
        data, kind, sigma, flip, sep = LOO_KINDS[kind_i]
        tr = Sc.gauss_classes(n, 200000 + t, flip, sep=sep) if data == "Gauß-Klassen" else Sc.ringe_streuung(n, 200000 + t, flip=flip)
        if len(np.unique(tr.y)) < 2:
            continue
        r = _loo_svc(tr, kind, sigma, c, 1e-9)
        r.update({"kind": kind_i, "c": c})
        e1.append(r)
    log("loo E1 fertig")
    rng = SplitMix64(2025)
    e5 = []
    for t in range(600):
        n = 6 + rng.below(5)
        c = float(np.exp(np.log(0.02) + (np.log(0.5) - np.log(0.02)) * rng.random()))
        kind_i = rng.below(3)
        data, kind, sigma, flip, sep = LOO_KINDS[kind_i]
        tr = Sc.gauss_classes(n, 300000 + t, flip) if data == "Gauß-Klassen" else Sc.ringe_streuung(n, 300000 + t, flip=flip)
        if min((tr.y > 0).sum(), (tr.y < 0).sum()) < 2:
            continue
        r = _loo_svc(tr, kind, sigma, c, 1e-12)
        r.update({"kind": kind_i, "c": c})
        e5.append(r)
    log("loo E5 fertig")
    return {"e1": e1, "e5": e5, "kinds": [list(k) for k in LOO_KINDS]}


# ---------------------------------------------------------------- schritte
def sec_schritte_c():
    res = {"n": 200, "zufalls_seeds": 5, "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "eps": 1e-6, "faelle": {}}
    for key, data, sigma in (("gauss", "Gauß-Klassen", 1.0), ("ringe", "Ringe mit Streuung", 0.5)):
        own = [[] for _ in res["C"]]
        lib = [[] for _ in res["C"]]
        sv = [[] for _ in res["C"]]
        for s in range(5):
            tr = make(data, 200, 500000 + s, 0.05)
            K = ev.gram(tr.X, tr.X, "gauss", sigma)
            for ci, c in enumerate(res["C"]):
                r = smo(K, tr.y, c, eps=1e-6, max_steps=3_000_000)
                assert r.converged
                own[ci].append(int(r.steps))
                sv[ci].append(float((r.alpha > 1e-8 * max(1.0, c)).mean()))
                m, _ = svc(tr.X, tr.y, "gauss", sigma, c, tol=1e-6, shrinking=False)
                lib[ci].append(n_iter(m))
            log(f"schritte_c {key} Zufalls-Seed {s}")
        res["faelle"][key] = {"sigma": sigma, "own": own, "lib": lib, "sv": sv}
    return res


def sec_schritte_n():
    ns = [40, 80, 160, 320, 640, 1280]
    res = {"n": ns, "zufalls_seeds": 20, "C": 1.0, "flip": 0.05, "sigma": 1.0, "eps": 1e-6, "own": [[] for _ in ns], "lib": [[] for _ in ns]}
    for ni, n in enumerate(ns):
        for s in range(20):
            tr = make("Gauß-Klassen", n, 600000 + s, 0.05)
            K = ev.gram(tr.X, tr.X, "gauss", 1.0)
            r = smo(K, tr.y, 1.0, eps=1e-6, max_steps=3_000_000)
            assert r.converged
            res["own"][ni].append(int(r.steps))
            m, _ = svc(tr.X, tr.y, "gauss", 1.0, 1.0, tol=1e-6, shrinking=False)
            res["lib"][ni].append(n_iter(m))
        log(f"schritte_n n = {n} fertig")
    return res


def _trust(Q, y, c):
    from scipy import optimize

    n = len(y)
    lc = optimize.LinearConstraint(y[None, :], 0.0, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = optimize.minimize(lambda a: 0.5 * a @ Q @ a - a.sum(), np.full(n, c / 2.0), jac=lambda a: Q @ a - 1.0, hess=lambda a: Q, bounds=optimize.Bounds(np.zeros(n), np.full(n, c)),
                                constraints=[lc], method="trust-constr", options={"gtol": 1e-8, "xtol": 1e-12, "maxiter": 2000})
    return res


def sec_barriere():
    ns = [20, 40, 80, 160, 320]
    res = {"n": ns, "zufalls_seeds": 3, "C": 1.0, "sigma": 1.0, "einstellungen": Bar.BarrierSettings().__dict__, "newton": [], "smo": [], "trust_nit": [], "trust_dual_rel": []}
    for n in ns:
        nt, sm, tc, td = [], [], [], []
        for s in range(3):
            tr = make("Gauß-Klassen", n, 700000 + s, 0.0)
            K = ev.gram(tr.X, tr.X, "gauss", 1.0)
            Q = (tr.y[:, None] * tr.y[None, :]) * K
            nt.append(Bar.barrier(K, tr.y, 1.0).newton)
            sm.append(int(smo(K, tr.y, 1.0, eps=1e-6, max_steps=3_000_000).steps))
            if n <= 160:
                m, _ = svc(tr.X, tr.y, "gauss", 1.0, 1.0, tol=1e-9, shrinking=False)
                dref = float(alpha_of(m, n).sum() - 0.5 * alpha_of(m, n) @ Q @ alpha_of(m, n))
                r = _trust(Q, tr.y, 1.0)
                tc.append(int(r.nit))
                td.append(abs(float(r.x.sum() - 0.5 * r.x @ Q @ r.x) - dref) / max(1.0, abs(dref)))
        res["newton"].append(nt)
        res["smo"].append(sm)
        res["trust_nit"].append(tc)
        res["trust_dual_rel"].append(td)
        log(f"barriere n = {n} fertig")
    # Einstellungsabhängigkeit bei n = 80
    abh = {}
    for mu in (2.0, 10.0, 50.0):
        for gt in (1e-4, 1e-6, 1e-8):
            v = []
            for s in range(3):
                tr = make("Gauß-Klassen", 80, 700000 + s, 0.0)
                K = ev.gram(tr.X, tr.X, "gauss", 1.0)
                v.append(Bar.barrier(K, tr.y, 1.0, Bar.BarrierSettings(mu=mu, gap_tol=gt)).newton)
            abh[f"{mu:g}|{gt:g}"] = v
    res["abhaengigkeit_n80"] = abh
    return res


def sec_live_vs_studie():
    ratios, dual_rel, same, rows = [], [], 0, 0
    for s in range(20):
        tr = make("Gauß-Klassen", 100, 800000 + s, 0.05)
        K = ev.gram(tr.X, tr.X, "gauss", 1.0)
        Q = (tr.y[:, None] * tr.y[None, :]) * K
        for c in (0.1, 1.0, 10.0):
            r = smo(K, tr.y, c, eps=1e-6, max_steps=3_000_000)
            m, _ = svc(tr.X, tr.y, "gauss", 1.0, c, tol=1e-6, shrinking=False)
            a = alpha_of(m, tr.n)
            d_lib = float(a.sum() - 0.5 * a @ Q @ a)
            dual_rel.append(abs(r.dual - d_lib) / max(1.0, abs(d_lib)))
            ni = n_iter(m)
            ratios.append(r.steps / max(ni, 1))
            same += int(r.steps == ni)
            rows += 1
    return {"instanzen": rows, "n": 100, "dual_rel_max": float(max(dual_rel)), "verhaeltnis": ratios, "gleich": same}


SECTIONS = {
    "bayes": sec_bayes, "c_wirkung": sec_c_wirkung, "warn_harte": sec_warn_harte, "wahl60": sec_wahl60, "wahl200": sec_wahl200, "loo": sec_loo,
    "schritte_c": sec_schritte_c, "schritte_n": sec_schritte_n, "barriere": sec_barriere, "live_vs_studie": sec_live_vs_studie,
}


def meta():
    import scipy
    import sklearn

    return {
        "werkzeug": "tools/wmr_make_frozen.py", "zufallsquelle": "SplitMix64 (wmr_rng.py)",
        "umgebung": f"Python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}",
        "hinweis": "Schrittzahlen und Meldungen der Studie stammen von libsvm (SVC) bzw. vom eigenen SMO (Abschnitte schritte_c, schritte_n); Zahlen sind plattformabhängig (Windows gemessen)",
    }


def load() -> dict:
    try:
        with open(OUT, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save(d: dict, path=OUT):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(d, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")


def check() -> int:
    """Stichproben neu rechnen und mit der Datei vergleichen; Rückgabe 0 bei Übereinstimmung."""
    d = load()
    bad = []
    if not d:
        print("wmr_frozen.json fehlt")
        return 1
    # c_wirkung: eine Zelle
    f = d["c_wirkung"]["faelle"]["gauss_rbf"]["gruppen"]["1"]
    tr = make("Gauß-Klassen", N_TRAIN, seed_of(1, 0), 0.0)
    te = Sc.test_set("Gauß-Klassen", N_TEST, seed_of(1, 0))
    m, _ = svc(tr.X, tr.y, "gauss", 1.0, 1.0)
    if abs(err_of(m, te.X, te.y) - f["test"][3][0]) > 5e-3:  # Plattformen und libsvm-Versionen können einzelne Punkte an der Grenze verschieben (10 von 2 000 Testpunkten)
        bad.append("c_wirkung gauss_rbf Gruppe 1 Zufalls-Seed 0 C = 1: Testfehler weicht ab")
    if abs(n_iter(m) - f["n_iter"][3][0]) > 0.2 * f["n_iter"][3][0] + 3:
        bad.append("c_wirkung: n_iter weicht über 20 % ab")
    # wahl60: eine Zelle
    g = d["wahl60"]["gruppen"]["1"]["flip0"]
    tr = make("Ringe mit Streuung", 60, seed_of(1, 0), 0.0)
    te = Sc.test_set("Ringe mit Streuung", N_TEST, seed_of(1, 0))
    m, _ = svc(tr.X, tr.y, "gauss", 1.6, 1.0)
    if abs(err_of(m, te.X, te.y) - g["te"][0][2 * 6 + 4]) > 5e-3:
        bad.append("wahl60: Testfehler der Zelle C = 1, σ = 1,6 weicht ab")
    # loo: erste Instanz von E1
    e = d["loo"]["e1"][0]
    if not (e["loo_solver"] >= 0 and e["nsv"] >= 1):
        bad.append("loo: erste Instanz unplausibel")
    # schritte_n: kleinstes n, erster Zufalls-Seed, eigenes SMO (Ganzzahl; Bandbreite 5 %)
    tr = make("Gauß-Klassen", 40, 600000, 0.05)
    K = ev.gram(tr.X, tr.X, "gauss", 1.0)
    r = smo(K, tr.y, 1.0, eps=1e-6, max_steps=3_000_000)
    ref = d["schritte_n"]["own"][0][0]
    if abs(r.steps - ref) > 0.05 * ref + 2:
        bad.append(f"schritte_n: {r.steps} Schritte gegen eingefroren {ref}")
    # barriere: n = 20, erster Zufalls-Seed
    tr = make("Gauß-Klassen", 20, 700000, 0.0)
    K = ev.gram(tr.X, tr.X, "gauss", 1.0)
    nt = Bar.barrier(K, tr.y, 1.0).newton
    ref = d["barriere"]["newton"][0][0]
    if abs(nt - ref) > 0.1 * ref + 3:
        bad.append(f"barriere: {nt} Newton-Schritte gegen eingefroren {ref}")
    # Vollständigkeit
    for k in SECTIONS:
        if k not in d:
            bad.append(f"Abschnitt {k} fehlt")
    for b in bad:
        print("FEHLER:", b)
    print("wmr_frozen.json: Stichproben stimmen" if not bad else f"{len(bad)} Abweichungen")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", action="append", choices=list(SECTIONS))
    ap.add_argument("--parts", help="Ordner für Teilergebnisse (eine Datei je Abschnitt)")
    ap.add_argument("--merge", help="Teilergebnisse aus diesem Ordner in wmr_frozen.json einsetzen")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(check())
    if a.merge:
        d = load()
        for k in SECTIONS:
            p = os.path.join(a.merge, k + ".json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    d[k] = json.load(fh)
        d["meta"] = meta()
        save(d)
        print("zusammengeführt:", [k for k in SECTIONS if k in d])
        return
    todo = a.section or list(SECTIONS)
    d = {} if a.parts else load()
    for k in todo:
        t0 = time.perf_counter()
        part = SECTIONS[k]()
        log(f"[{k}: {time.perf_counter() - t0:.0f} s, nur Orientierung]")
        if a.parts:
            os.makedirs(a.parts, exist_ok=True)
            save(part, os.path.join(a.parts, k + ".json"))
        else:
            d[k] = part
            d["meta"] = meta()
            save(d)


if __name__ == "__main__":
    main()
