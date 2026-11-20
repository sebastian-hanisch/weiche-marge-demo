"""Die Zahl-Fragmente der README: jede Zahl, die in der README aus der eingefrorenen Studie steht, als Text in genau der Form, in der sie dort steht.

`tests/test_claims.py` prüft, dass jedes Fragment in der README vorkommt, und rechnet die wichtigsten Zahlen unabhängig von `wmr_studies.py` direkt aus den Rohwerten nach. Das Format ist das der App
(Dezimalkomma, `wmr_evaluation.fmt`)."""

import json
from pathlib import Path

import numpy as np

import wmr_studies as St
from wmr_evaluation import fmt, fmt_c, fmt_g

ROOT = Path(__file__).resolve().parent.parent
D = json.loads((ROOT / "wmr_frozen.json").read_text(encoding="utf-8"))
CS = D["c_wirkung"]["C"]
GRID_C = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GRID_S = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]


def _g(key, g):
    return St.case_group(D, key, g)


def _means(key, g):
    return St.mean_by_c(_g(key, g)["test"])


# Welche Fragmente je Fall in der README stehen (die übrigen Werte der Kurven stehen nur in der App)
CASE_KEYS = {
    "gauss_lin": ("span", "span_ab_0_01", "c_min", "best"),
    "gauss_lin_flip": ("c_min", "span_ab_0_1", "c_1"),
    "gauss_rbf": ("span", "c_min", "c_1", "c_max"),
    "gauss_rbf_flip": ("c_min", "c_1", "c_max"),
    "ringe_rbf": ("c_min", "c_1", "c_max"),
    "ringe_rbf_flip": ("c_min", "c_1"),
}


def majority_check(c=0.001, seeds=range(1001, 1031)):
    """Zu kleines C (Gauß-Klassen mit 10 % vertauschten Etiketten, Gauß σ = 1, n = 60, eigenes SMO der App): in den Zufalls-Seeds mit unausgewogenen Klassen ordnet die Lösung alle Testpunkte der Mehrheitsklasse zu,
    in den ausgewogenen liegt der Testfehler nahe dem besten C."""
    import wmr_evaluation as ev
    import wmr_scenario as Sc

    n_un = n_all = 0
    errs = []
    for s in seeds:
        inst = Sc.make_instance("Gauß-Klassen", 60, s, 0.1)
        te = Sc.test_set("Gauß-Klassen", 2000, s)
        m = ev.fit(inst, "gauss", 1.0, c, 1e-3, 30000)
        pred = np.where(m.decision(te.X) >= 0, 1.0, -1.0)
        npos, nneg = int((inst.y > 0).sum()), int((inst.y < 0).sum())
        if npos != nneg:
            n_un += 1
            maj = 1.0 if npos > nneg else -1.0
            n_all += int((pred == maj).all())
        else:
            errs.append(float((pred != te.y).mean()))
    return {"n_unausgewogen": n_un, "n_alle_mehrheit": n_all, "n_ausgewogen": len(errs), "err_min": min(errs), "err_max": max(errs)}


def facts() -> dict:
    f = {}
    # --- C-Wirkung (Mittel über 30 Zufalls-Seeds je Gruppe)
    for key in D["c_wirkung"]["faelle"]:
        m = [_means(key, g) for g in (1, 2)]
        werte = {
            "span": lambda v: St.span(v), "span_ab_0_01": lambda v: St.span(v[1:]), "span_ab_0_1": lambda v: St.span(v[2:]),
            "c_min": lambda v: v[0], "c_max": lambda v: v[-1], "c_1": lambda v: v[3], "best": lambda v: min(v),
        }
        for name in CASE_KEYS.get(key, ()):  # nur die Fragmente, die die README nennt
            f[f"{key}_{name}"] = " und ".join(fmt(werte[name](v), 4) for v in m)
    f["bayes_gauss"] = fmt(D["bayes"]["gauss"], 4)
    f["bayes_ringe"] = fmt(D["bayes"]["ringe"]["0.35"]["integral"], 4)
    f["bayes_ringe_stichprobe"] = fmt(D["bayes"]["ringe"]["0.35"]["stichprobe"], 4)
    # --- Löser meldet nichts
    t = [St.solver_totals(D, g) for g in (1, 2)]
    f["cells"] = str(t[0]["cells"])
    f["warn_cap"] = f"{t[0]['warn_cap']} und {t[1]['warn_cap']}"
    f["gap_max"] = f"{fmt_g(t[0]['gap_max'], 3)} und {fmt_g(t[1]['gap_max'], 3)}"
    f["gap_over"] = f"{t[0]['gap_over_tol']} und {t[1]['gap_over_tol']}"
    f["bad"] = f"{t[0]['bad']} und {t[1]['bad']}"
    w = St.warn_harte(D)
    f["warn_harte"] = " / ".join(str(v) for v in w["gepoolt"]) + f" von {w['von']}"
    f["warn_harte_band"] = " / ".join(f"{a} bis {b}" for a, b in zip(w["min"], w["max"]))
    # --- Wahl
    for g in (1, 2):
        for fk in ("flip0", "flip10"):
            x = St.wahl(D, "wahl60", g, fk)
            f[f"wahl60_{g}_{fk}"] = (f"{fmt(x['rueckblick'], 4)} / {fmt(x['cv'], 4)} / {fmt(x['default'], 4)}", f"{fmt(x['diff_mean'], 4)} (Standardfehler {fmt(x['diff_se'], 4)}, {fmt(x['diff_se_zahl'], 1)} Standardfehler)",
                                     f"{x['cv_besser']} / {x['cv_schlechter']}")
            if fk == "flip10":  # die README nennt für 10 % vertauscht nur die ersten beiden Werte
                f[f"wahl60_{g}_{fk}"] = f[f"wahl60_{g}_{fk}"][:2]
    for m in (30, 100, 500):
        v = [St.validierung(D, g, m) for g in (1, 2)]
        f[f"val_{m}"] = " / ".join(f"{fmt(x['optimistisch'], 4)} gegen {fmt(x['ehrlich'], 4)} (Optimismus {fmt(x['optimismus'], 4)}, Standardfehler {fmt(x['se'], 4)})" for x in v)
    for fk in ("flip0", "flip10"):
        p = St.wahl_pooled(D, "wahl200", fk)
        f[f"wahl200_{fk}"] = f"{fmt(p['mean'], 4)} (Standardfehler {fmt(p['se'], 4)}, {fmt(p['se_zahl'], 1)} Standardfehler; besser in {p['besser']}, schlechter in {p['schlechter']}, gleich in {p['gleich']})"
        f[f"wahl200_{fk}_gruppen"] = " / ".join(fmt(v, 4) for v in p["je_gruppe"])
    for g in (1, 2):
        sp = St.sigma_zeile(D, g, "flip0")
        f[f"sigma_spans_{g}"] = ", ".join(fmt(v, 4) for v in sp)
    x1, x2 = St.wahl(D, "wahl60", 1, "flip0"), St.wahl(D, "wahl60", 2, "flip0")
    zellen = []
    for g in (1, 2):
        w = St.wahl(D, "wahl60", g, "flip0")
        zellen.append(f"C = {fmt_c(GRID_C[w['beste_zelle'][0]])}, σ = {fmt_g(GRID_S[w['beste_zelle'][1]], 2)} ({fmt(w['beste_zelle_fehler'], 4)})")
    f["beste_zelle"] = " und ".join(zellen)
    f["sigma0"] = fmt(x1["sigma0_mittel"], 2)
    for key in ("gauss_rbf_flip", "gauss_lin_flip"):
        halb, rest = [], []
        for g in (1, 2):
            t = np.array(_g(key, g)["test"])[0]
            h = np.isclose(t, 0.5, atol=1e-12)
            halb.append(int(h.sum()))
            rest.append(f"{fmt(t[~h].min(), 4)} bis {fmt(t[~h].max(), 4)}")
        f[f"{key}_halb"] = " und ".join(str(v) for v in halb)
        if key == "gauss_rbf_flip":  # die README nennt den Bereich der übrigen Zufalls-Seeds nur für den Gauß-Kern
            f[f"{key}_halb_rest"] = " bzw. ".join(rest)
    mj = majority_check()
    f["mehrheit"] = (f"{mj['n_unausgewogen']} unausgewogen", f"{mj['n_alle_mehrheit']} davon ordnen alle 2 000 Testpunkte der Mehrheitsklasse zu", f"{mj['n_ausgewogen']} ausgewogene", f"{fmt(mj['err_min'], 3)} und {fmt(mj['err_max'], 3)}")
    f["default_mittel"] = " und ".join(fmt(St.wahl(D, "wahl60", g, "flip0")["default"], 4) for g in (1, 2))
    f["abstand_best"] = " und ".join(fmt(St.wahl(D, "wahl60", g, "flip0")["default"] - St.wahl(D, "wahl60", g, "flip0")["beste_zelle_fehler"], 4) for g in (1, 2))
    for g in (1, 2):
        rd = St.sigma_raender(St.wahl(D, "wahl60", g, "flip0")["mean_te"])
        f[f"raender_{g}"] = (f"mindestens {fmt(rd['schmal_min'], 4)}", f"zwischen {fmt(rd['breit_klein_c_min'], 4)} und {fmt(rd['breit_klein_c_max'], 4)}", f"höchstens {fmt(rd['breit_gross_c_max'], 4)}")
    # --- Leave-one-out
    e1, e5 = St.loo_e1(D), St.loo_e5(D)
    f["e1"] = (f"{e1['n']} Instanzen", f"{e1['verletzungen']} Verletzungen", f"scharf (Gleichheit) in {e1['scharf']}", f"Median {fmt(e1['ratio_median'], 3)}", f"Maximum {fmt(e1['ratio_max'], 3)}",
               f"{e1['degeneriert']} Instanzen ohne freien Stützvektor", f"{e1['verletzungen_ungunstig']} verletzen")
    f["e5"] = (f"{e5['n']} kleine Instanzen", f"mit dem b des Lösers {e5['verl_solver']} Verletzungen", f"mit dem ungünstigsten b {e5['verl_ungunstig']}", f"{e5['verl_ungunstig_strikt']} auch ohne", f"scharf in {e5['scharf_ungunstig']}",
               f"unterscheiden sich beide Zählungen in {e5['unterschied']}")
    # --- Schritte
    sc = St.steps_c(D, "gauss")
    f["steps_c_gauss"] = " / ".join(fmt(v, 0) for v in sc["own_mean"])
    f["steps_c_gauss_lib"] = " / ".join(fmt(v, 0) for v in sc["lib_mean"])
    f["steps_c_gauss_1000"] = f"{sc['own_min'][-1]} bis {sc['own_max'][-1]}"
    sr = St.steps_c(D, "ringe")
    f["steps_c_ringe"] = " / ".join(fmt(v, 0) for v in sr["own_mean"])
    sn = St.steps_n(D)
    f["steps_n_mean"] = " / ".join(fmt(v, 0) for v in sn["mean_all"])
    f["slope"] = f"{fmt(sn['slope_all'], 2)} (Bootstrap-90-%-Band [{fmt(sn['band'][0], 2)}; {fmt(sn['band'][1], 2)}]"
    f["slope_ab"] = f"{fmt(sn['slope_a'], 2)} und {fmt(sn['slope_b'], 2)}"
    f["slope_lib"] = fmt(sn["slope_lib"], 2)
    f["slope_seed"] = f"Median {fmt(sn['slope_seed_median'], 2)} [{fmt(sn['slope_seed_min'], 2)}; {fmt(sn['slope_seed_max'], 2)}]"
    f["je_verdopplung"] = " / ".join(fmt(v, 2) for v in sn["je_verdopplung"])
    f["n640"] = f"{sn['min'][4]} und {sn['max'][4]}"
    f["je_punkt"] = f"{fmt(sn['je_punkt_min'], 2)} bis {fmt(sn['je_punkt_max'], 2)}"
    b = St.barrier(D)
    f["newton_mean"] = " / ".join(fmt(v, 0) for v in b["newton_mean"])
    f["smo_mean"] = " / ".join(fmt(v, 0) for v in b["smo_mean"])
    f["modell"] = " / ".join(fmt(v, 0) for v in b["modell"])
    f["newton_einzel"] = f"{b['newton_min']} bis {b['newton_max']}"
    f["abh"] = f"{fmt(b['abh_min'], 0)} bis {fmt(b['abh_max'], 0)}"
    f["trust"] = " / ".join(str(v) for v in b["trust"] if v is not None)
    f["trust_dual"] = fmt_g(b["trust_dual_rel_max"], 2)
    lv = St.live_vs_studie(D)
    f["lvs_dual"] = fmt_g(lv["dual_rel_max"], 2)
    f["lvs_gleich"] = f"{lv['gleich']} von {lv['instanzen']}"
    f["lvs_ratio"] = f"{fmt(lv['ratio_min'], 2)} bis {fmt(lv['ratio_max'], 2)} (Median {fmt(lv['ratio_median'], 3)})"
    return f
