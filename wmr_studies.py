"""Aggregate der eingefrorenen Studien (`wmr_frozen.json`): reine Funktionen auf den Rohwerten je Zufalls-Seed.

Die Rohwerte (Testfehler, Schritte, KKT-Lücken, Warnungen je Zufalls-Seed und C) liegen in der Datei; alle Zahlen in README, App und Verdikten werden hier daraus
gerechnet und in `tests/test_claims.py` festgehalten. Zähleinheiten: Zufalls-Seeds (30 je Gruppe, 2 Gruppen), Zellen (Zufalls-Seed × C), Instanzen, Aktualisierungsschritte,
Newton-Schritte. Standardfehler des Mittels: Standardabweichung (ddof = 1) durch Wurzel der Zahl der Werte."""

import math

import numpy as np

from wmr_rng import SplitMix64


def se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


# ---------------------------------------------------------------- C-Wirkung
def case_group(d: dict, key: str, group: int) -> dict:
    return d["c_wirkung"]["faelle"][key]["gruppen"][str(group)]


def mean_by_c(arr) -> np.ndarray:
    """Mittel über die Zufalls-Seeds je C (arr: [C][Zufalls-Seed])."""
    return np.array([np.mean(v) for v in arr])


def span(x) -> float:
    x = np.asarray(x, float)
    return float(x.max() - x.min())


def flat_from(means, cs, tol: float):
    """Kleinstes C, ab dem die mittleren Fehler (für alle größeren C) höchstens um `tol` auseinanderliegen; None, wenn schon der letzte Wert allein... (immer definiert)."""
    means = np.asarray(means, float)
    for i in range(len(means)):
        if span(means[i:]) <= tol:
            return cs[i]
    return cs[-1]


def bad_cells(arr, slack: float = 0.05) -> tuple:
    """Zellen (Zufalls-Seed × C) mit Testfehler mehr als `slack` über dem besten C desselben Zufalls-Seeds: (Anzahl, Zellen)."""
    a = np.array(arr, float)  # [C][seed]
    best = a.min(axis=0)
    return int((a > best[None, :] + slack).sum()), int(a.size)


def solver_totals(d: dict, group: int, tol: float = 1e-3) -> dict:
    """Die „Löser meldet nichts“-Zählung über alle Fälle einer Gruppe: Zellen, Warnungen ohne und mit Grenze 100 000, größte KKT-Lücke, Zellen mit Lücke über tol, Zellen mit Fehler > 0,05 über dem besten C."""
    cells = warn_unb = warn_cap = gap_over = bad = 0
    gmax = -np.inf
    by_case = {}
    for key in d["c_wirkung"]["faelle"]:
        g = case_group(d, key, group)
        n_cells = int(np.array(g["test"]).size)
        cells += n_cells
        wu = int(np.sum(g["warn_unbegrenzt"]))
        wc = int(np.sum(g["warn100k"]))
        warn_unb += wu
        warn_cap += wc
        gm = float(np.max(g["gap"]))
        gmax = max(gmax, gm)
        gap_over += int((np.array(g["gap"]) > tol + 1e-9).sum())
        b, _ = bad_cells(g["test"])
        bad += b
        by_case[key] = {"cells": n_cells, "warn_cap": wc, "bad": b}
    return {"cells": cells, "warn_unbegrenzt": warn_unb, "warn_cap": warn_cap, "gap_max": float(gmax), "gap_over_tol": gap_over, "bad": bad, "by_case": by_case}


def warn_cap_cells(d: dict, group: int) -> list:
    """(Fall, C, Zahl der Zufalls-Seeds mit Warnung bei Grenze 100 000) für alle Zellen mit mindestens einer Warnung."""
    out = []
    cs = d["c_wirkung"]["C"]
    for key in d["c_wirkung"]["faelle"]:
        g = case_group(d, key, group)
        for ci, c in enumerate(cs):
            k = int(np.sum(g["warn100k"][ci]))
            if k:
                out.append((key, c, k))
    return out


# ---------------------------------------------------------------- Warnungen bei „harter Marge“ per C (Gegenprobe; die Falle führt `harte-marge-demo`)
def warn_harte(d: dict) -> dict:
    w = np.array(d["warn_harte"]["warn"])  # [Satz][C]
    return {"C": d["warn_harte"]["C"], "min": w.min(axis=0).tolist(), "max": w.max(axis=0).tolist(), "gepoolt": w.sum(axis=0).tolist(), "von": int(w.shape[0] * d["warn_harte"]["je_satz"])}


# ---------------------------------------------------------------- Wahl von C und σ
def _arr(d, key, group, flip):
    return d[key]["gruppen"][str(group)][flip]


def wahl(d: dict, key: str, group: int, flip: str) -> dict:
    """Wahl der Rasterzelle (6 C × 6 σ) je Zufalls-Seed: Rückblick bestes Feld, 5-fache Kreuzvalidierung, Voreinstellung (C = 1, γ = 1/(p·Var(X))), gepaarte Differenz CV minus Voreinstellung."""
    g = _arr(d, key, group, flip)
    te = np.array(g["te"])  # [seed][36]
    cv = np.array(g["cv"])
    dflt = np.array(g["default"])
    pick = te[np.arange(len(te)), cv.argmin(axis=1)]
    diff = pick - dflt
    best = te.min(axis=1)
    mean_te = te.mean(axis=0)
    ci = int(mean_te.argmin())
    return {
        "n_seeds": len(te), "rueckblick": float(best.mean()), "cv": float(pick.mean()), "default": float(dflt.mean()),
        "diff_mean": float(diff.mean()), "diff_se": se(diff), "diff_se_zahl": float(diff.mean() / se(diff)) if se(diff) > 0 else 0.0,
        "cv_besser": int((diff < 0).sum()), "cv_schlechter": int((diff > 0).sum()), "cv_gleich": int((diff == 0).sum()),
        "cv_hoechstens_1_5": int((pick <= 1.5 * best + 1e-12).sum()),
        "beste_zelle": (int(ci // 6), int(ci % 6)), "beste_zelle_fehler": float(mean_te[ci]), "sigma0_mittel": float(np.mean(g["sigma0"])),
        "mean_te": mean_te.reshape(6, 6),
    }


def wahl_pooled(d: dict, key: str, flip: str) -> dict:
    """Gepaarte Differenz CV minus Voreinstellung über alle Gruppen eines Schlüssels (n = 200: drei Gruppen zu 20)."""
    diffs, per = [], []
    for g in d[key]["gruppen"]:
        gr = _arr(d, key, int(g), flip)
        te = np.array(gr["te"])
        cv = np.array(gr["cv"])
        dd = te[np.arange(len(te)), cv.argmin(axis=1)] - np.array(gr["default"])
        diffs.append(dd)
        per.append(float(dd.mean()))
    dp = np.concatenate(diffs)
    return {"n_seeds": len(dp), "mean": float(dp.mean()), "se": se(dp), "se_zahl": float(dp.mean() / se(dp)), "besser": int((dp < 0).sum()), "schlechter": int((dp > 0).sum()),
            "gleich": int((dp == 0).sum()), "je_gruppe": per}


def validierung(d: dict, group: int, m: int) -> dict:
    """Gewinnerauswahl auf einer frisch gezogenen Validierungsmenge der Größe m (ohne vertauschte Etiketten): optimistisch = kleinster Fehler dort, ehrlich = Fehler der gewählten Zelle auf den frischen Testpunkten."""
    g = _arr(d, "wahl60", group, "flip0")
    te = np.array(g["te"])
    va = np.array(g["va"][str(m)])
    k = va.argmin(axis=1)
    opt = va[np.arange(len(va)), k]
    hon = te[np.arange(len(te)), k]
    dd = hon - opt
    return {"m": m, "optimistisch": float(opt.mean()), "ehrlich": float(hon.mean()), "optimismus": float(dd.mean()), "se": se(dd), "n_seeds": len(dd)}


def sigma_zeile(d: dict, group: int, flip: str) -> list:
    """Spanne über C je σ im Mittel der Testfehler (Zeilen: σ; Spalten: C), für die Tabelle „C und Bandbreite“."""
    w = wahl(d, "wahl60", group, flip)
    m = w["mean_te"]  # [C][σ]
    return [span(m[:, si]) for si in range(m.shape[1])]


# ---------------------------------------------------------------- Leave-one-out
def loo_e1(d: dict) -> dict:
    e = d["loo"]["e1"]
    n = len(e)
    verl = sum(1 for r in e if r["loo_solver"] > r["nsv"])
    scharf = sum(1 for r in e if r["loo_solver"] == r["nsv"])
    ties = sum(1 for r in e if r["ties"] > 0)
    degen = [r for r in e if r["free"] == 0 and r["nsv"] < r["n"]]
    ratios = [r["loo_solver"] / max(r["nsv"], 1) for r in e]
    verl_worst = sum(1 for r in e if r["loo_worst"] > r["nsv"])
    per_c = {}
    for r in e:
        v = per_c.setdefault(r["c"], [0, 0, 0, 0])
        v[0] += r["loo_solver"]
        v[1] += r["nsv"]
        v[2] += r["n"]
        v[3] += 1
    return {"n": n, "verletzungen": verl, "scharf": scharf, "gleichstand": ties, "degeneriert": len(degen), "degeneriert_verletzt": sum(1 for r in degen if r["loo_solver"] > r["nsv"]),
            "ratio_median": float(np.median(ratios)), "ratio_max": float(np.max(ratios)), "verletzungen_ungunstig": verl_worst, "je_c": per_c}


def loo_e5(d: dict) -> dict:
    e = d["loo"]["e5"]
    n = len(e)
    return {
        "n": n, "verl_solver": sum(1 for r in e if r["loo_solver"] > r["nsv"]), "verl_ungunstig": sum(1 for r in e if r["loo_worst"] > r["nsv"]),
        "verl_ungunstig_strikt": sum(1 for r in e if r["loo_worst_strict"] > r["nsv"]), "scharf_ungunstig": sum(1 for r in e if r["loo_worst"] == r["nsv"]),
        "unterschied": sum(1 for r in e if r["loo_worst"] != r["loo_solver"]), "einklassig": sum(r["one_class"] for r in e),
    }


# ---------------------------------------------------------------- Schritte
def steps_c(d: dict, key: str) -> dict:
    s = d["schritte_c"]["faelle"][key]
    own = np.array(s["own"])  # [C][seed]
    lib = np.array(s["lib"])
    return {"C": d["schritte_c"]["C"], "own_mean": own.mean(axis=1).tolist(), "lib_mean": lib.mean(axis=1).tolist(), "own_min": own.min(axis=1).tolist(), "own_max": own.max(axis=1).tolist()}


def slope(ns, steps_by_n) -> float:
    return float(np.polyfit(np.log(ns), np.log([np.mean(v) for v in steps_by_n]), 1)[0])


def steps_n(d: dict) -> dict:
    s = d["schritte_n"]
    ns = s["n"]
    own = [np.array(v) for v in s["own"]]
    lib = [np.array(v) for v in s["lib"]]
    a, b = [v[:10] for v in own], [v[10:] for v in own]
    # Bootstrap-90-%-Band der Steigung der Mittel über die 20 Zufalls-Seeds (feste Zufallsquelle der Demo)
    rng = SplitMix64(12999)
    boots = []
    for _ in range(2000):
        idx = [rng.below(20) for _ in range(20)]
        boots.append(slope(ns, [v[idx] for v in own]))
    per_seed = [float(np.polyfit(np.log(ns), np.log([own[i][sd] for i in range(len(ns))]), 1)[0]) for sd in range(20)]
    mm = [float(np.mean(v)) for v in own]
    return {
        "n": ns, "mean_all": mm, "mean_a": [float(np.mean(v)) for v in a], "mean_b": [float(np.mean(v)) for v in b],
        "min": [int(v.min()) for v in own], "max": [int(v.max()) for v in own],
        "slope_a": slope(ns, a), "slope_b": slope(ns, b), "slope_all": slope(ns, own), "slope_lib": slope(ns, lib),
        "band": (float(np.percentile(boots, 5)), float(np.percentile(boots, 95))),
        "slope_seed_median": float(np.median(per_seed)), "slope_seed_min": float(min(per_seed)), "slope_seed_max": float(max(per_seed)),
        "je_verdopplung": [math.log(mm[i + 1] / mm[i]) / math.log(2) for i in range(len(ns) - 1)],
        "je_punkt_min": min(m / n for m, n in zip(mm, ns)), "je_punkt_max": max(m / n for m, n in zip(mm, ns)),
    }


def barrier(d: dict) -> dict:
    b = d["barriere"]
    ns = b["n"]
    newton = [np.array(v) for v in b["newton"]]
    smo = [np.array(v) for v in b["smo"]]
    modell = [float(nt.mean() * n ** 3 / 3.0 / (sm.mean() * n)) for n, nt, sm in zip(ns, newton, smo)]
    allv = np.concatenate(newton)
    abh = {k: float(np.mean(v)) for k, v in b["abhaengigkeit_n80"].items()}
    trust = [(int(np.mean(v)) if v else None) for v in b["trust_nit"]]
    return {"n": ns, "newton_mean": [float(v.mean()) for v in newton], "smo_mean": [float(v.mean()) for v in smo], "modell": modell,
            "newton_min": int(allv.min()), "newton_max": int(allv.max()), "abh": abh, "abh_min": min(abh.values()), "abh_max": max(abh.values()),
            "trust": trust, "trust_dual_rel_max": float(max((max(v) for v in b["trust_dual_rel"] if v), default=0.0))}


def live_vs_studie(d: dict) -> dict:
    v = d["live_vs_studie"]
    r = np.array(v["verhaeltnis"])
    return {"instanzen": v["instanzen"], "n": v["n"], "dual_rel_max": v["dual_rel_max"], "gleich": v["gleich"], "ratio_min": float(r.min()), "ratio_max": float(r.max()), "ratio_median": float(np.median(r))}


def sigma_raender(mean_te) -> dict:
    """Ränder des Rasters 6 C × 6 σ (Zeilen: C = 0,01 bis 1000, Spalten: σ = 0,1 bis 3,2) aus dem mittleren Testfehler: kleinster Fehler bei der schmalsten Bandbreite über alle C, Fehler der breitesten Bandbreite bei den
    zwei kleinsten und bei den zwei größten C."""
    m = np.asarray(mean_te, float)
    return {"schmal_min": float(m[:, 0].min()), "breit_klein_c_min": float(m[:2, -1].min()), "breit_klein_c_max": float(m[:2, -1].max()), "breit_gross_c_max": float(m[-3:, -1].max())}
