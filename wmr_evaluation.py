"""Live-Rechnungen (eigenes SMO), Kennzahlen, Leave-one-out und die eingefrorenen Studien.

Kleine Einheiten (DEMO-PLAYBOOK §1b), je eine Regel:
  gram                 Gram-Matrix (linear oder Gauß mit Bandbreite σ; γ = 1/(2σ²) nur zur Umrechnung)
  fit / Model          eigenes SMO (`wmr_smo.smo`, Kopie aus `harte-marge-demo`) mit Schrittgrenze; Entscheidungsfunktion f(x) = Σ α_j y_j K(x_j, x) + b
  Model.margin         Marge 2/‖w‖ (Abstand der Ebenen f = ±1; viele Lehrtexte meinen 1/‖w‖)
  Model.slack          Schlupf ξ_i = max(0, 1 − y_i f(x_i)); Summe
  Model.groups         drei Gruppen: α = 0 (außerhalb), 0 < α < C (auf der Marge), α = C (in oder jenseits der Marge)
  b_interval           Intervall des optimalen b aus den KKT-Bedingungen (bei α = 0 oder α = C aller Punkte nicht eindeutig)
  loo                  Leave-one-out durch echtes Herausnehmen (Neuberechnung je Punkt), mit dem b des Lösers und mit dem ungünstigsten b des Intervalls
  exp1 bis exp5        die Rechnungen der fünf Experimente der App
Die eingefrorenen Studien (Zufalls-Seeds, Zellen, Instanzen, Schritte) liegen in `wmr_frozen.json` (Werkzeug: tools/wmr_make_frozen.py); die Aggregate rechnet `wmr_studies.py`.
Verdikte sind reine Funktionen (`verdict_*`) mit Rückgabe (Art, Text)."""

import functools
import json
import os
from dataclasses import dataclass, field

import numpy as np

import wmr_barrier as Bar
import wmr_constants as C
import wmr_scenario as Sc
from wmr_rng import SplitMix64
from wmr_smo import kkt_gap, smo

LOO_EPS = 1e-7  # Abbruch der Neuberechnungen beim Herausnehmen (enger als die Voreinstellung der App, damit b und α stimmen)


# ---------------------------------------------------------------- Formatierung
def fmt(x, nd: int = 3) -> str:
    """Zahl mit Dezimalkomma und festen Nachkommastellen; −0,000 wird zu 0,000."""
    if x is None:
        return "–"
    if isinstance(x, float) and not np.isfinite(x):
        return "∞" if x > 0 else "−∞"
    s = f"{float(x):.{nd}f}".replace(".", ",")
    if s.startswith("-") and float(s.replace(",", ".")) == 0.0:
        s = s[1:]
    return s.replace("-", "−")


def fmt_g(x, nd: int = 3) -> str:
    """Zahl in allgemeiner Darstellung (g) mit Dezimalkomma."""
    return f"{float(x):.{nd}g}".replace(".", ",").replace("-", "−")


def fmt_c(c) -> str:
    """C kompakt: 0,001 / 0,1 / 1 / 100 / 1000."""
    c = float(c)
    return fmt_g(c, 4) if c < 1 else (f"{c:.0f}")


# ---------------------------------------------------------------- Studien
_STUDY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.STUDY_FILE)


@functools.lru_cache(maxsize=1)
def studies() -> dict:
    """Die eingefrorenen Studien (leer, wenn die Datei fehlt)."""
    try:
        with open(_STUDY_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------- Kern und Modell
def gram(X, Z, kind: str, sigma: float = 1.0) -> np.ndarray:
    """Gram-Matrix K(x_i, z_j): linear x·z oder Gauß exp(−‖x − z‖²/(2σ²))."""
    X = np.asarray(X, float)
    Z = np.asarray(Z, float)
    if kind == "linear":
        return X @ Z.T
    d2 = (X ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2.0 * (X @ Z.T)
    return np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma ** 2))


def gamma_of(sigma: float) -> float:
    """γ = 1/(2σ²) (Umrechnung für libsvm und scikit-learn, die `gamma` sagen)."""
    return 1.0 / (2.0 * sigma ** 2)


def kind_of(kern: str) -> str:
    return "linear" if kern == "linear" else "gauss"


def _tol_alpha(c: float) -> float:
    """Toleranz für „α = 0“ und „α = C“: relativ zu C (das SMO setzt Werte an den Schranken exakt; bei C = 1e-9 wäre eine absolute Toleranz größer als C)."""
    return 1e-9 * c if np.isfinite(c) else 1e-12


@dataclass
class Model:
    X: np.ndarray
    y: np.ndarray
    kind: str
    sigma: float
    c: float
    K: np.ndarray
    alpha: np.ndarray
    b: float
    steps: int
    gap: float
    converged: bool
    stopped_by: str
    message: str
    dual: float
    trace: dict | None = field(default=None, repr=False)

    def decision(self, Z) -> np.ndarray:
        """f(z) = Σ α_j y_j K(x_j, z) + b."""
        return gram(Z, self.X, self.kind, self.sigma) @ (self.alpha * self.y) + self.b

    def w_norm(self) -> float:
        ya = self.alpha * self.y
        return float(np.sqrt(max(ya @ self.K @ ya, 0.0)))

    def margin(self) -> float:
        """Marge 2/‖w‖; unendlich, wenn w = 0 (alle α = C bei ausgewogenen Klassen: die Funktion ist dann konstant)."""
        w = self.w_norm()
        return float("inf") if w < 1e-9 else 2.0 / w

    def slack(self) -> np.ndarray:
        f = self.K @ (self.alpha * self.y) + self.b
        return np.maximum(0.0, 1.0 - self.y * f)

    def groups(self) -> dict:
        """Zahl der Punkte: außerhalb (α = 0), auf der Marge (0 < α < C), an der Schranke (α = C)."""
        t = _tol_alpha(self.c)
        out = int((self.alpha <= t).sum())
        bound = int((self.alpha >= self.c - t).sum()) if np.isfinite(self.c) else 0
        return {"aussen": out, "marge": len(self.alpha) - out - bound, "schranke": bound}

    def kkt_gap(self) -> float:
        return kkt_gap(self.K, self.y, self.alpha, self.c)


def fit(inst: Sc.Instance, kind: str, sigma: float, c: float, eps: float, max_steps: int, wss: str = "second", trace: bool = False, trace_every: int = 1,
        rng_seed: int = 0) -> Model:
    """Eigenes SMO auf der Gram-Matrix; bei nur einer Klasse gibt es nichts zu lösen (`stopped_by == "no_pair"`)."""
    K = gram(inst.X, inst.X, kind, sigma)
    rng = SplitMix64(rng_seed) if wss == "random" else None
    r = smo(K, inst.y, c, eps=eps, max_steps=max_steps, wss=wss, rng=rng, trace=trace, trace_every=trace_every)
    return Model(inst.X, inst.y, kind, sigma, c, K, r.alpha, r.b, r.steps, r.gap, r.converged, r.stopped_by, r.message, r.dual, r.trace)


def error(model: Model, test: Sc.Instance) -> float:
    """Anteil falsch eingeordneter Punkte (f ≥ 0 heißt +1)."""
    f = model.decision(test.X)
    return float((np.where(f >= 0, 1.0, -1.0) != test.y).mean())


def train_error(model: Model) -> float:
    f = model.K @ (model.alpha * model.y) + model.b
    return float((np.where(f >= 0, 1.0, -1.0) != model.y).mean())


@functools.lru_cache(maxsize=8)
def bayes_error(data: str) -> float | None:
    """Bayes-Fehler des Erzeugers (exakt): Gauß-Klassen Φ(−1), Ringe mit Streuung 0,35 als Integral; das Handbeispiel hat keinen."""
    if data == C.DATASETS[0]:
        return Sc.bayes_gauss()
    if data == C.DATASETS[1]:
        return Sc.bayes_ringe(0.35)
    return None


# ---------------------------------------------------------------- b-Intervall und Leave-one-out
def b_interval(K, y, alpha, c: float, tol: float | None = None):
    """Intervall des optimalen b aus den KKT-Bedingungen: y_i (f0_i + b) ≥ 1 bei α_i = 0, ≤ 1 bei α_i = C, = 1 dazwischen (f0 = Σ α_j y_j K_ij ohne b).

    Gibt (lo, hi) zurück; ein Ende ist ±∞, wenn es keine Einschränkung gibt. Bei einem freien Stützvektor ist lo = hi."""
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    tol = _tol_alpha(c) if tol is None else tol
    f0 = K @ (np.asarray(alpha, float) * y)
    t = y - f0  # b-Wert, bei dem y_i (f0_i + b) = 1 gilt
    lo, hi = -np.inf, np.inf
    for i in range(len(y)):
        if alpha[i] <= tol:
            if y[i] > 0:
                lo = max(lo, t[i])
            else:
                hi = min(hi, t[i])
        elif alpha[i] >= c - tol:
            if y[i] > 0:
                hi = min(hi, t[i])
            else:
                lo = max(lo, t[i])
        else:
            lo = max(lo, t[i])
            hi = min(hi, t[i])
    return float(lo), float(hi)


def loo(inst: Sc.Instance, kind: str, sigma: float, c: float, max_steps: int = C.LOO_STEPS, eps: float = LOO_EPS) -> dict:
    """Leave-one-out durch echtes Herausnehmen: je Punkt eine Neuberechnung ohne ihn, dann y f an dem Punkt.

    Zwei Zählungen: mit dem b des Lösers (Rückgabe von `smo`: Mittel der freien Stützvektoren, sonst Mitte der Schranken) und mit dem ungünstigsten b des optimalen
    Intervalls (f = 0 zählt als Fehler). Ein Rest mit nur einer Klasse hat keine trennende Funktion und zählt als Fehler (Konvention, getrennt ausgewiesen).
    Vergleichsgröße: Zahl der Stützvektoren der vollen Lösung (α > 0, Verletzer eingeschlossen)."""
    n = inst.n
    full = fit(inst, kind, sigma, c, eps, max_steps)
    K = full.K
    y = inst.y
    err_solver = np.zeros(n, bool)
    err_worst = np.zeros(n, bool)
    one_class = np.zeros(n, bool)
    nonconv = np.zeros(n, bool)
    for i in range(n):
        keep = np.delete(np.arange(n), i)
        if len(np.unique(y[keep])) < 2:
            one_class[i] = err_solver[i] = err_worst[i] = True
            continue
        r = smo(K[np.ix_(keep, keep)], y[keep], c, eps=eps, max_steps=max_steps)
        nonconv[i] = not r.converged
        f0 = float(K[i, keep] @ (r.alpha * y[keep]))
        err_solver[i] = y[i] * (f0 + r.b) <= 0
        lo, hi = b_interval(K[np.ix_(keep, keep)], y[keep], r.alpha, c)
        ends = [b for b in (lo, hi) if np.isfinite(b)] or [r.b]
        err_worst[i] = min(y[i] * (f0 + b) for b in ends) <= 0
    t = _tol_alpha(c)
    n_sv = int((full.alpha > t).sum())
    return {
        "n": n, "n_sv": n_sv, "err_solver": err_solver, "err_worst": err_worst, "one_class": one_class, "nonconv": nonconv, "was_sv": full.alpha > t,
        "loo_solver": int(err_solver.sum()), "loo_worst": int(err_worst.sum()), "full": full,
        "violated_solver": int(err_solver.sum()) > n_sv, "violated_worst": int(err_worst.sum()) > n_sv,
    }


# ---------------------------------------------------------------- Handbeispiel in Brüchen
@functools.lru_cache(maxsize=1)
def hand_table():
    """Tabelle des Handbeispiels aus der Aufzählung in Brüchen (3^5 = 243 Zuordnungen je C), dazu der Dualwert des eigenen SMO und seine Schritte."""
    import wmr_exact as Ex
    from fractions import Fraction as F

    rows = []
    inst = Sc.fuenf_punkte()
    for num, den in C.HAND_C:
        z = Ex.knick_zeile(F(num, den))
        m = fit(inst, "linear", 1.0, num / den, 1e-9, 200000)
        z["smo_steps"] = m.steps
        z["smo_dual"] = m.dual
        z["smo_converged"] = m.converged
        rows.append(z)
    return tuple(rows)


# ---------------------------------------------------------------- Experiment 1 und 2
def study_case(data: str, kern: str, sigma: float, flip: float) -> str | None:
    """Schlüssel des eingefrorenen Falls zu den Einstellungen, sonst None (die Studie deckt nur die gemessenen Fälle ab)."""
    if data == C.DATASETS[0]:
        if kern == "linear":
            return "gauss_lin_flip" if flip > 0 else "gauss_lin"
        if abs(sigma - 1.0) < 1e-9:
            return "gauss_rbf_flip" if flip > 0 else "gauss_rbf"
    if data == C.DATASETS[1] and kern != "linear" and abs(sigma - 0.5) < 1e-9:
        return "ringe_rbf_flip" if flip > 0 else "ringe_rbf"
    return None


def settings_instance(data: str, n: int, seed: int, flip: float) -> Sc.Instance:
    return Sc.make_instance(data, n, seed, flip)


def exp1(data: str, kern: str, sigma: float, c: float, n: int, seed: int, flip: float, eps: float, max_steps: int) -> dict:
    """Eine Anpassung mit dem eigenen SMO: Marge, Schlupf, Gruppen, Fehler auf 2 000 frischen Punkten (nicht beim Handbeispiel: dort der Trainingsfehler)."""
    inst = settings_instance(data, n, seed, flip)
    kind = kind_of(kern)
    m = fit(inst, kind, sigma, c, eps, max_steps, trace=False)
    hand = data == C.DATASETS[2]
    test = None if hand else Sc.test_set(data, C.N_TEST, seed)
    return {
        "inst": inst, "model": m, "test": test, "hand": hand, "kind": kind, "sigma": sigma, "c": c,
        "test_err": None if hand else error(m, test), "train_err": train_error(m), "bayes": bayes_error(data),
        "margin": m.margin(), "slack_sum": float(m.slack().sum()), "groups": m.groups(), "gap": m.kkt_gap(),
        "case": study_case(data, kern, sigma, flip),
    }


def exp2(data: str, kern: str, sigma: float, n: int, seed: int, flip: float, eps: float, max_steps: int) -> dict:
    """Die C-Zeile: derselbe Datensatz mit allen C der Auswahl, je Zeile KKT-Lücke, Meldung des Lösers, Testfehler und Schritte."""
    inst = settings_instance(data, n, seed, flip)
    kind = kind_of(kern)
    hand = data == C.DATASETS[2]
    test = None if hand else Sc.test_set(data, C.N_TEST, seed)
    rows = []
    for c in C.C_OPTIONS:
        m = fit(inst, kind, sigma, c, eps, max_steps)
        rows.append({
            "c": c, "gap": m.kkt_gap(), "converged": m.converged, "message": m.message, "steps": m.steps,
            "err": train_error(m) if hand else error(m, test), "sv_frac": float((m.alpha > _tol_alpha(c)).mean()), "margin": m.margin(),
        })
    return {"rows": rows, "hand": hand, "eps": eps, "max_steps": max_steps, "inst": inst}


# ---------------------------------------------------------------- Experiment 4 (Leave-one-out) und 5 (Schritte)
def exp4(data: str, kern: str, sigma: float, c: float, n: int, seed: int, flip: float, max_steps: int = C.LOO_STEPS) -> dict:
    inst = settings_instance(data, n, seed, flip)
    r = loo(inst, kind_of(kern), sigma, c, max_steps=max_steps)
    r["inst"] = inst
    return r


def exp5(data: str, kern: str, sigma: float, c: float, n: int, seed: int, flip: float, eps: float, max_steps: int, wss: str) -> dict:
    """Spur des SMO (Dualwert und KKT-Lücke je Schritt) und die Newton-Schritte der Barriere auf demselben QP."""
    inst = settings_instance(data, n, seed, flip)
    kind = kind_of(kern)
    m = fit(inst, kind, sigma, c, eps, max_steps, wss=wss, trace=True, trace_every=1, rng_seed=seed)
    bar = None
    if len(np.unique(inst.y)) == 2 and np.isfinite(c):
        bar = Bar.barrier(m.K, inst.y, c)
    n_ = inst.n
    return {
        "inst": inst, "model": m, "barrier": bar,
        "modell_newton": (bar.newton * n_ ** 3 / 3.0) if bar else None, "modell_smo": m.steps * n_,
    }


# ---------------------------------------------------------------- Verdikte (reine Funktionen, Rückgabe (Art, Text); Art ist der Name einer st-Funktion)
FLAT_TOL = 0.01  # „flach“ heißt: Spanne der mittleren Testfehler ab diesem C höchstens 0,01
BIG_TOL = 0.05  # „stark“ bzw. „weit über dem besten“: mehr als 0,05 Testfehler
SMALL_TOL = 0.02  # Spanne höchstens 0,02: Zählschwelle für „C ändert den Testfehler wenig“


def verdict_c(cs, means_by_group, bayes=None, label: str = ""):
    """C-Wirkung aus der eingefrorenen Kurve (mittlerer Testfehler je C, eine Liste je Zufalls-Seed-Gruppe), nur mit gezählten Schwellen:
    Spanne in jeder Gruppe höchstens 0,02 (`info`), über 0,05 in jeder Gruppe (`warning`), sonst dazwischen (`info`); dazu der flache Bereich (Spanne ab C höchstens 0,01)."""
    import wmr_studies as St

    spans = [St.span(m) for m in means_by_group]
    flat = max(St.flat_from(m, cs, FLAT_TOL) for m in means_by_group)
    lo = min(min(m) for m in means_by_group)
    hi = max(max(m) for m in means_by_group)
    best_c = [cs[int(np.argmin(m))] for m in means_by_group]
    worst_c = [cs[int(np.argmax(m))] for m in means_by_group]
    head = f"{label}: " if label else ""
    sp_text = " und ".join(fmt(s, 4) for s in spans)

    def both(a, b):
        return fmt_c(a) if a == b else f"{fmt_c(a)} (Gruppe 1) bzw. {fmt_c(b)} (Gruppe 2)"

    txt = (f"{head}Über C = {fmt_c(cs[0])} bis {fmt_c(cs[-1])} liegt der mittlere Testfehler zwischen {fmt(lo, 4)} und {fmt(hi, 4)}, am kleinsten bei C = {both(best_c[0], best_c[-1])}, "
           f"am größten bei C = {both(worst_c[0], worst_c[-1])}; Spanne {sp_text} in den zwei Gruppen zu 30 Zufalls-Seeds (n = 60, 2 000 Testpunkte je Zufalls-Seed)")
    if bayes is not None:
        txt += f"; Bayes-Fehler {fmt(bayes, 4)}"
    txt += ". "
    later = [St.span(m[list(cs).index(flat):]) for m in means_by_group]
    txt += f"Ab C = {fmt_c(flat)} beträgt die Spanne höchstens {fmt(max(later), 4)} (flach heißt: höchstens {fmt(FLAT_TOL, 2)}). "
    if max(spans) <= SMALL_TOL:
        txt += f"Die Spanne liegt in beiden Gruppen bei höchstens {fmt(SMALL_TOL, 2)}: für diese Instanzfamilie und diesen Kern ändert C den mittleren Testfehler um höchstens diesen Betrag."
        return "info", txt
    if min(spans) > BIG_TOL:
        txt += f"Die Spanne liegt in beiden Gruppen über {fmt(BIG_TOL, 2)}: C wirkt hier stark (der Fehler fällt oder steigt mit C um mehr als diesen Betrag)."
        return "warning", txt
    txt += f"Die Spanne liegt zwischen {fmt(SMALL_TOL, 2)} und {fmt(BIG_TOL, 2)} (oder in den beiden Gruppen auf verschiedenen Seiten): C wirkt hier messbar, aber nicht stark."
    return "info", txt


def verdict_loeser(rows, hand: bool = False):
    """Die C-Zeile live: wie viele Zeilen der Löser als konvergiert meldet, die größte KKT-Lücke, wo er an der Schrittgrenze spricht, und ob das Modell trotz Konvergenz schlecht ist."""
    conv = [r for r in rows if r["converged"]]
    spoke = [r for r in rows if not r["converged"]]
    err_name = "Trainingsfehler" if hand else "Testfehler"
    if conv:
        best = min(r["err"] for r in rows)
        bad = [r for r in conv if r["err"] > best + BIG_TOL]
        gmax = max(r["gap"] for r in conv)
    else:
        best, bad, gmax = min(r["err"] for r in rows), [], float("nan")
    txt = (f"In {len(conv)} von {len(rows)} Zeilen meldet der Löser Konvergenz (KKT-Lücke höchstens {fmt_g(gmax, 3)} bei eps = {fmt_g(rows[0].get('eps', float('nan')), 2)}); "
           if conv else f"In keiner der {len(rows)} Zeilen erreicht der Löser die Toleranz; ")
    if spoke:
        txt += f"an der Schrittgrenze spricht er in {len(spoke)} Zeilen (C = {', '.join(fmt_c(r['c']) for r in spoke)}: „nicht konvergiert“). "
    else:
        txt += "eine Meldung gibt es in keiner Zeile. "
    if bad:
        txt += (f"Trotzdem liegt der {err_name} in {len(bad)} der konvergierten Zeilen mehr als {fmt(BIG_TOL, 2)} über dem besten dieser Zeile ({fmt(best, 4)}): "
                f"richtig gelöst, aber das gestellte C ist nicht das passende.")
        return "warning", txt
    txt += f"Der {err_name} liegt in keiner konvergierten Zeile mehr als {fmt(BIG_TOL, 2)} über dem besten ({fmt(best, 4)}): in dieser Zeile kostet kein C mehr als diesen Betrag."
    return "info", txt


def verdict_wahl(val, cv, pooled200=None):
    """Wahl von C (und σ) auf den Ringen: Optimismus der Gewinnerauswahl auf einer Validierungsmenge (`val` aus `wmr_studies.validierung`) und CV gegen Voreinstellung (`cv` aus `wahl`)."""
    z = val["optimismus"] / val["se"] if val["se"] > 0 else 0.0
    txt = (f"Gewinnerauswahl auf {val['m']} frisch gezogenen Validierungspunkten (36 Zellen C × σ, ohne vertauschte Etiketten, {val['n_seeds']} Zufalls-Seeds): "
           f"Fehler dort {fmt(val['optimistisch'], 4)}, dieselbe Wahl auf frischen Punkten {fmt(val['ehrlich'], 4)}, Optimismus {fmt(val['optimismus'], 4)} (Standardfehler {fmt(val['se'], 4)}). "
           f"5-fache Kreuzvalidierung auf 60 Punkten: {fmt(cv['cv'], 4)}, Voreinstellung (C = 1, γ = 1/(p·Var X)): {fmt(cv['default'], 4)}; gepaarte Differenz je Zufalls-Seed {fmt(cv['diff_mean'], 4)} "
           f"(Standardfehler {fmt(cv['diff_se'], 4)}, {fmt(cv['diff_se_zahl'], 1)} Standardfehler; Kreuzvalidierung besser in {cv['cv_besser']}, schlechter in {cv['cv_schlechter']} von {cv['n_seeds']}).")
    if pooled200 is not None:
        txt += f" Bei n = 200: {fmt(pooled200['mean'], 4)} (Standardfehler {fmt(pooled200['se'], 4)}, {pooled200['n_seeds']} Zufalls-Seeds)."
    return ("warning" if z > 2.0 else "info"), txt


def verdict_loo(res: dict):
    """Leave-one-out live: Zählung mit dem b des Lösers und mit dem ungünstigsten b gegen die Zahl der Stützvektoren."""
    n, nsv = res["n"], res["n_sv"]
    ls, lw = res["loo_solver"], res["loo_worst"]
    txt = (f"Von {n} Punkten ordnet das Herausnehmen {ls} falsch ein (b des Lösers), bei {nsv} Stützvektoren: "
           f"{'die Schranke LOO-Fehler ≤ Stützvektoren hält' if ls <= nsv else 'die Schranke wird verletzt'} für dieses b. "
           f"Mit dem ungünstigsten b des optimalen Intervalls sind es {lw}: {'ebenfalls höchstens' if lw <= nsv else 'mehr als'} {nsv}.")
    if res["one_class"].any():
        txt += f" {int(res['one_class'].sum())} Herausnahme(n) lassen nur eine Klasse übrig (zählt als Fehler)."
    if res["nonconv"].any():
        txt += f" Achtung: {int(res['nonconv'].sum())} Neuberechnung(en) erreichten die Schrittgrenze {C.LOO_STEPS}."
    kind = "warning" if (ls > nsv or lw > nsv or res["nonconv"].any()) else "info"
    return kind, txt


def verdict_schritte(res: dict):
    """Schritte zählen live: Aktualisierungsschritte des SMO, Meldung an der Grenze, Newton-Schritte der Barriere."""
    m = res["model"]
    txt = f"SMO: {m.steps} Aktualisierungsschritte (zwei Multiplikatoren je Schritt), KKT-Lücke am Ende {fmt_g(m.gap, 3)}"
    if not m.converged:
        return "warning", txt + f". {m.message}."
    txt += "."
    bar = res["barrier"]
    if bar is not None:
        txt += (f" Barriere (Newton-Schritte, Einstellungen siehe Abschnitt Methode): {bar.newton} Schritte in {bar.outer} äußeren Schritten; Modellzählung ohne Konstanten "
                f"Newton-Schritte · n³/3 gegen SMO-Schritte · n: das {fmt(res['modell_newton'] / res['modell_smo'], 0)}fache.")
    return "info", txt


def verdict_hand(rows):
    """Handbeispiel aus der Aufzählung: wo C die Funktion nicht mehr ändert (Knick), Monotonie von ‖w‖, Schlupfsumme und Dualwert."""
    import wmr_exact as Ex

    w = [r["w"] for r in rows]
    cs = [r["C"] for r in rows]
    mono_w = all(a <= b for a, b in zip(w, w[1:]))
    mono_dual = all(a["dual"] <= b["dual"] for a, b in zip(rows, rows[1:]))
    mono_xi = all(a["xi_sum"] >= b["xi_sum"] for a, b in zip(rows, rows[1:]))
    flat = [r for r in rows if r["w"] == w[-1] and r["b_lo"] == r["b_hi"] == rows[-1]["b_lo"]]
    first_flat = Ex.bruch(flat[0]["C"]) if flat else "–"
    txt = (f"Ab C = {first_flat} sind w = {Ex.bruch(w[-1])} und b = {Ex.bruch(rows[-1]['b_lo'])} in {len(flat)} von {len(rows)} Zeilen gleich: C wirkt dort nicht mehr, obwohl α mit C wächst "
           f"(Summe der Schlupfwerte {Ex.bruch(rows[-1]['xi_sum'])}). Darunter liegen alle vier Stützvektoren bei α = C und b ist nur ein Intervall. "
           f"‖w‖ {'fällt in keiner Zeile' if mono_w else 'fällt in einer Zeile'}, die Schlupfsumme {'steigt in keiner Zeile' if mono_xi else 'steigt in einer Zeile'}, der Dualwert {'fällt in keiner Zeile' if mono_dual else 'fällt in einer Zeile'}.")
    return "info", txt
