"""SMO für das duale Problem der Stützvektormaschine (Quelle für die Stücke 2, 3 und 4 der Linie „Kernmethoden“; die anderen Stücke kopieren diese Datei).

Das Problem (Dual, mit Q_ij = y_i y_j K_ij):  min 1/2 a'Qa - e'a  unter  0 <= a_i <= C  und  y'a = 0.
C = numpy.inf ist die harte Marge (nur eine untere Schranke 0). Verfahren: Aktualisierung zweier Multiplikatoren je Schritt (Platt 1998), Wahl
des Paars nach der Arbeitsmenge zweiter Ordnung (Fan/Chen/Lin 2005, wie LIBSVM; `wss="first"`: größtes verletzendes Paar, `wss="random"`: erster
Partner wie dort, zweiter zufällig aus den verletzenden). Der Gradient G = Qa - e wird je Schritt in O(n) nachgeführt, ohne Schrumpfen und ohne
Cache. Ein Schritt ändert genau zwei Multiplikatoren; `steps` zählt diese Aktualisierungsschritte, nicht Durchläufe (die Zahl von LIBSVM `n_iter_`
ist dieselbe Einheit, aber nicht dieselbe Zahl: andere Gleichstände, anderes Schrumpfen).

Schnittstelle (nur numpy; keine Zufallsquelle im Modul, der Zufall kommt von außen):
  smo(K, y, C, eps=1e-3, max_steps=100000, wss="second", rng=None, trace=False, trace_every=1, tau=1e-12) -> SmoResult
    K      Gram-Matrix (n x n, symmetrisch; muss nicht positiv semidefinit sein, ein Nenner <= 0 wird durch tau ersetzt)
    y      Etiketten in {-1, +1} (Array der Länge n)
    C      Schranke (> 0), numpy.inf für die harte Marge
    eps    Abbruch, sobald die KKT-Lücke m(a) - M(a) <= eps ist (Voreinstellung 1e-3 wie LIBSVM)
    max_steps  Schrittgrenze: bei Erreichen hält das Verfahren an und `message` sagt es (kein stilles Weiterlaufen)
    rng    nur für wss="random": Objekt mit Methode below(n) (z. B. SplitMix64 der aufrufenden Demo)
    trace  True: Spur der Zustände nach 0, trace_every, 2 trace_every, ... und zuletzt nach dem letzten Schritt
  SmoResult: alpha, b, steps, gap, converged, stopped_by ("eps" | "max_steps" | "no_pair"), message (leer bei Erfolg), dual, trace
    dual   Dualwert e'a - 1/2 a'Qa (zu maximieren) am Ende
    trace  None oder dict mit den Arrays "step", "dual", "gap" (gleiche Länge; gap = KKT-Lücke m - M des jeweiligen Zustands)
  dual_value(K, y, alpha)       Dualwert e'a - 1/2 a'Qa
  kkt_gap(K, y, alpha, C)       KKT-Lücke m(a) - M(a) (unabhängig vom Verfahren neu gerechnet; <= 0 oder klein heißt optimal)
"""

from typing import NamedTuple, Optional

import numpy as np


class SmoResult(NamedTuple):
    alpha: np.ndarray
    b: float
    steps: int
    gap: float
    converged: bool
    stopped_by: str
    message: str
    dual: float
    trace: Optional[dict]


def dual_value(K, y, alpha):
    """Dualwert e'a - 1/2 a'Qa (zu maximieren) mit Q_ij = y_i y_j K_ij."""
    ya = np.asarray(y, float) * np.asarray(alpha, float)
    return float(np.sum(alpha) - 0.5 * ya @ np.asarray(K, float) @ ya)


def kkt_gap(K, y, alpha, C):
    """KKT-Lücke m(a) - M(a) nach Fan/Chen/Lin: m = max über I_up von -y_i G_i, M = min über I_low von -y_i G_i, G = Qa - e.

    Gibt 0.0 zurück, wenn I_up oder I_low leer ist (dann ist jede zulässige Verschiebung der Gleichung y'a = 0 gesperrt)."""
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    alpha = np.asarray(alpha, float)
    G = y * (K @ (y * alpha)) - 1.0
    score = -y * G
    up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
    low = ((y < 0) & (alpha < C)) | ((y > 0) & (alpha > 0))
    if not up.any() or not low.any():
        return 0.0
    return float(score[up].max() - score[low].min())


def smo(K, y, C, eps=1e-3, max_steps=100000, wss="second", rng=None, trace=False, trace_every=1, tau=1e-12):
    """Duales QP der Stützvektormaschine mit SMO lösen (Aktualisierung zweier Multiplikatoren je Schritt); siehe Modulkopf für die Schnittstelle."""
    if wss not in ("second", "first", "random"):
        raise ValueError("wss muss 'second', 'first' oder 'random' sein")
    if wss == "random" and rng is None:
        raise ValueError("wss='random' braucht rng mit der Methode below(n)")
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    n = len(y)
    Q = (y[:, None] * y[None, :]) * K
    Kd = np.diag(K).copy()
    pos = y > 0
    yl = y.tolist()
    alpha = np.zeros(n)
    G = -np.ones(n)
    steps = 0
    gap = np.inf
    stopped_by = "eps"
    tr_step, tr_dual, tr_gap = [], [], []
    while True:
        score = -y * G
        lt = alpha < C
        gt = alpha > 0
        up = np.where(pos, lt, gt)
        low = np.where(pos, gt, lt)
        iu = np.flatnonzero(up)
        il = np.flatnonzero(low)
        if len(iu) == 0 or len(il) == 0:
            gap = 0.0
        else:
            i = int(iu[np.argmax(score[iu])])
            m = score[i]
            sl = score[il]
            gap = float(m - sl.min())
        if trace and (steps % trace_every == 0 or gap <= eps or steps >= max_steps):
            tr_step.append(steps)
            tr_dual.append(0.5 * (alpha.sum() - alpha @ G))
            tr_gap.append(gap)
        if len(iu) == 0 or len(il) == 0:
            stopped_by = "no_pair"
            break
        if gap <= eps:
            break
        if steps >= max_steps:
            stopped_by = "max_steps"
            break
        sel = sl < m
        cand = il[sel]
        if wss == "first":
            j = int(il[np.argmin(sl)])
        elif wss == "random":
            j = int(cand[rng.below(len(cand))])
        else:
            gain = m - sl[sel]
            a = Kd[i] + Kd[cand] - 2.0 * K[i, cand]
            a = np.where(a <= 0, tau, a)
            j = int(cand[np.argmin(-(gain * gain) / a)])
        ai_old, aj_old = float(alpha[i]), float(alpha[j])
        ai, aj = ai_old, aj_old
        Gi, Gj = float(G[i]), float(G[j])
        # Zwei-Variablen-Teilproblem entlang y_i a_i + y_j a_j = const, exakt gelöst und auf [0, C] beschnitten
        if yl[i] != yl[j]:
            quad = float(Q[i, i] + Q[j, j] + 2 * Q[i, j])
            quad = tau if quad <= 0 else quad
            delta = (-Gi - Gj) / quad
            diff = ai - aj
            ai += delta
            aj += delta
            if diff > 0:
                if aj < 0:
                    aj = 0.0
                    ai = diff
            else:
                if ai < 0:
                    ai = 0.0
                    aj = -diff
            if diff > 0:
                if ai > C:
                    ai = C
                    aj = C - diff
            else:
                if aj > C:
                    aj = C
                    ai = C + diff
        else:
            quad = float(Q[i, i] + Q[j, j] - 2 * Q[i, j])
            quad = tau if quad <= 0 else quad
            delta = (Gi - Gj) / quad
            s = ai + aj
            ai -= delta
            aj += delta
            if s > C:
                if ai > C:
                    ai = C
                    aj = s - C
            else:
                if aj < 0:
                    aj = 0.0
                    ai = s
            if s > C:
                if aj > C:
                    aj = C
                    ai = s - C
            else:
                if ai < 0:
                    ai = 0.0
                    aj = s
        alpha[i] = ai
        alpha[j] = aj
        G += Q[i] * (ai - ai_old) + Q[j] * (aj - aj_old)
        steps += 1
    score = -y * G
    tol_a = 1e-12 * max(1.0, C if np.isfinite(C) else 1.0)
    free = (alpha > tol_a) & (alpha < C - tol_a)
    if free.any():
        bias = float(np.mean(score[free]))
    else:
        up = np.where(pos, alpha < C, alpha > 0)
        low = np.where(pos, alpha > 0, alpha < C)
        bias = float(0.5 * (score[up].max() + score[low].min())) if up.any() and low.any() else 0.0
    converged = stopped_by != "max_steps"
    message = ""
    if stopped_by == "max_steps":
        message = f"Schrittgrenze {max_steps} erreicht, KKT-Lücke {gap:.3g} über eps {eps:g}: nicht konvergiert"
    dual = float(0.5 * (alpha.sum() - alpha @ G))
    spur = {"step": np.array(tr_step), "dual": np.array(tr_dual), "gap": np.array(tr_gap)} if trace else None
    return SmoResult(alpha, bias, steps, float(gap), converged, stopped_by, message, dual, spur)
