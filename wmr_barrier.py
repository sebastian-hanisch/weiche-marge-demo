"""Innere-Punkte-Verfahren (Log-Barriere mit Newton-Schritten) für das duale QP der weichen Marge, nur zum Zählen der Newton-Schritte (Experiment 5).

Das Problem (wie in `wmr_smo.py`, Q_ij = y_i y_j K_ij):  min 1/2 a'Qa - e'a  unter  0 <= a_i <= C  und  y'a = 0.
Je äußerem Schritt wird  t (1/2 a'Qa - e'a) - Σ log a_i - Σ log (C - a_i)  durch Newton-Schritte minimiert (Gleichung y'a = 0 im KKT-System jedes Schritts);
danach wächst t um den Faktor mu. Die Dualitätslücke der Barrierefolge ist 2n/t.

Einstellungen (alle Zahlen der Zählung gelten nur mit ihnen, siehe `BarrierSettings`): Startpunkt strikt innen mit y'a = 0, Startwert t0, Faktor mu, Abbruch
bei Lücke 2n/t <= gap_tol, Newton-Abbruch bei Dekrement^2/2 <= newton_tol, Liniensuche: Schritt zuerst auf das `frac`-fache des Wegs bis zum Rand von [0, C] begrenzt,
dann Halbierung bis zur Armijo-Bedingung mit Faktor `armijo`.

Quelle des Verfahrens: Boyd/Vandenberghe, Convex Optimization, Kap. 11 (Barrieremethode); die Einstellungen der Vormessung (`kn_v1_lib.barrier_qp`) sind hier nachgebaut, nicht kopiert."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class BarrierSettings:
    mu: float = 10.0
    t0: float = 1.0
    gap_tol: float = 1e-6
    newton_tol: float = 1e-10
    frac: float = 0.99
    armijo: float = 0.25
    max_newton: int = 500


class BarrierResult(NamedTuple):
    alpha: np.ndarray
    newton: int  # Newton-Schritte insgesamt
    outer: int  # äußere Schritte (Werte von t)
    gap: float  # Dualitätslücke 2n/t am Ende
    converged: bool


def start_point(y: np.ndarray, C: float) -> np.ndarray:
    """Strikt innen liegender Startpunkt mit y'a = 0: die kleinere Klasse bekommt C/2, die größere C/2 mal das Verhältnis der Klassengrößen (ausgewogen: überall C/2)."""
    y = np.asarray(y, float)
    npos, nneg = int((y > 0).sum()), int((y < 0).sum())
    if npos == 0 or nneg == 0:
        raise ValueError("beide Klassen nötig")
    a = np.empty(len(y))
    a[y > 0] = 0.5 * C * min(1.0, nneg / npos)
    a[y < 0] = 0.5 * C * min(1.0, npos / nneg)
    return a


def barrier(K, y, C: float, settings: BarrierSettings = BarrierSettings()) -> BarrierResult:
    """Löst das duale QP mit der Log-Barriere; C muss endlich sein."""
    if not np.isfinite(C):
        raise ValueError("die Barriere braucht endliches C")
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    n = len(y)
    Q = (y[:, None] * y[None, :]) * K
    a = start_point(y, C)
    t = settings.t0
    newton = 0
    outer = 0
    kkt = np.zeros((n + 1, n + 1))
    kkt[:n, n] = y
    kkt[n, :n] = y
    converged = False
    while True:
        outer += 1
        for _ in range(100):
            g = t * (Q @ a - 1.0) - 1.0 / a + 1.0 / (C - a)
            kkt[:n, :n] = t * Q + np.diag(1.0 / a ** 2 + 1.0 / (C - a) ** 2)
            sol = np.linalg.solve(kkt, np.r_[-g, 0.0])
            d = sol[:n]
            dec = float(-g @ d)
            newton += 1
            if dec / 2.0 <= settings.newton_tol:
                break
            s = 1.0
            neg = d < 0
            if neg.any():
                s = min(s, settings.frac * float(np.min(-a[neg] / d[neg])))
            pos = d > 0
            if pos.any():
                s = min(s, settings.frac * float(np.min((C - a[pos]) / d[pos])))
            f0 = _phi(a, Q, t, C)
            while _phi(a + s * d, Q, t, C) > f0 - settings.armijo * s * dec and s > 1e-14:
                s *= 0.5
            a = a + s * d
            if newton > settings.max_newton:
                break
        if 2.0 * n / t <= settings.gap_tol:
            converged = True
            break
        if newton > settings.max_newton:
            break
        t *= settings.mu
    return BarrierResult(a, newton, outer, 2.0 * n / t, converged)


def _phi(z, Q, t, C):
    return t * (0.5 * z @ Q @ z - z.sum()) - np.log(z).sum() - np.log(C - z).sum()
