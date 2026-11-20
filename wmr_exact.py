"""Das duale Problem in Brüchen durch Aufzählung (nur für das Handbeispiel und kleine Mengen; kein Optimierungsverfahren).

Für n Punkte gibt es 3^n Zuordnungen „α = 0 / frei / α = C“. Je Zuordnung ist die KKT-Bedingung ein lineares Gleichungssystem (die freien Punkte liegen auf der Marge,
y f = 1, dazu Σ yα = 0); es wird exakt in Brüchen gelöst und auf 0 ≤ α ≤ C und die Ungleichungen der anderen Punkte geprüft. Findet sich keine freie Menge, bestimmt nur Σ yα = 0,
und b ist ein Intervall. Gleiche α aus mehreren Zuordnungen (α = C zählt auch als frei) werden zusammengefasst, b als Vereinigung der Intervalle.

`knick_tabelle` rechnet daraus die Tabelle des Handbeispiels: w, b (oder sein Intervall), Dualwert und Summe der Schlupfwerte je C.
Einheit der Zählung: Zuordnungen (3^n; n = 5 gibt 243)."""

import itertools
from fractions import Fraction as F


def gram_linear(xs):
    return [[a * b for b in xs] for a in xs]


def _solve(A, b):
    """Gauß-Elimination in Brüchen; None bei Singularität."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(n + 1)]
    return [M[i][n] / M[i][i] for i in range(n)]


def kkt_aufzaehlung(xs, ys, C):
    """Alle KKT-Lösungen des weichen Duals (linearer Kern in 1D, xs Brüche). Gibt eine Liste (alpha, b_lo, b_hi) zurück; b_lo = b_hi bei einem freien Stützvektor."""
    n = len(xs)
    K = gram_linear(xs)
    found = []
    for zu in itertools.product((0, 1, 2), repeat=n):  # 0: α = 0, 1: frei, 2: α = C
        free = [i for i in range(n) if zu[i] == 1]
        alpha = [F(0)] * n
        for i in range(n):
            if zu[i] == 2:
                alpha[i] = C
        k = len(free)
        if k > 0:
            A = [[ys[j] * ys[i] * K[j][i] for i in free] + [F(ys[j])] for j in free]
            A.append([F(ys[i]) for i in free] + [F(0)])
            rhs = [1 - sum(ys[j] * ys[i] * K[j][i] * alpha[i] for i in range(n) if zu[i] == 2) for j in free]
            rhs.append(-sum(ys[i] * alpha[i] for i in range(n) if zu[i] == 2))
            sol = _solve(A, rhs)
            if sol is None:
                continue
            if not all(0 <= sol[pos] <= C for pos in range(k)):
                continue
            for pos, i in enumerate(free):
                alpha[i] = sol[pos]
            b = sol[k]
            g = [sum(alpha[i] * ys[i] * K[j][i] for i in range(n)) for j in range(n)]
            m = [ys[j] * (g[j] + b) for j in range(n)]
            if all((zu[j] != 0 or m[j] >= 1) and (zu[j] != 2 or m[j] <= 1) for j in range(n)):
                found.append((tuple(alpha), b, b))
        else:
            if sum(ys[i] * alpha[i] for i in range(n)) != 0:
                continue
            g = [sum(alpha[i] * ys[i] * K[j][i] for i in range(n)) for j in range(n)]
            lo, hi = None, None  # y_j (g_j + b) ≥ 1 bei α = 0, ≤ 1 bei α = C
            for j in range(n):
                ge = zu[j] == 0
                t = ys[j] - g[j]
                if (ge and ys[j] > 0) or (not ge and ys[j] < 0):
                    lo = t if lo is None else max(lo, t)
                else:
                    hi = t if hi is None else min(hi, t)
            if lo is None or hi is None or lo <= hi:
                found.append((tuple(alpha), lo, hi))
    merged = {}
    for al, lo, hi in found:
        if al in merged:
            lo0, hi0 = merged[al]
            merged[al] = (min(x for x in (lo0, lo) if x is not None), max(x for x in (hi0, hi) if x is not None))
        else:
            merged[al] = (lo, hi)
    return [(al, lo, hi) for al, (lo, hi) in merged.items()]


def dual_wert(alpha, xs, ys):
    K = gram_linear(xs)
    n = len(xs)
    return sum(alpha) - F(1, 2) * sum(alpha[i] * alpha[j] * ys[i] * ys[j] * K[i][j] for i in range(n) for j in range(n))


def primal_wert(alpha, b, xs, ys, C):
    """Primalwert 1/2 w² + C Σ ξ aus w = Σ α y x (1D) und b; gibt (Wert, w, ξ) zurück."""
    w = sum(alpha[i] * ys[i] * xs[i] for i in range(len(xs)))
    xi = [max(F(0), 1 - ys[j] * (w * xs[j] + b)) for j in range(len(xs))]
    return F(1, 2) * w * w + C * sum(xi), w, xi


HAND_X = [F(0), F(1), F(2), F(3), F(4)]
HAND_Y = [-1, -1, 1, -1, 1]


def knick_zeile(C: F) -> dict:
    """Die Lösung des Handbeispiels für ein C aus der Aufzählung: α, w, b-Intervall, Dualwert, Summe der Schlupfwerte, Primalwert an beiden Enden des b-Intervalls."""
    loes = kkt_aufzaehlung(HAND_X, HAND_Y, C)
    if len(loes) != 1:
        raise AssertionError(f"C = {C}: {len(loes)} Lösungen statt einer")
    alpha, blo, bhi = loes[0]
    D = dual_wert(alpha, HAND_X, HAND_Y)
    P_lo, w, xi_lo = primal_wert(alpha, blo, HAND_X, HAND_Y, C)
    P_hi, _, _ = primal_wert(alpha, bhi, HAND_X, HAND_Y, C)
    return {"C": C, "alpha": alpha, "w": w, "b_lo": blo, "b_hi": bhi, "dual": D, "primal": P_lo, "primal_hi": P_hi, "xi_sum": sum(xi_lo), "n_sv": sum(1 for a in alpha if a > 0)}


def knick_tabelle(cs) -> list:
    return [knick_zeile(F(c)) for c in cs]


def bruch(x: F) -> str:
    """Bruch als Text: 3, 5/9, −5/3 (echtes Minuszeichen)."""
    s = str(x) if x.denominator != 1 else str(x.numerator)
    return s.replace("-", "−")


def loo_exakt(xs, ys, C, einklassig_fehler: bool = True) -> dict:
    """Leave-one-out in Brüchen durch Neuaufzählung ohne jeden Punkt: Fehler mit ungünstigstem b des Intervalls (y f ≤ 0 zählt als Fehler) und mit b in der Intervallmitte,
    dazu die Zahl der Stützvektoren der vollen Lösung (α > 0). Ein Rest mit nur einer Klasse hat keine trennende Funktion: er zählt als Fehler (einklassig_fehler),
    sonst wird dieser Punkt nicht gezählt. Gibt auch die Zahl der einklassigen Reste zurück."""
    full = kkt_aufzaehlung(xs, ys, C)
    if len(full) != 1:
        raise AssertionError("volle Aufgabe: keine eindeutige KKT-Lösung")
    n_sv = sum(1 for a in full[0][0] if a > 0)
    worst = mid = one = 0
    for i in range(len(xs)):
        xr = [xs[k] for k in range(len(xs)) if k != i]
        yr = [ys[k] for k in range(len(xs)) if k != i]
        if len(set(yr)) < 2:
            one += 1
            if einklassig_fehler:
                worst += 1
                mid += 1
            continue
        loes = kkt_aufzaehlung(xr, yr, C)
        if not loes:
            raise AssertionError(f"ohne Punkt {i}: keine KKT-Lösung")
        al, lo, hi = loes[0]  # bei mehreren α-Lösungen (α nicht eindeutig) ist w dasselbe; die erste genügt
        w = sum(al[k] * yr[k] * xr[k] for k in range(len(xr)))
        if any(sum(a2[k] * yr[k] * xr[k] for k in range(len(xr))) != w for a2, _, _ in loes):
            raise AssertionError(f"ohne Punkt {i}: w nicht eindeutig")
        worst += int(any(ys[i] * (w * xs[i] + b) <= 0 for b in (lo, hi)))
        mid += int(ys[i] * (w * xs[i] + (lo + hi) / 2) <= 0)
    return {"n": len(xs), "n_sv": n_sv, "worst": worst, "mitte": mid, "einklassig": one}


# Die beiden Gegenbeispiele zur Schranke „LOO-Fehler ≤ Zahl der Stützvektoren“ bei nicht eindeutigem b (Handbeispiel Teil 4b und 4c)
GEGENBEISPIELE = (
    {"name": "drei Punkte", "x": [F(-1), F(1), F(3)], "y": [-1, 1, 1], "C": F(1, 10)},
    {"name": "fünf Punkte", "x": [F(-3), F(-2), F(-1), F(0), F(1)], "y": [-1, -1, -1, 1, 1], "C": F(1, 20)},
)
