"""Testhilfen mit anderem Rechenweg als die App (aus `handbeispiele_03.py` der Planung): SMO in Brüchen (erste Ordnung, zwei Implementierungen derselben Regel) und das Intervall des
optimalen b durch Minimierung des Primalwerts bei festem w an den Knickstellen. Nur Standardbibliothek; die App nutzt davon nichts."""

from fractions import Fraction as F

HX = [F(0), F(1), F(2), F(3), F(4)]
HY = [-1, -1, 1, -1, 1]


def gram(xs):
    return [[a * b for b in xs] for a in xs]


def dual_wert(alpha, xs, ys):
    K = gram(xs)
    n = len(xs)
    return sum(alpha) - F(1, 2) * sum(alpha[i] * alpha[j] * ys[i] * ys[j] * K[i][j] for i in range(n) for j in range(n))


def primal_wert(alpha, b, xs, ys, C):
    w = sum(alpha[i] * ys[i] * xs[i] for i in range(len(xs)))
    xi = [max(F(0), 1 - ys[j] * (w * xs[j] + b)) for j in range(len(xs))]
    return F(1, 2) * w * w + C * sum(xi), w, xi


def b_intervall_primal(alpha, xs, ys, C):
    """Bei festem w ist der Primalwert in b stückweise linear und konvex; das Minimum liegt an den Knickstellen b_j = y_j − w x_j. Das Intervall ist die Hülle der Knickstellen mit minimalem Wert."""
    w = sum(alpha[i] * ys[i] * xs[i] for i in range(len(xs)))
    cand = sorted({ys[j] - w * xs[j] for j in range(len(xs))})
    werte = {bb: primal_wert(alpha, bb, xs, ys, C)[0] for bb in cand}
    pmin = min(werte.values())
    arg = [bb for bb in cand if werte[bb] == pmin]
    return min(arg), max(arg), pmin


def smo_exakt(xs, ys, C, max_steps=20000):
    """SMO mit maximal verletzendem Paar (erste Ordnung, bei Gleichstand kleinster Index), exakt in Brüchen; prüft in jedem Schritt Σ yα = 0, 0 ≤ α ≤ C und steigenden Dualwert."""
    n = len(xs)
    K = gram(xs)
    alpha = [F(0)] * n
    G = [F(-1)] * n
    schritte = []
    last = F(0)
    for _ in range(max_steps):
        score = [-ys[i] * G[i] for i in range(n)]
        up = [i for i in range(n) if (ys[i] > 0 and alpha[i] < C) or (ys[i] < 0 and alpha[i] > 0)]
        low = [i for i in range(n) if (ys[i] < 0 and alpha[i] < C) or (ys[i] > 0 and alpha[i] > 0)]
        i = max(up, key=lambda t: (score[t], -t))
        j = min(low, key=lambda t: (score[t], t))
        if score[i] - score[j] <= 0:
            break
        a = K[i][i] + K[j][j] - 2 * K[i][j]
        t = min((score[i] - score[j]) / a, min((C - alpha[i]) if ys[i] > 0 else alpha[i], alpha[j] if ys[j] > 0 else (C - alpha[j])))
        di, dj = ys[i] * t, -ys[j] * t
        alpha[i] += di
        alpha[j] += dj
        for r in range(n):
            G[r] += ys[r] * ys[i] * K[r][i] * di + ys[r] * ys[j] * K[r][j] * dj
        wert = dual_wert(alpha, xs, ys)
        assert sum(ys[p] * alpha[p] for p in range(n)) == 0 and all(0 <= a_ <= C for a_ in alpha) and wert >= last
        last = wert
        schritte.append((i, j, tuple(alpha)))
    else:
        raise AssertionError("SMO nicht endlich")
    return alpha, schritte


def smo_exakt_paare(xs, ys, C, max_steps=20000):
    """Zweite Implementierung derselben Regel: das Paar wird durch Aufzählung ALLER Paare gewählt, der Gradient je Schritt aus α neu gerechnet (Gegenweg für die Umsetzung)."""
    n = len(xs)
    K = gram(xs)
    alpha = [F(0)] * n
    folge = []
    for _ in range(max_steps):
        G = [sum(ys[r] * ys[c] * K[r][c] * alpha[c] for c in range(n)) - 1 for r in range(n)]
        score = [-ys[r] * G[r] for r in range(n)]
        up = [i for i in range(n) if (ys[i] > 0 and alpha[i] < C) or (ys[i] < 0 and alpha[i] > 0)]
        low = [j for j in range(n) if (ys[j] < 0 and alpha[j] < C) or (ys[j] > 0 and alpha[j] > 0)]
        i, j = max([(i, j) for i in up for j in low], key=lambda pq: (score[pq[0]] - score[pq[1]], -pq[0], -pq[1]))
        if score[i] - score[j] <= 0:
            break
        a = K[i][i] + K[j][j] - 2 * K[i][j]
        t = min((score[i] - score[j]) / a, min((C - alpha[i]) if ys[i] > 0 else alpha[i], alpha[j] if ys[j] > 0 else (C - alpha[j])))
        alpha[i] += ys[i] * t
        alpha[j] -= ys[j] * t
        folge.append((i, j))
    return alpha, folge
