"""Texte der App: Einleitung, Erklärungen, Annahmen, Formeln. Zahlen aus den eingefrorenen Studien kommen über `wmr_studies` aus `wmr_frozen.json`; die Zahl-gegen-Test-Tabelle der
README und `tests/test_claims.py` halten sie fest. Die Annahmentabelle steht wortgleich in der README (ein Test vergleicht beide Fassungen)."""

import wmr_evaluation as ev
import wmr_studies as St
from wmr_evaluation import fmt, fmt_c, fmt_g

INTRO = """
Bei der **weichen Marge** dürfen Punkte die Marge verletzen, und der Regler **C** legt den Preis je Verletzung fest. Diese Demo zeigt, **was C kauft**, warum C kein Genauigkeitsregler
ist, und **warum der Löser schweigt**, wenn C nicht passt: er löst das gestellte Problem bis auf die Toleranz, nur ist es das falsche. Alle Punkte sind **ausgedachte Modelle** (zwei überlappende
Gauß-Klassen, zwei Ringe um ein Depot, fünf Punkte auf einer Geraden), keine Messdaten. Die **Marge heißt hier 2/‖w‖**, der Abstand der beiden Ebenen f = ±1; viele Lehrtexte meinen 1/‖w‖.

Weiter unten: fünf Experimente, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

HOW = """
1. **Hart oder weich.** Hart verlangt für jeden Punkt y·f(x) ≥ 1; das gibt es nur bei trennbaren Daten. Weich erlaubt Punkten, die Marge zu verletzen: der **Schlupf** ξ = max(0, 1 − y·f(x)) kostet C·ξ.
2. **Drei Gruppen von Punkten** im dualen Problem (je Punkt ein Multiplikator α mit 0 ≤ α ≤ C): α = 0 (außerhalb der Marge, ohne Einfluss), 0 < α < C (genau **auf** der Marge, y·f = 1) und α = C
   (in der Marge oder jenseits, y·f ≤ 1). **Stützvektor** heißt hier α > 0 und **schließt die Verletzer ein**.
3. **C ist ein Preis.** Ein kleines C macht Verletzungen billig (bei sehr kleinem C liegen viele Multiplikatoren an der Schranke C), ein großes teuer. Was das kauft, hängt vom Kern ab: Ob ein größeres C hilft oder schadet, zeigt Experiment 1.
4. **Der Löser** (SMO, Platt 1998, mit der Arbeitsmenge zweiter Ordnung nach Fan/Chen/Lin 2005) ändert je Schritt zwei Multiplikatoren und hält, sobald die **KKT-Lücke** höchstens eps ist oder die
   Schrittgrenze erreicht ist. Die KKT-Lücke misst, wie genau die Bedingungen des gestellten Problems erfüllt sind, nicht, ob das gestellte Problem das passende ist.
"""

KERNFRAGEN = (
    "1. Was kauft C? (Experiment 1)",
    "2. Warnt der Löser, wenn C falsch ist? (Experiment 2)",
    "3. Wie wählt man C, ohne sich zu täuschen? (Experiment 3)",
    "4. Wie viele Punkte kann das Verfahren falsch einordnen, wenn man jeden einmal herausnimmt (Leave-one-out)? (Experiment 4)",
    "5. Wie viele Schritte braucht der Löser? (Experiment 5)",
)


def _g(d, key, group):
    return St.case_group(d, key, group)


def loeser_studie_md() -> str:
    """Die Zählung „Löser meldet nichts“ aus der eingefrorenen Studie (beide Zufalls-Seed-Gruppen), als Satz für Experiment 2."""
    d = ev.studies()
    if not d:
        return "(Studie nicht gefunden)"
    t = [St.solver_totals(d, g) for g in (1, 2)]
    w = St.warn_harte(d)
    wc = [", ".join(f"{k} bei C = {fmt_c(c)} ({v} von 30 Zufalls-Seeds)" for k, c, v in St.warn_cap_cells(d, g)) for g in (1, 2)]
    return (
        f"**Studie (eingefroren, libsvm `SVC`, tol = 1e-3):** je Zufalls-Seed-Gruppe {t[0]['cells']} Zellen (6 Fälle × 30 Zufalls-Seeds × 7 C, n = 60). Ohne Iterationsgrenze (`max_iter` = −1) "
        f"gibt es {t[0]['warn_unbegrenzt']} und {t[1]['warn_unbegrenzt']} Warnungen; das ist **strukturell** (scikit-learn warnt bei `SVC` nur beim Erreichen einer endlichen Iterationsgrenze) und keine Messung. "
        f"Die größte KKT-Lücke ist {fmt_g(t[0]['gap_max'], 3)} und {fmt_g(t[1]['gap_max'], 3)} (Toleranz 1e-3; über der Toleranz in {t[0]['gap_over_tol']} und {t[1]['gap_over_tol']} Zellen, libsvm schrumpft). "
        f"In {t[0]['bad']} und {t[1]['bad']} Zellen liegt der Testfehler mehr als 0,05 über dem des besten C desselben Zufalls-Seeds. "
        f"Mit der Grenze 100 000 Schritte warnt der Löser in {t[0]['warn_cap']} und {t[1]['warn_cap']} Zellen ({wc[0]}; Gruppe 2: {wc[1]}). "
        f"Gegenprobe mit „harter Marge“ per riesigem C (Gauß-Klassen mit 10 % vertauschten Etiketten, n = 60, Gauß-Kern σ = 1, Grenze 100 000, sechs Sätze zu 30 Zufalls-Seeds): "
        f"Warnung bei C = 1e2 / 1e4 / 1e6 / 1e8 in {' / '.join(str(v) for v in w['gepoolt'])} von {w['von']} Läufen; die Falle selbst führt die Demo `harte-marge-demo` (Experiment 4)."
    )


def live_vs_studie() -> str:
    d = ev.studies()
    if not d:
        return ""
    v = St.live_vs_studie(d)
    return (
        "**Live-Modell ≠ Studie.** Die App rechnet mit einem eigenen SMO (zwei Multiplikatoren je Schritt, Arbeitsmenge zweiter Ordnung) und mit einer Schrittgrenze; die eingefrorenen Tabellen zu Testfehler, "
        "Warnungen und `n_iter_` stammen von libsvm (`SVC`, in der App nicht enthalten, nur im Messwerkzeug). Beide geben denselben Dualwert (größte relative Abweichung "
        f"{fmt_g(v['dual_rel_max'], 2)} auf {v['instanzen']} Instanzen, n = {v['n']}), aber nicht dieselbe Schrittzahl: gleich in {v['gleich']} von {v['instanzen']} Instanzen, Verhältnis eigene Schritte zu "
        f"`n_iter_` von {fmt(v['ratio_min'], 2)} bis {fmt(v['ratio_max'], 2)} (Median {fmt(v['ratio_median'], 3)}). Die Zähleinheit der Schritte gilt für das jeweilige Programm."
    )


def assumptions_md() -> str:
    d = ev.studies()
    e1, e5 = St.loo_e1(d), St.loo_e5(d)
    w60 = [St.wahl(d, "wahl60", g, "flip0") for g in (1, 2)]
    return f"""
| Annahme | Was passiert, wenn sie verletzt ist | Gemessen |
|---|---|---|
| **Ausgedachte Erzeuger** (Gauß-Klassen, Ringe mit Streuung, fünf Punkte) | Alle Zahlen gelten für diese Erzeuger; es gibt keine reale Datenquelle, und ein Nutzen in Logistik, Lager oder Fertigung ist in den gelesenen Quellen nicht gemessen. | nicht Gegenstand |
| **n = 60 Trainingspunkte** in den C-Messungen (live bis 200) | „Große C schaden beim Gauß-Kern“ gilt für n = 60 und σ = 1; bei Steinwart (2001) muss C mit n wachsen, damit das Verfahren konsistent bleibt. | C gegen n nicht gemessen |
| **Ein Kern je Lauf** | Linear oder Gauß mit fester Bandbreite σ; ob eine Funktion ein Kern ist, wird hier nicht geprüft (das behandelt `kernwahl-demo`). | – |
| **SMO dieser Demo** | Die Schrittzahlen gelten für dieses SMO (zweite Ordnung, ohne Schrumpfen); libsvm zählt eigene `n_iter_`, Schrittzahl gleich nur in {St.live_vs_studie(d)['gleich']} von {St.live_vs_studie(d)['instanzen']} Instanzen. | Windows, eine Python-Version |
| **Wahl von C per 5-facher Kreuzvalidierung** | Gemessen ist diese Regel auf dem Raster 6 C × 6 σ und diesen Ringen (n = 60, ohne Rauschen: Differenz CV minus Voreinstellung {fmt(w60[0]['diff_mean'], 4)} und {fmt(w60[1]['diff_mean'], 4)}); nicht verschachtelt, nicht für andere Verfahren. | nur diese Regel |
| **Leave-one-out ≤ Stützvektoren** | Die Zählung gilt für das b des Lösers ({e1['n']} + {e5['n']} Instanzen, 0 Verletzungen); mit dem ungünstigsten b des optimalen Intervalls verletzen {e5['verl_ungunstig']} von {e5['n']} kleinen Instanzen sie. Eigene Herleitung, kein Satz der Quelle. | Instanzen |
| **Der Löser meldet nichts** | Bei unbegrenzter Schrittzahl kann libsvm nicht warnen; mit endlicher Grenze warnt es nur an der Grenze. Die KKT-Lücke bleibt klein, falsch ist das gestellte C. | Zufalls-Seeds, Zellen |

**Nicht angeboten, nicht gebaut, nicht gemessen** (kein Regler, kein Versprechen): ν-SVM, `class_weight` (C je Klasse) und Platt-Skalierung; Stützvektor-Regression; C gegen n, verschachtelte Kreuzvalidierung,
Validierungsmenge bei vertauschten Etiketten; Arbeitsmengenwahl gegen C; ein Fremd-Innere-Punkte-Löser mit der Einheit Newton-Schritt; Reparatur unzulässiger Kerne (`kernwahl-demo`), Kriging (`kernel-ridge-demo`), Landmarken (`gram-naeherung-demo`),
Baumverfahren (`kernel-oder-baum-demo`); Nutzen im Betrieb.
"""


FORMULAS = r"""
**Lesehilfe.** Gegeben sind Punkte $x_i$ mit Etiketten $y_i \in \{-1, +1\}$ ($i = 1, \dots, n$, beide Klassen kommen vor), ein Kern $K$ (Skalarprodukt im Merkmalsraum) und die Strafe $C > 0$ je Verletzung.
Gesucht ist die Funktion $f$ mit kleinstem Preis aus Länge von $w$ und Schlupf.

| Symbol | Bedeutung |
|---|---|
| $\varphi(x)$ | Abbildung in den Merkmalsraum, $K(x, z) = \varphi(x) \cdot \varphi(z)$ (linear: $\varphi(x) = x$) |
| $w, b$ | Normalenvektor und Verschiebung; $f(x) = w \cdot \varphi(x) + b$ |
| $\xi_i$ | Schlupf des Punkts $i$: $\xi_i = \max(0,\, 1 - y_i f(x_i))$ |
| $C$ | Strafe je Verletzung (Preis), nie $\lambda$ (Ridge, `kernel-ridge-demo`) und nie $\nu$ |
| $\alpha_i$ | Multiplikator (Dualvariable) des Punkts $i$, $0 \le \alpha_i \le C$; $\alpha_i > 0$ heißt Stützvektor (Verletzer eingeschlossen) |
| $\sigma$, $\gamma$ | Bandbreite des Gauß-Kerns $K(x, z) = \exp(-\lVert x - z\rVert^2 / (2\sigma^2))$; $\gamma = 1/(2\sigma^2)$ nur zur Umrechnung (libsvm, scikit-learn) |

**Primal (weiche Marge):**
$$\min_{w,\,b,\,\xi}\ \tfrac12 \lVert w\rVert^2 + C \sum_{i=1}^{n} \xi_i \qquad \text{unter} \qquad y_i\,(w\cdot\varphi(x_i) + b) \ge 1 - \xi_i,\quad \xi_i \ge 0 \quad (i = 1,\dots,n).$$

**Dual** (Platt 1998, Cortes/Vapnik 1995; hier gilt die Schranke $\alpha_i \le C$):
$$\max_{\alpha}\ \sum_{i=1}^{n} \alpha_i - \tfrac12 \sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j\, y_i y_j\, K(x_i, x_j) \qquad \text{unter} \qquad 0 \le \alpha_i \le C \quad (i = 1,\dots,n),\qquad \sum_{i=1}^{n} y_i \alpha_i = 0 .$$

**Entscheidungsfunktion und Marge:**
$$f(x) = \sum_{j=1}^{n} \alpha_j\, y_j\, K(x_j, x) + b, \qquad \lVert w\rVert^2 = \sum_{i,j} \alpha_i \alpha_j\, y_i y_j\, K(x_i, x_j), \qquad \text{Marge} = \frac{2}{\lVert w\rVert}.$$
(Platt 1998 schreibt $u = \sum_j y_j\alpha_j K(x_j, x) - b$: das Vorzeichen von $b$ ist dort verschieden.)

**KKT-Bedingungen** am Optimum: $\alpha_i = 0 \Rightarrow y_i f(x_i) \ge 1$; $0 < \alpha_i < C \Rightarrow y_i f(x_i) = 1$; $\alpha_i = C \Rightarrow y_i f(x_i) \le 1$.
Gibt es einen Punkt mit $0 < \alpha_i < C$, ist $b = y_i - \sum_j \alpha_j y_j K(x_j, x_i)$; sonst bestimmen die Ungleichungen nur ein Intervall für $b$ (die App zeigt dann „b nicht eindeutig“).
Primalwert und Dualwert sind am Optimum gleich: $\tfrac12\lVert w\rVert^2 + C\sum_i \xi_i = \sum_i \alpha_i - \tfrac12\lVert w\rVert^2$.

**KKT-Lücke** des Lösers: mit $G = Q\alpha - e$ ($Q_{ij} = y_i y_j K_{ij}$, $e$ der Einsvektor), $I_{\text{auf}} = \{i : (y_i = +1,\ \alpha_i < C) \text{ oder } (y_i = -1,\ \alpha_i > 0)\}$ und
$I_{\text{ab}} = \{i : (y_i = -1,\ \alpha_i < C) \text{ oder } (y_i = +1,\ \alpha_i > 0)\}$ ist die Lücke $m - M$ mit $m = \max_{i \in I_{\text{auf}}} (-y_i G_i)$ und $M = \min_{i \in I_{\text{ab}}} (-y_i G_i)$;
das SMO hält bei $m - M \le \varepsilon$ (eps).

**Handbeispiel** (fünf Punkte auf der Geraden, $x = (0,1,2,3,4)$, $y = (-1,-1,+1,-1,+1)$, linearer Kern, nicht trennbar): für $C \ge 1/3$ ist $\alpha = (0, \tfrac{3C+2}{9}, C, C, \tfrac{3C+2}{9})$,
$w = 2/3$, $b = -5/3$ und der Dualwert $\tfrac29 + \tfrac{8C}{3}$; für $C \le 1/3$ liegen alle vier Stützvektoren bei $\alpha = C$, $w = 2C$, der Dualwert ist $4C - 2C^2$ und $b$ liegt nur
im Intervall $[-1-2C,\ \min(-1,\ 1-8C)]$.

**Konventionen der Reihe.** Marge $= 2/\lVert w\rVert$ (nicht $1/\lVert w\rVert$). **Zähleinheiten:** Zufalls-Seeds (Mittel über feste Datensätze), Instanzen (Leave-one-out), Aktualisierungsschritte
(SMO: ein Schritt ändert zwei Multiplikatoren), Newton-Schritte (Barriere), Zellen (Zufalls-Seed × C). Keine Sekunden als Befund.
"""
