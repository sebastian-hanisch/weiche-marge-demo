# Weiche Marge: was C kauft und warum der Löser schweigt – Streamlit-Demo

Diese Demo ist die dritte der Reihe **kernmethoden** („Skalarprodukte tauschen“) in der „Konzepte“-Reihe der Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie hebt die Annahme der
Demo zur harten Marge (`harte-marge-demo`) auf: **Die Daten sind trennbar, jeder Punkt liegt auf der richtigen Seite der Marge.** Hier dürfen Punkte die Marge verletzen (Schlupf ξ), und der Regler **C** legt den
Preis je Verletzung fest. Die neue Annahme, die eine spätere Demo der Reihe (`kernwahl-demo`) aufhebt, lautet: *der Kern ist gesetzt*; hier wird nur C gedreht (und die Bandbreite σ des Gauß-Kerns dort, wo sie C
erst verständlich macht). Die Punkte sind **ausgedachte Modelle** (keine Messdaten); gezählt werden Zufalls-Seeds, Zellen, Instanzen, Aktualisierungsschritte und Newton-Schritte. Die **Marge heißt hier 2/‖w‖**,
der Abstand der beiden Ebenen f = ±1 (viele Lehrtexte meinen 1/‖w‖).

## Kernfrage

C ist kein Genauigkeitsregler, sondern ein **Preis**: je Verletzung der Marge zahlt die Lösung C·ξ, und was C kauft, hängt am Kern. Fünf Experimente beantworten live (eigenes SMO) und mit eingefrorenen Studien:
(1) Was kauft C? (2) Warnt der Löser, wenn C falsch ist? (3) Wie wählt man C, ohne sich zu täuschen? (4) Wie viele Punkte kann das Verfahren falsch einordnen, wenn man jeden herausnimmt (Leave-one-out)?
(5) Wie viele Schritte braucht der Löser? Der Hook der Reihe („Der Löser meldet nichts“) hat in dieser Demo eine Hauptgestalt: **richtig gelöstes, falsch gestelltes C** (der Löser löst das gestellte Problem bis auf die Toleranz, das Modell ist trotzdem schlecht;
er spricht nur an der Schrittgrenze).

## Befunde und Korrekturen gegenüber der Vormessung

Die Vormessung der Planung lief mit `numpy.random.default_rng`; die App zieht aus SplitMix64 (`wmr_rng.py`). Darum sind alle Zahlen dieser README **im Bau neu gemessen** (zwei Gruppen zu je 30 Zufalls-Seeds,
`tools/wmr_make_frozen.py` mit scikit-learn nur im Werkzeug) und weichen ab. Die wichtigsten Abweichungen (Voraussetzungen jeweils im Befund unten):

| Vormessung | Neumessung mit den Strömen der App (Gruppe 1 und 2) |
|---|---|
| linear, ohne Rauschen: Spanne des mittleren Testfehlers 0,0096 | 0,0085 und 0,0151; ab C = 0,01 nur 0,0024 und 0,0050 |
| Gauß σ = 1: Testfehler 0,1686 bei C ≤ 0,01, 0,2520 bei C = 1000 | 0,1705 und 0,1711 bei C = 0,001, 0,2611 und 0,2533 bei C = 1000 |
| 840 Zellen (vier Fälle), 12 mit Warnung bei Grenze 100 000, 350 über 0,05 | die vier Fälle: 14 und 12 Warnungen, 320 und 330 Zellen über 0,05 (je von 840); mit den zwei zusätzlichen Fällen ohne Rauschen 1260 Zellen je Gruppe, 17 und 17 Warnungen, 364 und 374 Zellen |
| Kreuzvalidierung gegen Voreinstellung (n = 60): +0,0089 und +0,0054, „nicht besser“ | ohne Rauschen **0,0118 (Standardfehler 0,0041, 2,9 Standardfehler)** und **0,0150 (Standardfehler 0,0045, 3,3 Standardfehler)**: die Kreuzvalidierung ist hier **schlechter**; mit Rauschen kein nachweisbarer Unterschied (0,0034 (Standardfehler 0,0071, 0,5 Standardfehler); 0,0043 (Standardfehler 0,0067, 0,6 Standardfehler)) |
| Kreuzvalidierung gegen Voreinstellung (n = 200): +0,0026, 2,5 Standardfehler | ohne Rauschen 0,0042 (Standardfehler 0,0010, 4,0 Standardfehler; besser in 22, schlechter in 35, gleich in 3); mit Rauschen 0,0040 (Standardfehler 0,0014, 2,8 Standardfehler; besser in 24, schlechter in 35, gleich in 1) |
| Optimismus der Gewinnerauswahl bei m = 30 / 100 / 500: +0,0593 / +0,0155 / +0,0037 und +0,0570 / +0,0204 / +0,0119 | siehe Befunde unten (Gruppe 1 und 2) |
| Leave-one-out mit ungünstigstem b: 26 von 600 verletzen die Schranke | mit dem ungünstigsten b 23 von 600 (ohne einklassigen Rest) |
| Schritte, n = 200, Gauß, C = 0,01 bis 1000: 98 / 109 / 521 / 2062 / 17 294 / 68 127 | 97 / 125 / 470 / 2932 / 22623 / 97409 |
| Steigung der Schritte gegen n: 1,24 [1,17; 1,30] | 1,27 (Bootstrap-90-%-Band [1,23; 1,32]; zwei Gruppen zu 10: 1,24 und 1,31) |
| Newton-Schritte der Barriere n = 20 bis 320: 88 / 96 / 107 / 116 / 126 | 89 / 111 / 106 / 115 / 145; Einzelwerte 85 bis 207; je nach Einstellung bei n = 80 63 bis 398 |
| eigenes SMO gegen libsvm: Schrittzahl gleich in 1 von 60, Verhältnis 0,82 bis 2,19 | gleich in 5 von 60, Verhältnis 0,82 bis 1,32 (Median 1,012) |

Weitere Korrekturen gegenüber der Vormessung: das exakte Gegenbeispiel mit drei Punkten zur Leave-one-out-Schranke zählt den **einklassigen Rest als Fehler** (ohne diese Zählung 2 gegen 2, keine Verletzung); ein
neues exaktes Beispiel mit fünf Punkten braucht sie nicht. Die Newton-Schrittzahlen stehen jetzt mit allen Einstellungen (Startwert von t, Liniensuche) und als Band. Die Spanne beim linearen Kern kommt zum größten Teil
von C = 0,001 (Spanne ab C = 0,01: 0,0024 und 0,0050). Das Handbeispiel läuft in der Vormessung als Skript (484 Prüfungen, rund 2 Minuten); hier sind es zerlegte Tests mit denselben Werten.

## Modell

Trainingsdaten (xᵢ, yᵢ), yᵢ ∈ {−1, +1}; f(x) = Σⱼ αⱼ yⱼ K(xⱼ, x) + b. **Weiche Marge (Primal):** min ½‖w‖² + C Σ ξᵢ unter yᵢ(w·φ(xᵢ) + b) ≥ 1 − ξᵢ, ξᵢ ≥ 0. **Dual:** max Σαᵢ − ½ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ)
unter 0 ≤ αᵢ ≤ C und Σ yᵢαᵢ = 0 (obere Schranke C statt „αᵢ ≥ 0 ohne Grenze“; Platt 1998). **Marge** heißt 2/‖w‖ auch bei weicher Marge. **Stützvektor** heißt αᵢ > 0 und **schließt die Verletzer ein**
(αᵢ = C); drei Gruppen: αᵢ = 0 (außerhalb), 0 < αᵢ < C (**auf** der Marge, yᵢ f(xᵢ) = 1), αᵢ = C (in oder jenseits). **Regler getrennt benennen:** **C** (SVM, Preis je Verletzung, nie λ; die Ridge-Fassung gehört zu
`kernel-ridge-demo`), ν wird nicht verwendet. **Gauß-Kern** exp(−‖x − x′‖²/(2σ²)) mit **Bandbreite σ** (Längenmaß); γ = 1/(2σ²) nur zur Umrechnung (libsvm und scikit-learn sagen `gamma`); **eine** Festlegung: σ ist der Regler.
Vollständiges Modell mit Lesehilfe im 📐-Expander der App. Das Vorzeichen von b: Platt 1998 schreibt u = Σ yⱼαⱼK − b, hier gilt f = Σ αⱼyⱼK + b. Namensfalle: in `ocsvm-demo` ist ν ein Anteil (Ein-Klassen-SVM, andere Aufgabe).

## Methodik

* **Löser** (`wmr_smo.py`, wörtliche Kopie von `hmg_smo.py` aus `harte-marge-demo`, Kopier-Test: Hash der Funktion und Bytegleichheit): SMO nach Platt 1998 mit der Arbeitsmenge zweiter Ordnung nach Fan/Chen/Lin 2005 (wie LIBSVM, ohne
  Schrumpfen und Cache), je Schritt zwei Multiplikatoren, Schrittgrenze mit Meldung, Spur mit Dualwert und KKT-Lücke je Schritt. Die Schrittzahlen gelten für **dieses** SMO; das Original von Platt und „SMO-Typ“ in LIBSVM sind verschieden.
* **Aufzählung in Brüchen** (`wmr_exact.py`): für das Handbeispiel alle 3⁵ = 243 Zuordnungen „α = 0 / frei / α = C“, je Zuordnung ein lineares KKT-System, kein Optimierer; daraus α, w, das b-Intervall, Dualwert und Schlupfsumme je C.
* **Leave-one-out** (`wmr_evaluation.loo`): jeder Punkt wird wirklich herausgenommen und der Rest neu berechnet; gezählt mit dem b des Lösers und mit dem ungünstigsten b des optimalen Intervalls (y f ≤ 0 zählt als Fehler); ein Rest mit nur einer
  Klasse zählt als Fehler und wird ausgewiesen. Die Schranke „LOO-Fehler ≤ Zahl der Stützvektoren“ ist **eigene Herleitung** (bei Cortes/Vapnik 1995 Gl. 5 steht eine Erwartungswert-Aussage für trennbare Daten).
* **Barriere** (`wmr_barrier.py`): Log-Barriere mit Newton-Schritten nur zum Zählen. Einstellungen: Startpunkt strikt innen mit Σ yα = 0, Startwert t₀ = 1, Faktor μ = 10, Abbruch bei Lücke 2n/t ≤ 1e-6, Newton-Abbruch Dekrement²/2 ≤ 1e-10,
  Liniensuche mit dem 0,99-fachen Weg bis zum Rand von [0, C] und Armijo-Faktor 0,25; die Zahl der Newton-Schritte gilt nur mit diesen Einstellungen.
* **Erzeuger** (`wmr_scenario.py`, SplitMix64): „Gauß-Klassen“ (N((±1, 0), I), Abstand 2, Bayes-Fehler Φ(−1) = 0,1587), „Ringe mit Streuung“ (zwei Ringe um ein Depot, Radius 1 und 2, Radiusstreuung 0,35, bei 0 abgeschnitten, Winkel gleichverteilt;
  Bayes-Fehler 0,0766 als Integral, 0,0774 an 200 000 Punkten) und „Fünf Punkte auf der Geraden“. Normalwerte der Trainingsinstanzen aus `normal()` der Vorlage, der Testmengen (2 000 Punkte) aus der Inversen der Verteilungsfunktion.
* **Eingefroren** (`wmr_frozen.json`, erzeugt von `tools/wmr_make_frozen.py`, Stichproben mit `--check` geprüft): C-Wirkung, Warnzählung, Wahl von C und σ, Leave-one-out, Schritte und Barriere. Fremdreferenz im Werkzeug und in den Tests:
  libsvm (`SVC`), scipy (`quad`, SLSQP, `trust-constr`); **die App braucht beides nicht**. Gemessen mit Windows, Python 3.14, numpy 2.5, scipy 1.18, scikit-learn 1.9; ein Linux-Lauf (`demo_linux_check.py`) steht vor dem ersten Push aus.

## Befunde (gemessen, keine Behauptungen)

Zähleinheiten: Zufalls-Seeds (je Gruppe 30, zwei Gruppen), Zellen (Zufalls-Seed × C), Instanzen, Aktualisierungsschritte, Newton-Schritte. Alle C-Messungen: n = 60 Trainingspunkte, 2 000 frische Testpunkte je Zufalls-Seed, Mittel über 30 Zufalls-Seeds; die Zahl
steht für Gruppe 1 und 2. Die Spalte „Beleg“ nennt den Test (Tabelle „Zahl → Test“ unten).

| Frage | Befund | Beleg |
|---|---|---|
| Handbeispiel (fünf Punkte auf der Geraden, nicht trennbar, exakte Brüche) | Genau eine KKT-Lösung für jedes geprüfte C. Für C ≥ 1/3: α = (0, (3C+2)/9, C, C, (3C+2)/9), w = 2/3, b = −5/3, Dualwert 2/9 + 8C/3, Summe der Schlupfwerte 8/3; für C ≤ 1/3: alle vier Stützvektoren bei α = C, w = 2C, Dualwert 4C − 2C², Schlupfsumme 4 − 4C, b nur im Intervall [−1−2C, min(−1, 1−8C)] (die Obergrenze −1 gilt nur bis C = 1/4). Beide Formeln geben bei C = 1/3 den Wert 10/9. Primalwert und Dualwert sind exakt gleich, auch an beiden Enden des b-Intervalls. SMO in Brüchen (erste Ordnung): 3 / 6 / 7 / 11 / 21 / 39 / 935 Schritte für C = 1/8 / 1/4 / 1/2 / 1 / 2 / 4 / 100 | `test_exact.py` |
| C beim linearen Kern, Gauß-Klassen ohne Rauschen | Spanne des mittleren Testfehlers über C = 0,001 bis 1000: 0,0085 und 0,0151; ab C = 0,01 0,0024 und 0,0050; Bayes-Fehler 0,1587. Der Rand C = 0,001 liegt bei 0,1725 und 0,1804, der beste Wert bei 0,1640 und 0,1653 | `test_claims.py` |
| C beim linearen Kern, 10 % vertauschte Etiketten | Bei C = 0,001 kippt der lineare Kern: Testfehler 0,3402 und 0,4063; ab C = 0,1 ist er flach (Spanne 0,0030 und 0,0022, bei C = 1: 0,1712 und 0,1737). Die Aussage „C wirkt kaum“ gilt hier nur für C ≥ 0,1 | `test_claims.py` |
| C beim Gauß-Kern σ = 1, ohne Rauschen | Spanne 0,0907 und 0,0822; Testfehler 0,1705 und 0,1711 bei C = 0,001, 0,1743 und 0,1749 bei C = 1, 0,2611 und 0,2533 bei C = 1000: **bei n = 60 und σ = 1 schaden in diesen Gauß-Klassen große C** (bei Steinwart 2001 muss C mit n wachsen, das ist nicht gemessen) | `test_claims.py` |
| Zu kleines C, 10 % vertauscht, Gauß σ = 1 | Mittlerer Testfehler 0,3396 und 0,4032 bei C = 0,001, 0,1893 und 0,1800 bei C = 1, 0,3155 und 0,3043 bei C = 1000. Das Mittel bei C = 0,001 ist eine **Mischung**: in 15 und 21 von je 30 Zufalls-Seeds beträgt der Testfehler genau 0,5 (libsvm; beim linearen Kern 15 und 21), in den übrigen 0,1525 bis 0,2080 bzw. 0,1570 bis 0,2060. Mechanismus im eigenen SMO (30 Zufalls-Seeds, eigene Messung im Test): von 18 unausgewogenen Zufalls-Seeds (die Vertauschung hat die Klassen ungleich besetzt) 18 davon ordnen alle 2 000 Testpunkte der Mehrheitsklasse zu; in den 12 ausgewogenen Zufalls-Seeds liegt der Testfehler zwischen 0,155 und 0,212 | `test_claims.py` |
| Ringe mit Streuung, Gauß σ = 0,5 | Testfehler 0,1317 und 0,1281 bei C = 0,001 (Bayes-Fehler 0,0766), 0,1078 und 0,1130 bei C = 1, 0,1493 und 0,1576 bei C = 1000: zu kleines C ist hier weniger schädlich als bei vertauschten Etiketten (0,3454 und 0,4167 bei C = 0,001, 0,1331 und 0,1489 bei C = 1) | `test_claims.py` |
| Der Löser meldet nichts | Je Gruppe 1260 Zellen (6 Fälle × 30 Zufalls-Seeds × 7 C). Ohne Iterationsgrenze (`max_iter` = −1) 0 und 0 Warnungen: das ist **strukturell** (scikit-learn warnt bei `SVC` nur beim Erreichen einer endlichen Grenze), keine Messung. Größte KKT-Lücke 0,00162 und 0,00138 bei Toleranz 1e-3 (über der Toleranz in 24 und 27 Zellen: libsvm schrumpft). Mehr als 0,05 Testfehler über dem besten C desselben Zufalls-Seeds in 364 und 374 Zellen. Mit der Grenze 100 000 Schritte warnt der Löser in 17 und 17 Zellen, alle beim linearen Kern und C ≥ 100 | `test_claims.py` |
| Gegenprobe: „harte Marge“ per riesigem C | Gauß-Klassen mit 10 % vertauschten Etiketten, n = 60, Gauß σ = 1, Grenze 100 000, sechs Sätze zu 30 Zufalls-Seeds: Warnung bei C = 1e2 / 1e4 / 1e6 / 1e8 in 0 / 2 / 71 / 81 von 180 Läufen (je Satz 0 bis 0 / 0 bis 1 / 9 bis 17 / 11 bis 18). Die Falle führt `harte-marge-demo` | `test_claims.py` |
| C und Bandbreite zusammen (Ringe, Raster 6 C × 6 σ) | Spanne über C je σ = 0,1 / 0,2 / 0,4 / 0,8 / 1,6 / 3,2: 0,1110, 0,0807, 0,0302, 0,0493, 0,0697, 0,2050 (Gruppe 1), 0,0955, 0,0487, 0,0346, 0,0515, 0,0753, 0,2157 (Gruppe 2). Beste Zelle im Mittel: C = 1, σ = 1,6 (0,0942) und C = 1, σ = 1,6 (0,0952); die Voreinstellung (C = 1, γ = 1/(p·Var X), mittleres σ = 1,14) hat den mittleren Testfehler 0,0972 und 0,0986 und liegt damit 0,0030 und 0,0034 über der besten Zelle. Die gute Lage ist eine Lage im Raster aus C und σ zusammen: bei σ = 0,1 liegt der Fehler bei jedem C bei mindestens 0,2593 (Gruppe 1) bzw. mindestens 0,2623 (Gruppe 2); bei σ = 3,2 bei C = 0,01 und 0,1 zwischen 0,3023 und 0,3023 bzw. zwischen 0,3132 und 0,3132, ab C = 10 höchstens 0,1072 bzw. höchstens 0,1060 | `test_claims.py` |
| Wahl von C: Gewinnerauswahl auf einer Validierungsmenge | Auswahl auf m = 30 Punkten: 0,0567 gegen 0,1125 (Optimismus 0,0559, Standardfehler 0,0094) / 0,0678 gegen 0,1102 (Optimismus 0,0424, Standardfehler 0,0085); m = 100: 0,0727 gegen 0,0998 (Optimismus 0,0272, Standardfehler 0,0065) / 0,0787 gegen 0,1002 (Optimismus 0,0216, Standardfehler 0,0044); m = 500: 0,0856 gegen 0,0911 (Optimismus 0,0055, Standardfehler 0,0028) / 0,0899 gegen 0,0912 (Optimismus 0,0013, Standardfehler 0,0026) (Ringe ohne Rauschen, 36 Zellen, 30 Zufalls-Seeds; gemessen ist die Auswahl auf einer frisch gezogenen Validierungsmenge, **nicht** die Wahl auf einer Testmenge) | `test_claims.py` |
| Wahl von C: Kreuzvalidierung gegen Voreinstellung (n = 60) | Fehler Rückblick (nicht erreichbar) / 5-fache Kreuzvalidierung / Voreinstellung, ohne Rauschen: 0,0882 / 0,1089 / 0,0972 (Gruppe 1) und 0,0884 / 0,1136 / 0,0986 (Gruppe 2); mit 10 % vertauscht: 0,1048 / 0,1249 / 0,1215 und 0,1040 / 0,1321 / 0,1278. Gepaarte Differenz CV minus Voreinstellung ohne Rauschen 0,0118 (Standardfehler 0,0041, 2,9 Standardfehler) und 0,0150 (Standardfehler 0,0045, 3,3 Standardfehler) (CV besser / schlechter in 7 / 22 und 10 / 20 von 30); mit Rauschen 0,0034 (Standardfehler 0,0071, 0,5 Standardfehler) und 0,0043 (Standardfehler 0,0067, 0,6 Standardfehler). Gilt für diese Ringe, dieses Raster und diese Regel (5-fach, nicht verschachtelt); Deutung, nicht gemessen: die Voreinstellung trifft schon fast die beste Zelle (Abstand 0,0030 und 0,0034), deshalb kann die Kreuzvalidierung sie kaum übertreffen | `test_claims.py` |
| Wahl von C bei n = 200 | Gepaarte Differenz CV minus Voreinstellung, drei Gruppen zu 20 Zufalls-Seeds, ohne Rauschen 0,0042 (Standardfehler 0,0010, 4,0 Standardfehler; besser in 22, schlechter in 35, gleich in 3) (je Gruppe 0,0032 / 0,0040 / 0,0053), mit Rauschen 0,0040 (Standardfehler 0,0014, 2,8 Standardfehler; besser in 24, schlechter in 35, gleich in 1) (je Gruppe 0,0052 / 0,0012 / 0,0055) | `test_claims.py` |
| Leave-one-out: Schranke mit dem b des Lösers | 300 Instanzen (n = 12 bis 30; Gauß-Klassen linear und Gauß, mit und ohne 10 % vertauschten Etiketten, Ringe, getrennte Klassen; C = 0,01 bis 1000, libsvm als Löser): 0 Verletzungen von „LOO-Fehler ≤ Zahl der Stützvektoren“, scharf (Gleichheit) in 27, Median 0,387, Maximum 1,000; 30 Instanzen ohne freien Stützvektor, darin 0 Verletzungen. Mit dem ungünstigsten b des Intervalls 13 verletzen die Schranke | `test_claims.py` |
| Leave-one-out: gezielte Suche | 600 kleine Instanzen (n = 6 bis 10, C = 0,02 bis 0,5, beide Klassen mindestens zweimal): mit dem b des Lösers 0 Verletzungen, **mit dem ungünstigsten b 23** (23 auch ohne die Zählung von f = 0 als Fehler), scharf in 293; unterscheiden sich beide Zählungen in 191 Instanzen. Exakt in Brüchen: x = (−1, 1, 3), y = (−1, +1, +1), C = 1/10 zeigt 3 Fehler bei 2 Stützvektoren **nur mit der Zählung „einklassiger Rest = Fehler“** (sonst 2 gegen 2); x = (−3, −2, −1, 0, 1), y = (−1, −1, −1, +1, +1), C = 1/20 zeigt 5 Fehler bei 4 Stützvektoren **ohne** diese Zählung (mit dem b des Lösers 3) | `test_exact.py`, `test_claims.py` |
| Schritte gegen C (eigenes SMO, n = 200, 5 Zufalls-Seeds, eps = 1e-6) | Gauß-Klassen mit 5 % vertauscht, Gauß σ = 1, C = 0,01 / 0,1 / 1 / 10 / 100 / 1000: 97 / 125 / 470 / 2932 / 22623 / 97409 Schritte (libsvm `n_iter_` 98 / 130 / 525 / 3188 / 20307 / 101804); bei C = 1000 je Zufalls-Seed 42509 bis 133483. Ringe, Gauß σ = 0,5: 105 / 181 / 283 / 1129 / 2902 / 8284 | `test_claims.py` |
| Schritte gegen n (C = 1, 20 Zufalls-Seeds, 5 % vertauscht, Gauß σ = 1) | Mittel bei n = 40 / 80 / 160 / 320 / 640 / 1280: 73 / 192 / 431 / 1069 / 2565 / 6163; Steigung von log(Schritte) gegen log(n) 1,27 (Bootstrap-90-%-Band [1,23; 1,32]; zwei Gruppen zu 10: 1,24 und 1,31; libsvm `n_iter_` 1,28; je Zufalls-Seed Median 1,26 [1,03; 1,56]; je Verdopplung 1,40 / 1,16 / 1,31 / 1,26 / 1,26; bei n = 640 zwischen 1555 und 5424 Schritten je Zufalls-Seed; 1,82 bis 4,81 Schritte je Punkt. Das Band der Steigung schließt 1 nicht ein (diese Instanzfamilie, C = 1, n = 40 bis 1280); über n = 1280 hinaus ist nichts gesagt | `test_claims.py` |
| Innere Punkte auf demselben QP | Newton-Schritte der eigenen Barriere (Einstellungen siehe Methodik) für n = 20 / 40 / 80 / 160 / 320: 89 / 111 / 106 / 115 / 145 (Mittel über 3 Zufalls-Seeds, Einzelwerte 85 bis 207); SMO-Schritte 45 / 109 / 156 / 719 / 1609; Modellzählung Newton-Schritte · n³/3 gegen SMO-Schritte · n: das 264 / 543 / 1457 / 1365 / 3075fache. Bei n = 80 hängt die Newton-Zahl an μ und der Abbruchlücke: 63 bis 398. `trust-constr` zählt Iterationen anderer Einheit (22 / 25 / 32 / 34 für n = 20 bis 160, Dualwert relativ nur auf 1,2e−05): nur als Band. Die Zählung der exakten Rechnung in n führt `gram-naeherung-demo` | `test_claims.py` |
| Eigenes SMO gegen libsvm | Dualwert relativ höchstens 5,3e−12 auf 60 Instanzen (20 Zufalls-Seeds × C = 0,1 / 1 / 10, n = 100, 5 % vertauscht, Gauß σ = 1); Schrittzahl gleich in 5 von 60, Verhältnis eigene Schritte zu `n_iter_` 0,82 bis 1,32 (Median 1,012): die Zähleinheit gilt für das jeweilige Programm | `test_claims.py`, `test_oracle_weiche_marge.py` |

## Ehrliche Grenzen

* **Ausgedachte Erzeuger, n = 60, ein Kern je Lauf.** Alle Zahlen gelten für diese Erzeuger und diese Einstellungen; „große C schaden beim Gauß-Kern“ ist für n = 60 und σ = 1 gemessen, nicht für wachsendes n. Ein Nutzen von Kernmethoden in Logistik, Lager oder Fertigung ist in den gelesenen Quellen nicht gemessen und nicht Gegenstand.
* **Der Löser schweigt nicht immer.** Bei unbegrenzter Schrittzahl kann libsvm nicht warnen (strukturell); mit endlicher Grenze warnt es nur an der Grenze. Das eigene SMO der App sagt es dort („nicht konvergiert“). Die KKT-Lücke bleibt klein, falsch ist das gestellte C.
* **Live-Modell ≠ Studie.** Die App rechnet mit dem eigenen SMO und der eigenen Barriere; Warnungen, Testfehler der C-Kurven und `n_iter_` der eingefrorenen Studie stammen von libsvm (Abweichung gemessen, siehe Befund „Eigenes SMO gegen libsvm“). Die Schrittzahlen der Studie zu C und n sind die des eigenen SMO.
* **Leave-one-out.** Die Zählung gilt für das b des Lösers; mit dem ungünstigsten b des optimalen Intervalls gilt die Schranke nicht (Gegenbeispiele oben). Eigene Herleitung, kein Satz der Quelle; ein Beweis für das b des Lösers fehlt.
* **Die Kreuzvalidierung ist hier nicht besser als die Voreinstellung**, ohne Rauschen im Mittel schlechter. Das gilt für 5-fache, nicht verschachtelte Kreuzvalidierung auf diesem Raster und diesen Ringen, nicht für „die Kreuzvalidierung“. Verschachtelte Kreuzvalidierung und die Validierungsmenge bei vertauschten Etiketten sind nicht gemessen.
* **Plattform.** Nur Windows und eine Python-Version; Schrittzahlen sind plattformabhängig (Rundung, Gleichstände), Tests prüfen Bänder. Keine Sekunden als Befund.
* **Nicht in dieser Demo** (nur Namensfallen oder Verweise): ν-SVM, `class_weight`, Platt-Skalierung, Stützvektor-Regression, Reparatur unzulässiger Kerne, Kriging, Landmarken, Baumverfahren, TabPFN.

### App-Fassung der Annahmen (🚧 Wo die Annahmen enden)

| Annahme | Was passiert, wenn sie verletzt ist | Gemessen |
|---|---|---|
| **Ausgedachte Erzeuger** (Gauß-Klassen, Ringe mit Streuung, fünf Punkte) | Alle Zahlen gelten für diese Erzeuger; es gibt keine reale Datenquelle, und ein Nutzen in Logistik, Lager oder Fertigung ist in den gelesenen Quellen nicht gemessen. | nicht Gegenstand |
| **n = 60 Trainingspunkte** in den C-Messungen (live bis 200) | „Große C schaden beim Gauß-Kern“ gilt für n = 60 und σ = 1; bei Steinwart (2001) muss C mit n wachsen, damit das Verfahren konsistent bleibt. | C gegen n nicht gemessen |
| **Ein Kern je Lauf** | Linear oder Gauß mit fester Bandbreite σ; ob eine Funktion ein Kern ist, wird hier nicht geprüft (das behandelt `kernwahl-demo`). | – |
| **SMO dieser Demo** | Die Schrittzahlen gelten für dieses SMO (zweite Ordnung, ohne Schrumpfen); libsvm zählt eigene `n_iter_`, Schrittzahl gleich nur in 5 von 60 Instanzen. | Windows, eine Python-Version |
| **Wahl von C per 5-facher Kreuzvalidierung** | Gemessen ist diese Regel auf dem Raster 6 C × 6 σ und diesen Ringen (n = 60, ohne Rauschen: Differenz CV minus Voreinstellung 0,0118 und 0,0150); nicht verschachtelt, nicht für andere Verfahren. | nur diese Regel |
| **Leave-one-out ≤ Stützvektoren** | Die Zählung gilt für das b des Lösers (300 + 600 Instanzen, 0 Verletzungen); mit dem ungünstigsten b des optimalen Intervalls verletzen 23 von 600 kleinen Instanzen sie. Eigene Herleitung, kein Satz der Quelle. | Instanzen |
| **Der Löser meldet nichts** | Bei unbegrenzter Schrittzahl kann libsvm nicht warnen; mit endlicher Grenze warnt es nur an der Grenze. Die KKT-Lücke bleibt klein, falsch ist das gestellte C. | Zufalls-Seeds, Zellen |

**Nicht angeboten, nicht gebaut, nicht gemessen** (kein Regler, kein Versprechen): ν-SVM, `class_weight` (C je Klasse) und Platt-Skalierung; Stützvektor-Regression; C gegen n, verschachtelte Kreuzvalidierung,
Validierungsmenge bei vertauschten Etiketten; Arbeitsmengenwahl gegen C; ein Fremd-Innere-Punkte-Löser mit der Einheit Newton-Schritt; Reparatur unzulässiger Kerne (`kernwahl-demo`), Kriging (`kernel-ridge-demo`), Landmarken (`gram-naeherung-demo`),
Baumverfahren (`kernel-oder-baum-demo`); Nutzen im Betrieb.

## Zahl → Test

| Zahl in der README | Test |
|---|---|
| Handbeispiel in Brüchen (α, w, b-Intervall, Dualwert, Schlupfsumme, Knick 10/9, SMO-Schritte 3 / 6 / 7 / 11 / 21 / 39 / 935, LOO 4 und 3, Gegenbeispiele 3 > 2 und 5 > 4) | `tests/test_exact.py` (exakt) |
| Spannen, Fehler je C, Löser-Zählung, Warnungen, Wahl, Leave-one-out, Schritte, Barriere, libsvm-Abweichung | `tests/test_claims.py` (jedes Fragment steht in der README; die wichtigsten Zahlen werden aus den Rohwerten unabhängig nachgerechnet; Zufallsergebnisse als Band) |
| Bayes-Fehler der Erzeuger | `tests/test_scenario.py` (Integral gegen `quad` und Stichprobe, vier Standardfehler) |
| SMO gegen libsvm, Primal (SLSQP), KKT nach Definition, harte Marge ab C ≥ max α, Leave-one-out durch Neuberechnung, Barriere | `tests/test_oracle_weiche_marge.py` |
| Eigenschaften (Σ yα = 0, Schranken, steigender Dualwert, Etiketten vertauschen, Skalierung, Daten verdoppeln und C halbieren, Monotonie in C) | `tests/test_eigenschaften_weiche_marge.py` |
| Verdikt-Texte (jede Zahl im Text, je Preset die erwartete Zeile) | `tests/test_verdicts.py` |
| Kopien (`wmr_smo.py`, `wmr_rng.py`) | `tests/test_copies.py` |
| Eingefrorene Studie (Aufbau, Stichproben mit `--check`) | `tests/test_studies.py` |
| Zahlen, die nur als Abweichung von der Vormessung genannt sind (Tabelle „Befunde und Korrekturen“) | kein Test; die Vormessung ist nicht Teil des Repos |

## Tests

200 Tests (`pytest --collect-only -q tests`): Kopier-Tests, Erzeuger und Bayes-Fehler, Handbeispiel in Brüchen mit Gegenwegen, Orakel, Eigenschaften, Randfälle, Barriere, Literaturwerte, Verdikte, eingefrorene Studie, Claims, Presets, Layout, AppTest-Smoke, Footer,
Laufzeit-Importe. Die Orakel-Tests laufen gegen libsvm, SLSQP, die Aufzählung in Brüchen und die harte Marge; die Tests mit scikit-learn und scipy überspringen sich, wenn die Pakete fehlen (`importorskip`). Der
Handbeispiel-Teil (Aufzählung in Brüchen, SMO in Brüchen mit 935 Schritten) und die AppTest-Läufe dauern am längsten (Sekundenangaben sind nicht gemessen und kein Befund).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (fünf Experimente, Kernfragen, 🚧-Abschnitt, 📐-Expander) |
| `wmr_smo.py` | SMO (wörtliche Kopie von `hmg_smo.py`, Kopier-Test) |
| `wmr_rng.py` | SplitMix64 (Kopie der Vorlage, Kopier-Test) |
| `wmr_scenario.py` | Erzeuger (Gauß-Klassen, Ringe mit Streuung, Handbeispiel), Vertauschen von Etiketten, Bayes-Fehler |
| `wmr_exact.py` | Aufzählung in Brüchen (Handbeispiel), Leave-one-out in Brüchen, die zwei exakten Gegenbeispiele |
| `wmr_barrier.py` | Log-Barriere mit Newton-Schritten (nur zum Zählen) |
| `wmr_evaluation.py` | Gram-Matrix, Modell, b-Intervall, Leave-one-out, die Rechnungen der Experimente, Verdikte als reine Funktionen |
| `wmr_studies.py` | Aggregate der eingefrorenen Studien aus den Rohwerten |
| `wmr_frozen.json` | eingefrorene Studien (libsvm-Messungen, Schritte, Leave-one-out, Barriere) |
| `tools/wmr_make_frozen.py` | Messwerkzeug (scikit-learn und scipy nur hier), `--check` |
| `wmr_constants.py`, `wmr_presets.py`, `wmr_layout.py`, `wmr_visualization.py`, `wmr_texts.py` | Regler und Presets, Permalink und Hervorhebung, fester Kopf, Plotly-Figuren, Texte |
| `tests/` | siehe Abschnitt Tests |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, pandas, plotly) und Tests (pytest, scipy, scikit-learn) |

## Bewusst nicht umgesetzt

ν-SVM, `class_weight` (C je Klasse) und Platt-Skalierung (Wahrscheinlichkeiten aus Entscheidungswerten); Stützvektor-Regression; C gegen n, verschachtelte Kreuzvalidierung, Validierung bei vertauschten Etiketten; Arbeitsmengenwahl gegen C
(die drei Regeln stehen als Regler, gemessen sind sie nur im Zweig-Test); ein Fremd-Innere-Punkte-Löser mit der Einheit Newton-Schritt (`cvxopt` und `cvxpy` sind im Test-venv nicht vorhanden); die Erklärung des SMO-Schritts und der
Gram-Speicher (dafür `ocsvm-demo`); Kalibrierung von Entscheidungswerten (`kalibrierung-schwelle-demo`).

## Quellen

Cortes/Vapnik 1995 (Support-Vector Networks; weiche Marge, Gl. 13 und 22 bis 29, Gl. 5), Platt 1998 (SMO, KKT Gl. 12), Fan/Chen/Lin 2005 (Arbeitsmenge zweiter Ordnung), Chang/Lin (LIBSVM), scikit-learn-Dokumentation zu `SVC`,
Steinwart 2001 (Konsistenz, C wächst mit n; nur als Hinweis), Wainberg u. a. 2016 (Hyperparameter auf Testdaten; die Falle, nicht die Zahlen), Boyd/Vandenberghe (Barrieremethode). Gleichungen und Befunde sind in eigenen Worten wiedergegeben; die
Leave-one-out-Zählung ist eigene Herleitung.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, pandas und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html).
