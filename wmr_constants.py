"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

SEED_MAX, DEFAULT_SEED = 999999, 7

DATASETS = ("Gauß-Klassen", "Ringe mit Streuung", "Fünf Punkte auf der Geraden")
DEFAULT_DATA = DATASETS[2]  # das Handbeispiel: klein, von Hand nachrechenbar, mit dem Knick bei C = 1/3
FLIP_OPTIONS = (0.0, 0.1)  # Anteil vertauschter Labels im Training; der Test bleibt sauber
DEFAULT_FLIP = 0.0
KERN_OPTIONS = ("linear", "Gauß")
DEFAULT_KERN = "linear"
SIGMA_OPTIONS = (0.1, 0.2, 0.4, 0.5, 0.8, 1.0, 1.6, 3.2)  # Bandbreite σ (Längenmaß); γ = 1/(2σ²) nur in der Umrechnungstabelle
DEFAULT_SIGMA = 1.0
C_OPTIONS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_C = 1.0
N_MIN, N_MAX, DEFAULT_N = 20, 200, 60
STEPS_OPTIONS = (10000, 30000, 100000)  # Schrittgrenze des eigenen SMO: bei Erreichen hält es an und sagt es
DEFAULT_STEPS = 30000
EPS_OPTIONS = (1e-3, 1e-6)  # Abbruch bei KKT-Lücke ≤ eps (1e-3 ist die Voreinstellung von libsvm)
DEFAULT_EPS = 1e-3
N_TEST = 2000  # frische Testpunkte je Zufalls-Seed

# Experiment 3: Wahl von C
VAL_M_OPTIONS = (30, 100, 500)
DEFAULT_VAL_M = 30
GROUP_OPTIONS = (1, 2)  # Zufalls-Seed-Gruppe der eingefrorenen Studie
DEFAULT_GROUP = 1

# Experiment 4: Leave-one-out (Live-Rechnung)
LOO_N_MIN, LOO_N_MAX, DEFAULT_LOO_N = 8, 40, 20
LOO_STEPS = 8000  # Schrittgrenze je Neuberechnung beim Herausnehmen (Live); wird sie erreicht, sagt die Tabelle es

# Experiment 5: Schritte zählen
E5_N_MIN, E5_N_MAX, DEFAULT_E5_N = 20, 320, 100
WSS_OPTIONS = ("zweite Ordnung", "erstes Paar", "zufälliges zweites")
WSS_CODE = {"zweite Ordnung": "second", "erstes Paar": "first", "zufälliges zweites": "random"}
DEFAULT_WSS = WSS_OPTIONS[0]

# Handbeispiel: C-Werte der Tabelle (Brüche)
HAND_C = ((1, 8), (1, 4), (3, 10), (1, 3), (1, 2), (1, 1), (2, 1), (4, 1), (100, 1))

# eingefrorene Studien (tools/wmr_make_frozen.py)
STUDY_FILE = "wmr_frozen.json"

COLOR_POS = "#2F6B65"
COLOR_NEG = "#B5543A"
COLOR_LINE = "#222222"
COLOR_MARGIN = "#7a7a7a"
COLOR_ALT = "#4c78a8"
COLOR_BOUND = "#c9a227"

PRESETS = {
    "Handbeispiel (fünf Punkte auf der Geraden)": {"data": "Fünf Punkte auf der Geraden", "kern": "linear", "c": 1.0},
    "Linearer Kern: C ist egal": {"data": "Gauß-Klassen", "kern": "linear", "c": 1.0},
    "Gauß-Kern: C zu groß": {"data": "Gauß-Klassen", "kern": "Gauß", "sigma": 1.0, "c": 1000.0, "steps": 100000},
    "C zu klein (Mehrheitsklasse)": {"data": "Gauß-Klassen", "kern": "Gauß", "sigma": 1.0, "c": 0.001, "flip": 0.1},
    "Ringe: C und Bandbreite": {"data": "Ringe mit Streuung", "kern": "Gauß", "sigma": 0.5, "c": 1.0},
    "Rausch-Labels": {"data": "Gauß-Klassen", "kern": "linear", "c": 0.001, "flip": 0.1},
}
PRESET_HELP = {
    "Handbeispiel (fünf Punkte auf der Geraden)": "Fünf Punkte auf einer Geraden, einer mit dem falschen Label: nicht trennbar. Die Tabelle aus der Aufzählung zeigt den Knick bei C = 1/3: darüber ändert C die Funktion nicht mehr. Von Hand in Brüchen nachrechenbar.",
    "Linearer Kern: C ist egal": "Gauß-Klassen ohne Rauschen, linearer Kern, C = 1: die eingefrorene Kurve zeigt, wie weit C den mittleren Testfehler über sieben Stufen von 1e-3 bis 1e3 ändert (Mittel über 30 Zufalls-Seeds; die einzelne Live-Instanz kann am kleinsten C abweichen).",
    "Gauß-Kern: C zu groß": "Gauß-Klassen, Gauß-Kern σ = 1, C = 1000, Schrittgrenze 100 000: der Löser rechnet mit der Schrittgrenze 100 000; die eingefrorene Kurve zeigt, dass der mittlere Testfehler bei großem C höher liegt als bei kleinem.",
    "C zu klein (Mehrheitsklasse)": "Gauß-Klassen mit 10 % vertauschten Labels, Gauß-Kern, C = 1e-3: fast alle Multiplikatoren stoßen an die Schranke; sind die Klassen durch das Vertauschen ungleich besetzt, ordnet die Lösung jeden Punkt der Mehrheitsklasse zu. Der Testfehler wechselt dann je Zufalls-Seed; die Kurve zeigt das Mittel.",
    "Ringe: C und Bandbreite": "Ringe mit Streuung, Gauß-Kern σ = 0,5, C = 1: die gute Lage ist eine Lage im Raster aus C und σ zusammen; die eingefrorene Tabelle zeigt die Spanne über C je Bandbreite.",
    "Rausch-Labels": "Gauß-Klassen mit 10 % vertauschten Labels, linearer Kern, C = 1e-3: auch der lineare Kern kippt bei sehr kleinem C; die eingefrorene Kurve zeigt, ab welchem C der mittlere Testfehler flach ist.",
}
