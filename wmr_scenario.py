"""Erzeuger der Instanzen (alle ausgedacht, keine Messdaten) und das feste Handbeispiel.

Zufallsquelle überall SplitMix64 (`wmr_rng.py`, plattformfest). Normalwerte für die Trainingsinstanzen der App kommen aus `normal()` der Vorlage (Box-Muller mit exaktem
Logarithmus und Kosinus, bitgleich auf Windows und Linux); die großen Testmengen (2 000 Punkte) nutzen `fast_normal` (Inverse der Verteilungsfunktion aus `random()`,
nur bis auf die letzte Binärstelle plattformgleich). Erzeuger:

* „Gauß-Klassen“: zwei überlappende Klassen N((±Abstand/2, 0), I) im Abstand 2 (Bayes-Fehler Φ(−1) = 0,1587), Labels abwechselnd.
* „Ringe mit Streuung“: zwei Ringe um ein Depot, Radius 1 (Klasse −1) und 2 (Klasse +1), Radius normalverteilt mit Streuung s (bei 0 abgeschnitten, durch Verwerfen),
  Winkel gleichverteilt; Bayes-Fehler bei s = 0,35: 0,0766 (Integral, siehe `bayes_ringe`).
* „Fünf Punkte auf der Geraden“: das Handbeispiel (nicht trennbar), x = (0, 1, 2, 3, 4), y = (−1, −1, +1, −1, +1).
* Vertauschen von Labels: ein Anteil `flip` der Punkte (gerundet) bekommt das umgekehrte Label; die Auswahl ist ein `shuffle` der Indizes (die ersten k).
"""

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from wmr_rng import SplitMix64

_ND = NormalDist()
R_IN, R_OUT = 1.0, 2.0


@dataclass
class Instance:
    X: np.ndarray
    y: np.ndarray
    name: str = ""
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)


def fast_normal(rng: SplitMix64) -> float:
    """Standardnormal über die Inverse der Verteilungsfunktion (schnell, für große Testmengen)."""
    return _ND.inv_cdf(min(max(rng.random(), 1e-12), 1.0 - 1e-12))


def _labels(n: int) -> np.ndarray:
    return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def flip_labels(y: np.ndarray, flip: float, rng: SplitMix64) -> np.ndarray:
    """Ein Anteil `flip` der Labels (k = round(flip · n) Punkte, ausgewählt als die ersten k eines `shuffle` der Indizes) wird umgekehrt; flip = 0 ändert nichts und zieht keine Zufallszahl."""
    y = np.array(y, float)
    k = int(round(flip * len(y)))
    if k <= 0:
        return y
    idx = list(range(len(y)))
    rng.shuffle(idx)
    for i in idx[:k]:
        y[i] = -y[i]
    return y


def gauss_classes(n: int, seed: int, flip: float = 0.0, sep: float = 2.0, fast: bool = False) -> Instance:
    """Gauß-Klassen N((±sep/2, 0), I), Labels abwechselnd (+1 bei geradem Index), danach Vertauschen eines Anteils `flip`."""
    rng = SplitMix64(seed)
    nrm = fast_normal if fast else (lambda r: r.normal())
    y = _labels(n)
    X = np.array([(nrm(rng), nrm(rng)) for _ in range(n)])
    X[:, 0] += y * sep / 2.0
    y_obs = flip_labels(y, flip, rng)
    return Instance(X, y_obs, "Gauß-Klassen", {"seed": seed, "sep": sep, "flip": flip, "y_true": y})


def ringe_streuung(n: int, seed: int, s: float = 0.35, flip: float = 0.0, fast: bool = False) -> Instance:
    """Ringe mit Streuung: Klasse +1 außen (Radius 2), Klasse −1 innen (Radius 1), Radius = Mitte + s · Normalwert (nur Werte > 0, sonst neu gezogen), Winkel gleichverteilt."""
    rng = SplitMix64(seed)
    nrm = fast_normal if fast else (lambda r: r.normal())
    y = _labels(n)
    pts = []
    for i in range(n):
        mu = R_OUT if y[i] > 0 else R_IN
        while True:
            r = mu + s * nrm(rng)
            if r > 0:
                break
        pts.append(r)
    pts = np.array(pts)
    ang = np.array([2.0 * math.pi * rng.random() for _ in range(n)])
    X = np.c_[pts * np.cos(ang), pts * np.sin(ang)]
    y_obs = flip_labels(y, flip, rng)
    return Instance(X, y_obs, "Ringe mit Streuung", {"seed": seed, "s": s, "flip": flip, "y_true": y})


def fuenf_punkte() -> Instance:
    """Handbeispiel: fünf Punkte auf einer Geraden; der Punkt x = 3 hat das Label −1, liegt aber zwischen +1-Punkten: nicht trennbar."""
    return Instance(np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), np.array([-1.0, -1.0, 1.0, -1.0, 1.0]), "Fünf Punkte auf der Geraden")


DATASETS = ("Gauß-Klassen", "Ringe mit Streuung", "Fünf Punkte auf der Geraden")


def make_instance(data: str, n: int, seed: int, flip: float, fast: bool = False) -> Instance:
    if data == DATASETS[0]:
        return gauss_classes(n, seed, flip, fast=fast)
    if data == DATASETS[1]:
        return ringe_streuung(n, seed, flip=flip, fast=fast)
    return fuenf_punkte()


def test_set(data: str, m: int, seed: int) -> Instance:
    """Frische, saubere Testpunkte (keine vertauschten Labels) desselben Modells; Zufalls-Seed getrennt vom Training."""
    return make_instance(data, m, seed + 7_000_003, 0.0, fast=True)


def bayes_gauss(sep: float = 2.0) -> float:
    """Bayes-Fehler zweier Gauß-Klassen im Abstand `sep` (gleiche Anteile, Einheitskovarianz): Φ(−sep/2)."""
    return _ND.cdf(-sep / 2.0)


def bayes_ringe(s: float, r_in: float = R_IN, r_out: float = R_OUT, steps: int = 20000) -> float:
    """Bayes-Fehler der Ringe: ½ ∫ min(f_innen, f_außen) dr mit f = bei 0 abgeschnittene Normalverteilung in r (Trapezregel auf [0, 6]; die Messwerkzeuge prüfen es gegen `scipy.integrate.quad`)."""
    def f(r, mu):
        return _ND.pdf((r - mu) / s) / s / (1.0 - _ND.cdf(-mu / s))
    h = 6.0 / steps
    tot = 0.0
    for k in range(steps + 1):
        r = k * h
        v = min(f(r, r_in), f(r, r_out))
        tot += v * (0.5 if k in (0, steps) else 1.0)
    return 0.5 * tot * h
