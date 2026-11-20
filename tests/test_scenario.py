"""Erzeuger der Instanzen: eingefrorene Werte je festem Zufalls-Seed (erste Zahlen und Prüfsumme), Eigenschaften der Modelle als Band, Zufallsquelle SplitMix64, Bayes-Fehler gegen Integral.

Normalwerte der Trainingsinstanzen kommen aus `normal()` der Vorlage (exakter Logarithmus und Kosinus, bitgleich auf Windows und Linux); Winkel und Ringe nutzen `math.cos` und `math.sin`
der C-Bibliothek, darum stehen die eingefrorenen Werte mit Toleranz 1e-9 (und die Prüfsumme auf 9 Stellen gerundet)."""

import hashlib
import math

import numpy as np
import pytest

import wmr_scenario as Sc
from wmr_rng import SplitMix64


def _checksum(ins):
    return hashlib.sha256(np.round(np.c_[ins.X, ins.y], 9).tobytes()).hexdigest()[:16]


def test_splitmix64_referenzwerte_und_normalzahlen():
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [16294208416658607535, 7960286522194355700, 487617019471545679]
    n = SplitMix64(1)
    assert [n.normal() for _ in range(3)] == pytest.approx([-0.028249746095855, -0.227919522867635, 0.10309095168574], abs=1e-12)


def test_eingefrorene_instanzen_je_zufalls_seed():
    g = Sc.gauss_classes(8, 7)
    assert np.round(g.X[:3], 6).tolist() == [[2.364992, -0.396524], [-0.995501, -0.580613], [-0.712889, 2.062278]]
    assert g.y.tolist() == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0] and _checksum(g) == "8dd01fca02cbc58f"
    r = Sc.ringe_streuung(8, 7, s=0.35)
    assert np.round(r.X[:3], 6).tolist() == [[1.802078, -1.700514], [-0.397533, 0.763977], [-1.466633, -1.362089]] and _checksum(r) == "658a95dff83f6d56"
    gf = Sc.gauss_classes(10, 7, flip=0.2)
    assert gf.y.tolist() == [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0] and _checksum(gf) == "9ac287f0479e3fbb"


def test_gleicher_zufalls_seed_gleiche_instanz_anderer_seed_andere_instanz():
    assert np.array_equal(Sc.gauss_classes(12, 5, 0.1).X, Sc.gauss_classes(12, 5, 0.1).X)
    assert not np.array_equal(Sc.gauss_classes(12, 5).X, Sc.gauss_classes(12, 6).X)
    assert np.array_equal(Sc.ringe_streuung(12, 5, flip=0.1).y, Sc.ringe_streuung(12, 5, flip=0.1).y)


def test_vertauschen_zaehlt_k_punkte_und_aendert_nichts_bei_null():
    for n, fl in ((20, 0.1), (60, 0.1), (200, 0.1), (13, 0.1)):
        ins = Sc.gauss_classes(n, 3, fl)
        assert int((ins.y != ins.info["y_true"]).sum()) == int(round(fl * n))
    ins = Sc.gauss_classes(30, 3, 0.0)
    assert np.array_equal(ins.y, ins.info["y_true"])
    r = Sc.ringe_streuung(30, 3, flip=0.1)
    assert int((r.y != r.info["y_true"]).sum()) == 3


def test_ringe_radien_positiv_und_lage_der_klassen_als_band():
    ins = Sc.ringe_streuung(4000, 11, s=0.35, fast=True)
    rad = np.linalg.norm(ins.X, axis=1)
    assert (rad > 0).all()
    inner, outer = rad[ins.info["y_true"] < 0], rad[ins.info["y_true"] > 0]
    assert abs(inner.mean() - 1.0) < 4.5 * 0.35 / math.sqrt(2000) + 0.01 and abs(outer.mean() - 2.0) < 4.5 * 0.35 / math.sqrt(2000)
    assert abs(inner.std() - 0.35) < 0.03 and abs(outer.std() - 0.35) < 0.03


def test_gauss_klassen_mittelpunkte_als_band():
    ins = Sc.gauss_classes(4000, 13, fast=True)
    pos, neg = ins.X[ins.y > 0], ins.X[ins.y < 0]
    se = 1.0 / math.sqrt(2000)
    assert abs(pos[:, 0].mean() - 1.0) < 4.5 * se and abs(neg[:, 0].mean() + 1.0) < 4.5 * se and abs(pos[:, 1].mean()) < 4.5 * se


def test_bayes_fehler_gauss_gegen_stichprobe_und_wert():
    ins = Sc.gauss_classes(100000, 17, fast=True)
    e = float((np.where(ins.X[:, 0] >= 0, 1.0, -1.0) != ins.y).mean())
    bay = Sc.bayes_gauss()
    assert bay == pytest.approx(0.1586552539, abs=1e-9)
    assert abs(e - bay) < 4 * math.sqrt(bay * (1 - bay) / 100000)


def test_bayes_fehler_ringe_integral_gegen_quad_und_stichprobe():
    pytest.importorskip("scipy")
    from scipy import integrate
    from scipy.stats import norm

    for s, soll in ((0.25, 0.02275), (0.35, 0.07665), (0.45, 0.13415)):
        def f(r, mu, s=s):
            return norm.pdf(r, mu, s) / (1 - norm.cdf(0, mu, s))
        val, _ = integrate.quad(lambda r: min(f(r, 1.0), f(r, 2.0)), 0, 6, points=[1.5], limit=200)
        assert Sc.bayes_ringe(s) == pytest.approx(0.5 * val, abs=1e-5)  # Trapezregel mit Knick der min-Funktion: Abweichung gemessen höchstens 4,3e-6
        assert Sc.bayes_ringe(s) == pytest.approx(soll, abs=1e-4)
    ins = Sc.ringe_streuung(100000, 19, s=0.35, fast=True)
    e = float((np.where(np.linalg.norm(ins.X, axis=1) < 1.5, -1.0, 1.0) != ins.y).mean())
    bay = Sc.bayes_ringe(0.35)
    assert abs(e - bay) < 4 * math.sqrt(bay * (1 - bay) / 100000)


def test_handbeispiel_fest_und_nicht_trennbar_von_hand(hand):
    assert hand.X[:, 0].tolist() == [0, 1, 2, 3, 4] and hand.y.tolist() == [-1, -1, 1, -1, 1] and hand.n == 5
    # x = 3 (Etikett −1) liegt zwischen den +1-Punkten x = 2 und x = 4: keine Schwelle trennt (jede Schwelle t, f = ±(x − t), gibt mindestens einen Fehler)
    fehler = min(int((np.where(s * (hand.X[:, 0] - t) >= 0, 1.0, -1.0) != hand.y).sum()) for t in np.arange(-0.5, 4.6, 0.5) for s in (1, -1))
    assert fehler >= 1


def test_testmenge_ist_sauber_frisch_und_vom_training_getrennt():
    tr = Sc.make_instance("Gauß-Klassen", 60, 5, 0.1)
    te = Sc.test_set("Gauß-Klassen", 2000, 5)
    assert te.n == 2000 and np.array_equal(te.y, te.info["y_true"]) and te.info["flip"] == 0.0
    assert not np.array_equal(tr.X[:5], te.X[:5])
