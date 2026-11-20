"""Aggregate der eingefrorenen Studien (`wmr_studies.py`) an handgesetzten Mini-Daten, Aufbau und Stimmigkeit von `wmr_frozen.json`, Stichprobe des Messwerkzeugs (`--check`)."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

import wmr_studies as St

ROOT = Path(__file__).resolve().parent.parent
D = json.loads((ROOT / "wmr_frozen.json").read_text(encoding="utf-8"))


def test_se_span_flat_from_und_bad_cells_von_hand():
    assert St.se([1.0, 3.0]) == pytest.approx(math.sqrt(2.0) / math.sqrt(2.0))  # Standardabweichung (ddof 1) sqrt(2), durch sqrt(2)
    assert St.span([0.2, 0.1, 0.4]) == pytest.approx(0.3)
    cs = [1, 2, 3, 4]
    assert St.flat_from([0.5, 0.2, 0.205, 0.2], cs, 0.01) == 2 and St.flat_from([0.1, 0.2, 0.3, 0.4], cs, 0.01) == 4
    arr = [[0.1, 0.1], [0.2, 0.16], [0.3, 0.1]]  # [C][Zufalls-Seed]; bester je Zufalls-Seed: 0,1 und 0,1; über 0,05 darüber: 0,2 / 0,3 / 0,16 / (0,1 nicht)
    assert St.bad_cells(arr, 0.05) == (3, 6)


def test_wahl_aggregat_von_hand():
    te = [[0.1, 0.2] + [0.5] * 34, [0.3, 0.1] + [0.5] * 34]
    cv = [[0.1, 0.2] + [0.9] * 34, [0.1, 0.3] + [0.9] * 34]  # CV wählt Zelle 0 in beiden Zufalls-Seeds
    d = {"x": {"gruppen": {"1": {"f": {"te": te, "cv": cv, "default": [0.15, 0.1], "sigma0": [1.0, 1.2]}}}}}
    w = St.wahl(d, "x", 1, "f")
    assert w["rueckblick"] == pytest.approx(0.1) and w["cv"] == pytest.approx(0.2) and w["default"] == pytest.approx(0.125)
    assert w["diff_mean"] == pytest.approx(0.075) and w["cv_besser"] == 1 and w["cv_schlechter"] == 1 and w["cv_hoechstens_1_5"] == 1 and w["sigma0_mittel"] == pytest.approx(1.1)
    assert w["beste_zelle"] == (0, 1) and w["beste_zelle_fehler"] == pytest.approx(0.15)  # mittlerer Testfehler 0,2 in Zelle 0, 0,15 in Zelle 1 (Zeile 0, Spalte 1)
    assert w["mean_te"].shape == (6, 6)


def test_validierung_optimismus_von_hand():
    te = [[0.2, 0.3] + [0.5] * 34, [0.1, 0.4] + [0.5] * 34]
    va = [[0.05, 0.3] + [0.5] * 34, [0.2, 0.05] + [0.5] * 34]  # gewählt: Zelle 0 bzw. Zelle 1
    d = {"wahl60": {"gruppen": {"1": {"flip0": {"te": te, "va": {"30": va}}}}}}
    v = St.validierung(d, 1, 30)
    assert v["optimistisch"] == pytest.approx(0.05) and v["ehrlich"] == pytest.approx(0.3) and v["optimismus"] == pytest.approx(0.25) and v["n_seeds"] == 2


def test_aufbau_der_eingefrorenen_studie():
    c = D["c_wirkung"]
    assert c["C"] == [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] and list(c["faelle"]) == ["gauss_lin", "gauss_lin_flip", "gauss_rbf", "gauss_rbf_flip", "ringe_rbf", "ringe_rbf_flip"]
    for key, case in c["faelle"].items():
        for g in ("1", "2"):
            for name, arr in case["gruppen"][g].items():
                assert np.array(arr).shape == (7, 30), (key, g, name)
    for gk, n_seeds in (("wahl60", 30), ("wahl200", 20)):
        for g, gd in D[gk]["gruppen"].items():
            for fk, v in gd.items():
                assert np.array(v["te"]).shape == (n_seeds, 36) and np.array(v["cv"]).shape == (n_seeds, 36) and len(v["default"]) == n_seeds
    assert sorted(D["wahl60"]["gruppen"]) == ["1", "2"] and sorted(D["wahl200"]["gruppen"]) == ["1", "2", "3"]
    for m in ("30", "100", "500"):
        assert np.array(D["wahl60"]["gruppen"]["1"]["flip0"]["va"][m]).shape == (30, 36)
    assert len(D["loo"]["e1"]) == 300 and len(D["loo"]["e5"]) == 600
    assert D["schritte_n"]["n"] == [40, 80, 160, 320, 640, 1280] and np.array(D["schritte_n"]["own"]).shape == (6, 20)
    assert D["barriere"]["n"] == [20, 40, 80, 160, 320] and np.array(D["barriere"]["newton"]).shape == (5, 3)


def test_werte_liegen_in_den_zulaessigen_bereichen():
    for case in D["c_wirkung"]["faelle"].values():
        for g in case["gruppen"].values():
            assert 0 <= np.min(g["test"]) and np.max(g["test"]) <= 1 and np.min(g["n_iter"]) >= 1 and set(np.unique(g["warn100k"])) <= {0, 1}
            assert np.min(g["sv"]) > 0 and np.max(g["sv"]) <= 1 and np.max(g["gap"]) < 1e-2
    assert D["bayes"]["gauss"] == pytest.approx(0.158655, abs=1e-6)


def test_zwei_gruppen_sind_verschiedene_zufalls_seed_mengen():
    for key in D["c_wirkung"]["faelle"]:
        a, b = St.case_group(D, key, 1), St.case_group(D, key, 2)
        assert a["test"] != b["test"]


def test_loeser_zaehlung_ohne_grenze_ist_strukturell_null_und_mit_grenze_nur_bei_linearem_kern():
    for g in (1, 2):
        t = St.solver_totals(D, g)
        assert t["warn_unbegrenzt"] == 0 and t["cells"] == 1260 and t["gap_max"] <= 2e-3
        assert 0 < t["warn_cap"] <= 40 and all(k.startswith("gauss_lin") for k, _, _ in St.warn_cap_cells(D, g))
        assert all(c >= 100.0 for _, c, _ in St.warn_cap_cells(D, g))  # nur bei großem C
    w = St.warn_harte(D)
    assert w["von"] == 180 and w["gepoolt"][0] <= 2 and w["gepoolt"][2] > 30 and w["gepoolt"][3] > 30  # bei C = 1e2 fast nie, bei 1e6 und 1e8 oft


def test_schritte_nehmen_mit_c_zu_und_die_steigung_liegt_im_band():
    sc = St.steps_c(D, "gauss")
    assert sc["own_mean"] == sorted(sc["own_mean"]) and sc["own_mean"][-1] > 20 * sc["own_mean"][2]
    sn = St.steps_n(D)
    assert 1.0 < sn["slope_all"] < 1.6 and sn["band"][0] < sn["slope_all"] < sn["band"][1] and sn["min"][4] < sn["max"][4]
    assert sn["mean_all"] == sorted(sn["mean_all"])


def test_barriere_zaehlung_einstellungsabhaengig_und_modellzaehlung_waechst():
    b = St.barrier(D)
    assert b["abh_min"] < b["abh_max"] and 50 <= b["newton_min"] <= b["newton_max"] <= 400
    assert b["modell"][-1] > b["modell"][0] > 100  # Modellzählung Newton · n³/3 gegen SMO · n wächst mit n


def test_leave_one_out_studie_schranke_mit_b_des_loesers_und_luecke_beim_ungunstigen_b():
    e1, e5 = St.loo_e1(D), St.loo_e5(D)
    assert e1["n"] == 300 and e1["verletzungen"] == 0 and e1["degeneriert_verletzt"] == 0 and e1["ratio_max"] <= 1.0
    assert e5["n"] == 600 and e5["verl_solver"] == 0 and e5["verl_ungunstig"] > 0 and e5["einklassig"] == 0 and e5["verl_ungunstig_strikt"] > 0


def test_messwerkzeug_check_stichproben_stimmen():
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("wmr_make_frozen", ROOT / "tools" / "wmr_make_frozen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check() == 0


def test_sigma_raender_von_hand():
    m = np.full((6, 6), 0.2)
    m[:, 0] = [0.4, 0.4, 0.3, 0.26, 0.27, 0.28]
    m[:, -1] = [0.31, 0.30, 0.12, 0.11, 0.10, 0.09]
    r = St.sigma_raender(m)
    assert r == {"schmal_min": 0.26, "breit_klein_c_min": 0.30, "breit_klein_c_max": 0.31, "breit_gross_c_max": 0.11}
