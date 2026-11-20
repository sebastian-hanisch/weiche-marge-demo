"""Verdikte (reine Funktionen in wmr_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text; je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a Punkt 7)."""

from fractions import Fraction as F

import numpy as np
import pytest

import wmr_constants as C
import wmr_evaluation as ev
import wmr_exact as Ex
import wmr_studies as St

CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def test_verdict_c_flach_unter_der_kleinen_schwelle_info_mit_allen_zahlen():
    flach = [0.170, 0.168, 0.169, 0.170, 0.171, 0.172, 0.173]
    kind, text = ev.verdict_c(CS, [flach, flach], 0.1587, label="Test")
    assert kind == "info" and "Spanne 0,0050 und 0,0050" in text and "zwischen 0,1680 und 0,1730" in text and "Bayes-Fehler 0,1587" in text
    assert "kleinsten bei C = 0,01" in text and "am größten bei C = 1000" in text and "höchstens 0,02" in text and text.startswith("Test: ")
    assert "Ab C = 0,001 beträgt die Spanne höchstens 0,0050" in text


def test_verdict_c_stark_warnung_und_flacher_bereich_wird_genannt():
    rausch = [0.40, 0.25, 0.172, 0.173, 0.174, 0.173, 0.173]  # Rauschetiketten, linear: kippt am linken Rand, ab C = 0,1 flach
    kind, text = ev.verdict_c(CS, [rausch, [v + 0.01 for v in rausch]], None)
    assert kind == "warning" and "C wirkt hier stark" in text and "Spanne 0,2280 und 0,2280" in text
    assert "Ab C = 0,1 beträgt die Spanne höchstens 0,0020" in text and "am größten bei C = 0,001" in text and "Bayes" not in text


def test_verdict_c_dazwischen_und_verschiedene_gruppen():
    mid = [0.13, 0.13, 0.13, 0.11, 0.13, 0.15, 0.15]
    kind, text = ev.verdict_c(CS, [mid, mid], 0.0766)
    assert kind == "info" and "Spanne 0,0400 und 0,0400" in text and "messbar, aber nicht stark" in text
    g2 = [0.17 + 0.0001 * i for i in range(7)]
    assert "(Gruppe 1) bzw." in ev.verdict_c(CS, [mid, g2], None)[1]


def _row(c, gap, conv, err, msg=""):
    return {"c": c, "gap": gap, "converged": conv, "err": err, "message": msg, "eps": 1e-3, "steps": 10}


def test_verdict_loeser_richtig_geloest_falsch_gestellt_ist_eine_warnung_mit_zaehlung():
    rows = [_row(0.001, 5e-4, True, 0.40), _row(1.0, 8e-4, True, 0.17), _row(1000.0, 9e-4, True, 0.26)]
    kind, text = ev.verdict_loeser(rows)
    assert kind == "warning" and text.startswith("In 3 von 3 Zeilen meldet der Löser Konvergenz (KKT-Lücke höchstens 0,0009 bei eps = 0,001)") and "eine Meldung gibt es in keiner Zeile" in text
    assert "in 2 der konvergierten Zeilen mehr als 0,05 über dem besten dieser Zeile (0,1700)" in text and "Testfehler" in text


def test_verdict_loeser_meldung_nur_an_der_schrittgrenze_und_kein_schaden_ohne_abweichung():
    rows = [_row(1.0, 5e-4, True, 0.17), _row(100.0, 9e-3, False, 0.17, "Schrittgrenze 500 erreicht"), _row(1000.0, 2e-2, False, 0.18, "Schrittgrenze 500 erreicht")]
    kind, text = ev.verdict_loeser(rows)
    assert kind == "info" and "In 1 von 3 Zeilen" in text and "in 2 Zeilen (C = 100, 1000: „nicht konvergiert“)" in text and "kostet kein C mehr als diesen Betrag" in text
    kind, text = ev.verdict_loeser([_row(1.0, 0.1, False, 0.2, "x")], hand=True)
    assert "In keiner der 1 Zeilen erreicht der Löser die Toleranz" in text


def test_verdict_wahl_optimismus_ueber_zwei_standardfehlern_ist_eine_warnung():
    val = {"m": 30, "optimistisch": 0.05, "ehrlich": 0.11, "optimismus": 0.06, "se": 0.01, "n_seeds": 30}
    cv = {"cv": 0.109, "default": 0.097, "diff_mean": 0.012, "diff_se": 0.004, "diff_se_zahl": 3.0, "cv_besser": 7, "cv_schlechter": 22, "n_seeds": 30}
    kind, text = ev.verdict_wahl(val, cv, {"mean": 0.004, "se": 0.001, "n_seeds": 60})
    assert kind == "warning" and "30 frisch gezogenen Validierungspunkten" in text and "Fehler dort 0,0500" in text and "auf frischen Punkten 0,1100" in text and "Optimismus 0,0600 (Standardfehler 0,0100)" in text
    assert "gepaarte Differenz je Zufalls-Seed 0,0120 (Standardfehler 0,0040, 3,0 Standardfehler; Kreuzvalidierung besser in 7, schlechter in 22 von 30)" in text and "Bei n = 200: 0,0040" in text
    assert ev.verdict_wahl({**val, "optimismus": 0.015}, cv)[0] == "info"  # 1,5 Standardfehler: keine Warnung


def _loo(n=10, nsv=5, ls=3, lw=4, one=0, nonconv=0):
    z = np.zeros(n, bool)
    oc, nc = z.copy(), z.copy()
    oc[:one] = True
    nc[:nonconv] = True
    return {"n": n, "n_sv": nsv, "loo_solver": ls, "loo_worst": lw, "one_class": oc, "nonconv": nc}


def test_verdict_loo_schranke_haelt_oder_faellt():
    kind, text = ev.verdict_loo(_loo())
    assert kind == "info" and "ordnet das Herausnehmen 3 falsch ein (b des Lösers), bei 5 Stützvektoren: die Schranke LOO-Fehler ≤ Stützvektoren hält für dieses b" in text and "sind es 4: ebenfalls höchstens 5" in text
    kind, text = ev.verdict_loo(_loo(lw=6))
    assert kind == "warning" and "sind es 6: mehr als 5" in text
    kind, text = ev.verdict_loo(_loo(ls=6))
    assert kind == "warning" and "die Schranke wird verletzt" in text
    assert "2 Herausnahme(n) lassen nur eine Klasse übrig" in ev.verdict_loo(_loo(one=2))[1]
    assert "3 Neuberechnung(en) erreichten die Schrittgrenze 8000" in ev.verdict_loo(_loo(nonconv=3))[1] and ev.verdict_loo(_loo(nonconv=3))[0] == "warning"


def test_verdict_hand_nennt_den_knick_und_die_monotonie():
    rows = Ex.knick_tabelle([F(1, 8), F(1, 4), F(1, 3), F(1), F(100)])
    kind, text = ev.verdict_hand(rows)
    assert kind == "info" and "Ab C = 1/3 sind w = 2/3 und b = −5/3 in 3 von 5 Zeilen gleich" in text and "Summe der Schlupfwerte 8/3" in text
    assert "‖w‖ fällt in keiner Zeile" in text and "die Schlupfsumme steigt in keiner Zeile" in text and "der Dualwert fällt in keiner Zeile" in text


class _Res(dict):
    pass


def test_verdict_schritte_konvergiert_nicht_konvergiert_und_barriere():
    class M:
        steps, gap, converged, message = 49, 5e-4, True, ""

    class B:
        newton, outer = 80, 10

    kind, text = ev.verdict_schritte({"model": M, "barrier": B, "modell_newton": 8000.0, "modell_smo": 2.0})
    assert kind == "info" and "SMO: 49 Aktualisierungsschritte" in text and "80 Schritte in 10 äußeren Schritten" in text and "das 4000fache" in text

    class N(M):
        converged, message = False, "Schrittgrenze 49 erreicht, KKT-Lücke 0.1 über eps 0.001: nicht konvergiert"

    kind, text = ev.verdict_schritte({"model": N, "barrier": None})
    assert kind == "warning" and "nicht konvergiert" in text


PRESET_ERWARTUNG = {
    "Handbeispiel (fünf Punkte auf der Geraden)": ("Ab C = 1/3 sind w = 2/3", None),
    "Linearer Kern: C ist egal": ("Über C = 0,001 bis 1000", "info"),
    "Gauß-Kern: C zu groß": ("C wirkt hier stark", "warning"),
    "C zu klein (Mehrheitsklasse)": ("C wirkt hier stark", "warning"),
    "Ringe: C und Bandbreite": ("messbar, aber nicht stark", "info"),
    "Rauschetiketten": ("Ab C = 0,1 beträgt die Spanne höchstens", "warning"),
}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_je_preset_die_erwartete_verdikt_zeile_aus_der_eingefrorenen_kurve(name):
    p = {**{"data": C.DEFAULT_DATA, "kern": C.DEFAULT_KERN, "sigma": C.DEFAULT_SIGMA, "c": C.DEFAULT_C, "flip": C.DEFAULT_FLIP}, **C.PRESETS[name]}
    D = ev.studies()
    if p["data"] == C.DATASETS[2]:
        kind, text = ev.verdict_hand(ev.hand_table())
    else:
        case = ev.study_case(p["data"], p["kern"], p["sigma"], p["flip"])
        assert case is not None, f"Preset {name} hat keine eingefrorene Kurve"
        kind, text = ev.verdict_c(D["c_wirkung"]["C"], [St.mean_by_c(St.case_group(D, case, g)["test"]) for g in (1, 2)], ev.bayes_error(p["data"]))
    erwartet, art = PRESET_ERWARTUNG[name]
    assert erwartet in text
    if art:
        assert kind == art


def test_preset_ohne_effekt_meldet_keinen_effekt_linearer_kern_ohne_rauschen_ist_kein_warnung():
    D = ev.studies()
    kind, text = ev.verdict_c(D["c_wirkung"]["C"], [St.mean_by_c(St.case_group(D, "gauss_lin", g)["test"]) for g in (1, 2)], 0.1587)
    assert kind == "info" and "stark" not in text


def test_studienfall_zuordnung():
    assert ev.study_case("Gauß-Klassen", "linear", 1.0, 0.0) == "gauss_lin" and ev.study_case("Gauß-Klassen", "linear", 0.4, 0.1) == "gauss_lin_flip"
    assert ev.study_case("Gauß-Klassen", "Gauß", 1.0, 0.1) == "gauss_rbf_flip" and ev.study_case("Gauß-Klassen", "Gauß", 0.5, 0.0) is None
    assert ev.study_case("Ringe mit Streuung", "Gauß", 0.5, 0.0) == "ringe_rbf" and ev.study_case("Ringe mit Streuung", "linear", 0.5, 0.0) is None
    assert ev.study_case("Fünf Punkte auf der Geraden", "linear", 1.0, 0.0) is None
