"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt sicher, dass wirklich JEDER Regler der Seitenleiste
dort steht: sonst bliebe das Preset trotz Änderung markiert. Auswahlregler und Optionsregler bekommen einen anderen erlaubten Wert, Zahlregler den Rand."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import wmr_constants as C
import wmr_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel  # und jeder Eintrag ist wirklich ein Regler der Seitenleiste


# Regler, die im Start-Preset (Handbeispiel) bzw. bei linearem Kern ausgegraut sind: ein Browser-Nutzer kann sie dort nicht bedienen (AppTest verbietet es ebenfalls).
# Der Test lädt zuerst ein Preset, in dem der Regler wirksam ist, und ändert ihn dann.
PRESET_MIT_WIRKSAMEM_REGLER = {"sigma_select": "Gauß-Kern: C zu groß"}
STANDARD_AUSWEICH_PRESET = "Linearer Kern: C ist egal"


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    name = next(iter(C.PRESETS))
    regler = _regler(at, state_key)
    if regler.proto.disabled:
        name = PRESET_MIT_WIRKSAMEM_REGLER.get(state_key, STANDARD_AUSWEICH_PRESET)
        at.sidebar.button(key=f"preset_{name}").click().run()
        assert _typen(at)[name] == "primary"
        regler = _regler(at, state_key)
        assert not regler.proto.disabled, state_key
    if spec.choices is not None:
        neu = next(c for c in spec.choices if c != regler.value)
    else:
        neu = spec.hi if regler.value != spec.hi else spec.lo
    regler.set_value(neu).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert _typen(at)[name] != "primary"  # das Preset ist nicht mehr markiert (ein anderes darf es sein, wenn der neue Wert genau dessen Einstellung ist)
    # erneutes Anklicken markiert wieder
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def test_ausgegraute_regler_sind_im_handbeispiel_und_beim_linearen_kern_genau_die_unwirksamen():
    at = _app()  # Start: Handbeispiel, linearer Kern
    aus = {k for k in P.SETTING_SPECS if _regler(at, k).proto.disabled}
    assert aus == {"seed_input", "flip_select", "n_slider", "loo_n_slider", "e5_n_slider", "sigma_select"}
    at.sidebar.button(key="preset_Gauß-Kern: C zu groß").click().run()
    assert {k for k in P.SETTING_SPECS if _regler(at, k).proto.disabled} == set()
