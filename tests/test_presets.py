"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung."""

import wmr_constants as C
import wmr_evaluation as ev
import wmr_presets as P


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP[name] for name in C.PRESETS)


def test_preset_keys_are_known_and_values_are_inside_the_regler_bounds():
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    p = C.PRESETS[first]
    for key, state_key in P.PRESET_KEYS.items():
        assert p.get(key, P.SETTING_SPECS[state_key].default) == P.SETTING_SPECS[state_key].default, key
    assert p["data"] == C.DEFAULT_DATA == C.DATASETS[2]


def test_setting_specs_cover_every_preset_key_and_every_default_is_allowed():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for state_key, spec in P.SETTING_SPECS.items():
        assert P.clean(spec, spec.default) == spec.default, state_key


def test_permalink_begrenzt_und_ungueltige_auswahl_gibt_den_standard():
    n = P.SETTING_SPECS["n_slider"]
    assert P.clean(n, 5) == C.N_MIN and P.clean(n, 5000) == C.N_MAX and P.clean(n, 77) == 77
    c = P.SETTING_SPECS["c_select"]
    assert P.clean(c, 10.0) == 10.0 and P.clean(c, 3.0) == C.DEFAULT_C
    assert P.clean(P.SETTING_SPECS["data_select"], "unbekannt") == C.DEFAULT_DATA


def test_jedes_preset_hat_eine_eingefrorene_kurve_oder_ist_das_handbeispiel():
    for name, preset in C.PRESETS.items():
        p = {"data": C.DEFAULT_DATA, "kern": C.DEFAULT_KERN, "sigma": C.DEFAULT_SIGMA, "flip": C.DEFAULT_FLIP, **preset}
        if p["data"] != C.DATASETS[2]:
            assert ev.study_case(p["data"], p["kern"], p["sigma"], p["flip"]) is not None, name
