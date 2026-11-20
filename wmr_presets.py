"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import wmr_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None  # erlaubte Werte (Auswahl, Optionsregler); sonst gelten lo und hi


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "data_select": SettingSpec("data", str, C.DEFAULT_DATA, choices=C.DATASETS),
    "flip_select": SettingSpec("flip", float, C.DEFAULT_FLIP, choices=C.FLIP_OPTIONS),
    "kern_select": SettingSpec("kern", str, C.DEFAULT_KERN, choices=C.KERN_OPTIONS),
    "sigma_select": SettingSpec("sigma", float, C.DEFAULT_SIGMA, choices=C.SIGMA_OPTIONS),
    "c_select": SettingSpec("c", float, C.DEFAULT_C, choices=C.C_OPTIONS),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "steps_select": SettingSpec("steps", int, C.DEFAULT_STEPS, choices=C.STEPS_OPTIONS),
    "eps_select": SettingSpec("eps", float, C.DEFAULT_EPS, choices=C.EPS_OPTIONS),
    "val_m_select": SettingSpec("m", int, C.DEFAULT_VAL_M, choices=C.VAL_M_OPTIONS),
    "group_select": SettingSpec("grp", int, C.DEFAULT_GROUP, choices=C.GROUP_OPTIONS),
    "loo_n_slider": SettingSpec("loon", int, C.DEFAULT_LOO_N, C.LOO_N_MIN, C.LOO_N_MAX),
    "e5_n_slider": SettingSpec("e5n", int, C.DEFAULT_E5_N, C.E5_N_MIN, C.E5_N_MAX),
    "wss_select": SettingSpec("wss", str, C.DEFAULT_WSS, choices=C.WSS_OPTIONS),
}
PRESET_KEYS = {
    "seed": "seed_input", "data": "data_select", "flip": "flip_select", "kern": "kern_select", "sigma": "sigma_select", "c": "c_select", "n": "n_slider", "steps": "steps_select",
    "eps": "eps_select", "m": "val_m_select", "group": "group_select", "loo_n": "loo_n_slider", "e5_n": "e5_n_slider", "wss": "wss_select",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clean(spec: SettingSpec, value):
    """Wert auf den erlaubten Bereich begrenzen (Permalink); ungültige Auswahl gibt den Standardwert."""
    if spec.choices is not None:
        return value if value in spec.choices else spec.default
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = clean(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt alle Regler: die im Preset genannten auf ihren Wert, alle übrigen auf den Standardwert."""
    preset = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = preset.get(key, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf
    (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (jeder Knopf setzt alle Regler; die Hilfe nennt das Experiment):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
