"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        if at.button[index].proto.disabled:  # der Zufalls-Seed-Knopf ist beim Handbeispiel ausgegraut (dort gibt es keinen Zufall)
            continue
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_der_zufalls_seed_knopf_wirkt_wenn_er_wirksam_ist():
    at = _app_mit_wirksamen_reglern()
    alt = at.sidebar.number_input(key="seed_input").value
    knopf = next(b for b in at.sidebar.button if "Neue Instanz" in b.label)
    knopf.click().run(timeout=TIMEOUT)
    assert not at.exception and at.sidebar.number_input(key="seed_input").value != alt


def _app_mit_wirksamen_reglern() -> AppTest:
    """Im Start-Preset (Handbeispiel) sind Zufalls-Seed, n, Vertauschung und die Punktzahlen ausgegraut; ein Preset mit Gauß-Klassen macht sie wirksam."""
    at = _fresh_app()
    at.sidebar.button(key="preset_Linearer Kern: C ist egal").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _app_mit_wirksamen_reglern().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _app_mit_wirksamen_reglern()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])
