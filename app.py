"""Weiche Marge: was C kauft und warum der Löser schweigt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünf Experimente: was C kauft (Marge, Schlupf, Stützvektoren), warum der Löser schweigt, wie man C wählt, ohne sich zu täuschen, Leave-one-out gegen Stützvektoren, Schritte zählen (SMO und Barriere).
Aufbau nach DEMO-PLAYBOOK §2. Alle Daten sind ausgedachte Modelle.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import wmr_constants as C
import wmr_evaluation as ev
import wmr_exact as Ex
import wmr_studies as St
import wmr_texts as T
from wmr_evaluation import fmt, fmt_c, fmt_g
from wmr_layout import dunkles_design, kopf_css
from wmr_presets import (
    aktives_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from wmr_visualization import C_GRID, S_GRID, fig_c_curve, fig_c_live, fig_heat, fig_line1d, fig_loo, fig_loo_scatter, fig_plane, fig_steps_c, fig_steps_n, fig_trace, fig_wahl

st.set_page_config(page_title="Weiche Marge und SMO – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _exp1(data, kern, sigma, c, n, seed, flip, eps, steps):
    return ev.exp1(data, kern, sigma, c, n, seed, flip, eps, steps)


@st.cache_data(show_spinner=False)
def _exp2(data, kern, sigma, n, seed, flip, eps, steps):
    return ev.exp2(data, kern, sigma, n, seed, flip, eps, steps)


@st.cache_data(show_spinner=False)
def _exp4(data, kern, sigma, c, n, seed, flip):
    return ev.exp4(data, kern, sigma, c, n, seed, flip)


@st.cache_data(show_spinner=False)
def _exp5(data, kern, sigma, c, n, seed, flip, eps, steps, wss):
    return ev.exp5(data, kern, sigma, c, n, seed, flip, eps, steps, wss)


@st.cache_data(show_spinner=False)
def _exakt_loo(i, einklassig):
    g = Ex.GEGENBEISPIELE[i]
    return Ex.loo_exakt(g["x"], g["y"], g["C"], einklassig)


def _flip_label(v):
    return f"{int(round(v * 100))} %"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in wmr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⚖️ Weiche Marge: was C kauft und warum der Löser schweigt")
st.markdown(T.INTRO)

with st.expander("So funktioniert die weiche Marge", expanded=True):
    st.markdown(T.HOW)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

hand = st.session_state.get("data_select", C.DEFAULT_DATA) == C.DATASETS[2]
linear = st.session_state.get("kern_select", C.DEFAULT_KERN) == "linear"

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🧪 Daten")
    data = st.selectbox("Datensatz", C.DATASETS, key="data_select", help="Gauß-Klassen (überlappend, Abstand 2), Ringe um ein Depot (Radiusstreuung 0,35) oder das Handbeispiel mit fünf festen Punkten.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1, disabled=hand,
                           help="Legt die Trainingspunkte und die 2 000 frischen Testpunkte fest. Beim Handbeispiel gibt es keinen Zufall.")
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, disabled=hand)
    n = st.slider("Zahl der Trainingspunkte n", *bounds("n_slider"), key="n_slider", disabled=hand, help="Live bis 200; die eingefrorenen Studien rechnen mit n = 60 bzw. 200.")
    flip = st.radio("Vertauschte Labels im Training", C.FLIP_OPTIONS, key="flip_select", horizontal=True, format_func=_flip_label, disabled=hand,
                    help="Anteil der Trainings-Labels, der umgekehrt wird (Rauschen). Der Test bleibt sauber.")
    st.header("🔀 Modell")
    kern = st.radio("Kern", C.KERN_OPTIONS, key="kern_select", horizontal=True, help="Linear oder Gauß mit Bandbreite σ (Längenmaß, γ = 1/(2σ²)).")
    sigma = st.select_slider("Bandbreite σ des Gauß-Kerns", options=C.SIGMA_OPTIONS, key="sigma_select", format_func=lambda v: fmt_g(v, 2), disabled=linear,
                             help="Nur beim Gauß-Kern wirksam. Die eingefrorenen Kurven gibt es für σ = 1 (Gauß-Klassen) und σ = 0,5 (Ringe).")
    c = st.select_slider("C (Preis je Verletzung)", options=C.C_OPTIONS, key="c_select", format_func=fmt_c, help="Obergrenze der Multiplikatoren α und Preis je Einheit Schlupf; logarithmische Stufen von 1e-3 bis 1e3.")
    st.header("🧮 Löser")
    steps = st.select_slider("Schrittgrenze des SMO", options=C.STEPS_OPTIONS, key="steps_select", help="Bei Erreichen hält das SMO an und sagt es („nicht konvergiert“); die App rechnet nie ohne Grenze.")
    eps = st.select_slider("Toleranz eps (KKT-Lücke)", options=C.EPS_OPTIONS, key="eps_select", format_func=lambda v: fmt_g(v, 1), help="Abbruch bei KKT-Lücke ≤ eps; 1e-3 ist die Voreinstellung von libsvm.")
    st.header("🎯 Experiment 3: Wahl von C")
    val_m = st.select_slider("Größe der Validierungsmenge m", options=C.VAL_M_OPTIONS, key="val_m_select", help="Die Gewinnerauswahl auf m frisch gezogenen Punkten (eingefrorene Studie).")
    group = st.radio("Zufalls-Seed-Gruppe der Studie", C.GROUP_OPTIONS, key="group_select", horizontal=True, help="Zwei unabhängige Gruppen zu je 30 Zufalls-Seeds: so sieht man, wie weit die Zahlen streuen.")
    st.header("🔁 Experiment 4: Leave-one-out")
    loo_n = st.slider("Punkte beim Herausnehmen", *bounds("loo_n_slider"), key="loo_n_slider", disabled=hand, help="Je Punkt eine Neuberechnung; beim Handbeispiel sind es immer die fünf Punkte.")
    st.header("📈 Experiment 5: Schritte")
    e5_n = st.slider("Punkte für Schritte und Barriere", *bounds("e5_n_slider"), key="e5_n_slider", disabled=hand, help="Die Barriere löst n+1 Gleichungen je Newton-Schritt; bis 320 live.")
    wss = st.selectbox("Arbeitsmenge des SMO", C.WSS_OPTIONS, key="wss_select", help="Zweite Ordnung (wie libsvm), erstes verletzendes Paar oder zufälliger zweiter Partner (mit der Zufallsquelle der Demo).")

sync_query_params({k: st.session_state[k] for k in (
    "seed_input", "data_select", "flip_select", "kern_select", "sigma_select", "c_select", "n_slider", "steps_select", "eps_select", "val_m_select", "group_select", "loo_n_slider", "e5_n_slider",
    "wss_select")})

seed, n, steps, loo_n, e5_n, group, val_m = int(seed), int(n), int(steps), int(loo_n), int(e5_n), int(group), int(val_m)
sigma, c, eps, flip = float(sigma), float(c), float(eps), float(flip)

aktiv = aktives_preset()
if aktiv:
    st.info(f"Preset „{aktiv}“: {C.PRESET_HELP.get(aktiv, '')}")

D = ev.studies()
r1 = _exp1(data, kern, sigma, c, n, seed, flip, eps, steps)
bayes = r1["bayes"]

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------------
st.markdown("## ❓ Kernfragen")
st.markdown("\n".join(f"* {q}" for q in T.KERNFRAGEN))
g = r1["groups"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Marge 2/‖w‖ (Exp. 1)", fmt(r1["margin"], 3) if np.isfinite(r1["margin"]) else "w = 0", delta=f"Summe Schlupf {fmt(r1['slack_sum'], 2)}", delta_color="off")
m2.metric("Fehler auf frischen Punkten" if not r1["hand"] else "Trainingsfehler", fmt(r1["test_err"] if not r1["hand"] else r1["train_err"], 4),
          delta=(f"Bayes-Fehler {fmt(bayes, 4)}" if bayes is not None else "kein Bayes-Fehler"), delta_color="off")
m3.metric("Gruppen α = 0 / auf der Marge / α = C", f"{g['aussen']} / {g['marge']} / {g['schranke']}", delta=f"von {r1['inst'].n} Punkten", delta_color="off")
m4.metric("KKT-Lücke des Lösers", fmt_g(r1["gap"], 3), delta=("konvergiert" if r1["model"].converged else "nicht konvergiert"), delta_color="off")

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 1: Was kauft C?")
st.caption(f"{r1['inst'].name}, n = {r1['inst'].n}, {'Kern ' + kern + (f' σ = {fmt_g(sigma, 2)}' if not linear else '')}, C = {fmt_c(c)}, Zufalls-Seed {seed}; Zähleinheiten Zufalls-Seeds (eingefroren) und Instanzen (live). "
           "Schwarze Höhenlinien: f = 0 und f = ±1; α = 0 gewöhnliche Punkte, 0 < α < C schwarz umkreist (auf der Marge), α = C golden umkreist (Verletzer oder auf der Marge).")
cA, cB = st.columns([3, 2])
with cA:
    if r1["hand"]:
        st.plotly_chart(fig_line1d(r1["model"]), width="stretch", key="chart_exp1")
    else:
        st.plotly_chart(fig_plane(r1["model"], title=f"Live: f = 0 und f = ±1, C = {fmt_c(c)}"), width="stretch", key="chart_exp1")
with cB:
    case = r1["case"]
    if r1["hand"]:
        rows_h = ev.hand_table()
        st.markdown("**Handbeispiel, aus der Aufzählung aller 243 Zuordnungen in Brüchen** (nicht aus dem Löser):")
        st.dataframe(pd.DataFrame([{
            "C": Ex.bruch(r["C"]), "α₂ … α₅": ", ".join(Ex.bruch(a) for a in r["alpha"][1:]), "w": Ex.bruch(r["w"]),
            "b": Ex.bruch(r["b_lo"]) if r["b_lo"] == r["b_hi"] else f"[{Ex.bruch(r['b_lo'])}; {Ex.bruch(r['b_hi'])}]", "Dualwert": Ex.bruch(r["dual"]), "Σ ξ": Ex.bruch(r["xi_sum"]),
            "SMO-Schritte": r["smo_steps"],
        } for r in rows_h]), hide_index=True, width="stretch")
        st.caption("α₁ = 0 in jeder Zeile. Primalwert und Dualwert sind in jeder Zeile exakt gleich (auch an beiden Enden des b-Intervalls). SMO-Schritte: eigenes SMO (zweite Ordnung, eps = 1e-9) auf denselben Punkten; "
                   "diese Zahlen sind nicht von Hand gerechnet.")
    elif case and D:
        st.plotly_chart(fig_c_curve(D, case, bayes), width="stretch", key="chart_curve")
    else:
        st.info("Für diese Kombination aus Datensatz, Kern und Bandbreite gibt es keine eingefrorene Kurve; eingefroren sind Gauß-Klassen (linear und Gauß σ = 1) und Ringe (Gauß σ = 0,5), je ohne und mit 10 % vertauschten Labels.")
if r1["hand"]:
    kind, text = ev.verdict_hand(ev.hand_table())
elif case and D:
    cs = D["c_wirkung"]["C"]
    kind, text = ev.verdict_c(cs, [St.mean_by_c(St.case_group(D, case, gg)["test"]) for gg in (1, 2)], bayes, label=f"Eingefroren: {r1['inst'].name}, {kern}{'' if linear else ' σ = ' + fmt_g(sigma, 2)}, {_flip_label(flip)} vertauscht")
else:
    kind, text = "info", "Keine eingefrorene Kurve für diese Einstellung."
getattr(st, kind)(text)
if not r1["model"].converged:
    st.warning(f"Das eigene SMO: {r1['model'].message}.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 2: Warnt der Löser, wenn C falsch ist?")
st.caption(f"Dieselbe Instanz mit allen C der Auswahl, eigenes SMO, eps = {fmt_g(eps, 1)}, Schrittgrenze {steps}; Zähleinheiten Zellen und Aktualisierungsschritte. "
           "Die Spalten „konvergiert“ und „Meldung“ zeigen, was der Löser sagt; die Spalte Fehler, was das Modell taugt. Die Falle „nicht trennbar, C als Ersatz der harten Marge“ führt die Demo `harte-marge-demo`.")
with st.spinner(f"Das eigene SMO rechnet (bis zu {steps} Aktualisierungsschritte je C) …"):
    r2 = _exp2(data, kern, sigma, n, seed, flip, eps, steps)
for row in r2["rows"]:
    row["eps"] = eps
df2 = pd.DataFrame([{
    "C": fmt_c(r["c"]), "KKT-Lücke": fmt_g(r["gap"], 3), "konvergiert (Lücke ≤ eps)": "✅ ja" if r["converged"] else "⛔ nein",
    "Meldung des Lösers": r["message"] or "keine", "Trainingsfehler" if r2["hand"] else "Testfehler": fmt(r["err"], 4), "Schritte": r["steps"], "Stützvektoren (α > 0)": fmt(r["sv_frac"] * 100, 0) + " %",
} for r in r2["rows"]])
c2a, c2b = st.columns([3, 2])
with c2a:
    st.dataframe(df2, hide_index=True, width="stretch")
with c2b:
    st.plotly_chart(fig_c_live(r2["rows"], r2["hand"]), width="stretch", key="chart_exp2")
kind, text = ev.verdict_loeser(r2["rows"], r2["hand"])
getattr(st, kind)(text)
st.markdown(T.loeser_studie_md())
st.caption(T.live_vs_studie())

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 3: Wie wählt man C, ohne sich zu täuschen?")
st.caption("Ringe mit Streuung (Radiusstreuung 0,35), Gauß-Kern, Raster 6 C × 6 σ; eingefroren, Zähleinheit Zufalls-Seeds (30 je Gruppe). „Optimistisch“ ist der kleinste Fehler auf einer kleinen, frisch gezogenen "
           "Validierungsmenge, „ehrlich“ der Fehler derselben Zelle auf 2 000 frischen Punkten. Gemessen ist die Gewinnerauswahl auf einer Validierungsmenge, nicht die Wahl auf einer Testmenge.")
if D:
    vals = [St.validierung(D, group, m) for m in C.VAL_M_OPTIONS]
    w60 = St.wahl(D, "wahl60", group, "flip0")
    p200 = St.wahl_pooled(D, "wahl200", "flip0")
    c3a, c3b = st.columns([2, 3])
    with c3a:
        st.plotly_chart(fig_wahl(vals, f"Optimismus der Gewinnerauswahl (Gruppe {group}, ohne Rauschen)"), width="stretch", key="chart_wahl")
    with c3b:
        st.plotly_chart(fig_heat(w60["mean_te"], f"Mittlerer Testfehler über dem Raster (Gruppe {group}, n = 60, ohne Rauschen)"), width="stretch", key="chart_heat")
    rows3 = []
    for gg in (1, 2):
        for fk, fl in (("flip0", "ohne Rauschen"), ("flip10", "10 % vertauscht")):
            w = St.wahl(D, "wahl60", gg, fk)
            rows3.append({"Gruppe": gg, "Labels": fl, "im Rückblick bestes Feld": fmt(w["rueckblick"], 4), "5-fache Kreuzvalidierung": fmt(w["cv"], 4), "Voreinstellung": fmt(w["default"], 4),
                          "Differenz CV − Voreinstellung (Standardfehler)": f"{fmt(w['diff_mean'], 4)} ({fmt(w['diff_se'], 4)})", "CV besser / schlechter": f"{w['cv_besser']} / {w['cv_schlechter']} von {w['n_seeds']}"})
    st.markdown("**n = 60, 30 Zufalls-Seeds je Gruppe** (Fehler auf 2 000 frischen Testpunkten; die Voreinstellung ist C = 1 und γ = 1/(p·Var X))")
    st.dataframe(pd.DataFrame(rows3), hide_index=True, width="stretch")
    rows3b = []
    for fk, fl in (("flip0", "ohne Rauschen"), ("flip10", "10 % vertauscht")):
        p = St.wahl_pooled(D, "wahl200", fk)
        rows3b.append({"Labels": fl, "Zufalls-Seeds": p["n_seeds"], "Differenz CV − Voreinstellung (Standardfehler)": f"{fmt(p['mean'], 4)} ({fmt(p['se'], 4)})",
                       "je Gruppe zu 20": " / ".join(fmt(v, 4) for v in p["je_gruppe"]), "CV besser / schlechter / gleich": f"{p['besser']} / {p['schlechter']} / {p['gleich']}"})
    st.markdown("**n = 200, drei Gruppen zu je 20 Zufalls-Seeds**")
    st.dataframe(pd.DataFrame(rows3b), hide_index=True, width="stretch")
    kind, text = ev.verdict_wahl(St.validierung(D, group, val_m), w60, p200)
    getattr(st, kind)(text)
    spans = St.sigma_zeile(D, group, "flip0")
    rd = St.sigma_raender(w60["mean_te"])
    st.caption("Spanne des mittleren Testfehlers über C je Bandbreite σ = " + ", ".join(f"{fmt_g(s_, 2)}: {fmt(sp, 4)}" for s_, sp in zip(S_GRID, spans))
               + f" (Gruppe {group}, ohne Rauschen). Die gute Lage ist eine Lage im Raster aus C und σ zusammen: bei σ = 0,1 liegt der Fehler bei jedem C bei mindestens {fmt(rd['schmal_min'], 4)}; bei σ = 3,2 liegt er bei C = 0,01 und 0,1 "
               f"zwischen {fmt(rd['breit_klein_c_min'], 4)} und {fmt(rd['breit_klein_c_max'], 4)}, ab C = 10 bei höchstens {fmt(rd['breit_gross_c_max'], 4)}.")

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 4: Leave-one-out gegen Stützvektoren")
st.caption("Jeder Punkt wird wirklich herausgenommen und der Rest neu berechnet (Zähleinheit Instanzen). Verglichen wird die Zahl der falsch eingeordneten Punkte mit der Zahl der Stützvektoren (α > 0). "
           "Das b ist bei α = 0 oder α = C aller Punkte nur ein Intervall: gezählt wird mit dem b des Lösers und mit dem ungünstigsten b des optimalen Intervalls. Die Schranke ist hier eine Zählung mit eigener "
           "Herleitung; bei Cortes/Vapnik (1995, Gl. 5) ist es eine Erwartungswert-Aussage für trennbare Daten.")
with st.spinner(f"Neuberechnung ohne jeden der {5 if hand else loo_n} Punkte …"):
    r4 = _exp4(data, kern, sigma, c, loo_n, seed, flip)
c4a, c4b = st.columns([2, 3])
with c4a:
    if r4["inst"].X.shape[1] == 2:
        st.plotly_chart(fig_loo(r4), width="stretch", key="chart_loo")
    else:
        st.dataframe(pd.DataFrame({"Punkt": [int(x) for x in r4["inst"].X[:, 0]], "Label": [int(v) for v in r4["inst"].y], "Stützvektor": ["ja" if v else "nein" for v in r4["was_sv"]],
                                   "falsch (b des Lösers)": ["ja" if v else "nein" for v in r4["err_solver"]], "falsch (ungünstigstes b)": ["ja" if v else "nein" for v in r4["err_worst"]]}),
                     hide_index=True, width="stretch")
with c4b:
    if D:
        st.plotly_chart(fig_loo_scatter(D), width="stretch", key="chart_loo_scatter")
kind, text = ev.verdict_loo(r4)
getattr(st, kind)(text)
if D:
    e1, e5 = St.loo_e1(D), St.loo_e5(D)
    st.markdown(
        f"**Studie (eingefroren, libsvm `SVC`, Zähleinheit Instanzen).** {e1['n']} Instanzen (n = 12 bis 30; Gauß-Klassen linear und Gauß mit und ohne 10 % vertauschten Labels, Ringe, getrennte Klassen; "
        f"C aus 0,01 / 0,1 / 1 / 10 / 1000): mit dem b des Lösers {e1['verletzungen']} Verletzungen von „LOO-Fehler ≤ Zahl der Stützvektoren“, scharf (Gleichheit) in {e1['scharf']}, Median des Verhältnisses {fmt(e1['ratio_median'], 3)}, "
        f"Maximum {fmt(e1['ratio_max'], 3)}; {e1['degeneriert']} Instanzen ohne freien Stützvektor und mit Punkten α = 0 (b nicht eindeutig), darin {e1['degeneriert_verletzt']} Verletzungen. "
        f"Gezielte Suche ({e5['n']} kleine Instanzen, n = 6 bis 10, C von 0,02 bis 0,5, beide Klassen mindestens zweimal): mit dem b des Lösers {e5['verl_solver']} Verletzungen, **mit dem ungünstigsten b des "
        f"optimalen Intervalls {e5['verl_ungunstig']}** ({e5['verl_ungunstig_strikt']} auch ohne die Zählung von f = 0 als Fehler; einklassige Reste: {e5['einklassig']}); scharf in {e5['scharf_ungunstig']}."
    )
    ex_rows = []
    for i, gb in enumerate(Ex.GEGENBEISPIELE):
        a = _exakt_loo(i, True)
        b2 = _exakt_loo(i, False)
        ex_rows.append({"Beispiel": gb["name"], "x": ", ".join(Ex.bruch(x) for x in gb["x"]), "y": ", ".join(f"{v:+d}" for v in gb["y"]), "C": Ex.bruch(gb["C"]), "Stützvektoren": a["n_sv"],
                        "LOO ungünstigstes b": a["worst"], "LOO b in der Mitte": a["mitte"], "einklassige Reste": a["einklassig"],
                        "LOO ungünstigstes b ohne Zählung des einklassigen Rests": b2["worst"]})
    st.markdown("**Exakte Gegenbeispiele (Brüche, Aufzählung):** die Schranke fällt bei ungünstigstem b. Das Beispiel mit drei Punkten verletzt sie nur, wenn der Rest mit einer Klasse als Fehler zählt (2 gegen 2 ohne diese Zählung); das mit fünf Punkten braucht diese Zählung nicht.")
    st.dataframe(pd.DataFrame(ex_rows), hide_index=True, width="stretch")

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 5: Wie viele Schritte braucht der Löser?")
st.caption("Zähleinheiten Aktualisierungsschritte (SMO: zwei Multiplikatoren je Schritt) und Newton-Schritte (Barriere); keine Sekunden. Das SMO selbst erklärt `ocsvm-demo` (anderes QP: Σα = νn statt Σ yα = 0); "
           "hier wird es gezählt. Die Zählung der exakten Rechnung in n (n³ gegen n²) führt `gram-naeherung-demo`; die Modellzeile unten ist nur eine Zeile mit Verweis darauf.")
with st.spinner("SMO mit Spur und Barriere rechnen …"):
    r5 = _exp5(data, kern, sigma, c, e5_n, seed, flip, eps, steps, C.WSS_CODE[wss])
c5a, c5b = st.columns(2)
with c5a:
    tr = r5["model"].trace
    if tr is not None and len(tr["step"]) > 1:
        st.plotly_chart(fig_trace(tr, eps), width="stretch", key="chart_trace")
with c5b:
    if D:
        st.plotly_chart(fig_steps_c(D), width="stretch", key="chart_steps_c")
kind, text = ev.verdict_schritte(r5)
getattr(st, kind)(text)
if D:
    c5c, c5d = st.columns(2)
    sn, sc, bs = St.steps_n(D), {k: St.steps_c(D, k) for k in ("gauss", "ringe")}, St.barrier(D)
    with c5c:
        st.plotly_chart(fig_steps_n(D), width="stretch", key="chart_steps_n")
    with c5d:
        st.dataframe(pd.DataFrame({"n": bs["n"], "Newton-Schritte (Mittel)": [fmt(v, 0) for v in bs["newton_mean"]], "SMO-Schritte (Mittel)": [fmt(v, 0) for v in bs["smo_mean"]],
                                   "Modell n³/3 · Newton ÷ (n · SMO)": [fmt_g(v, 3) for v in bs["modell"]],
                                   "scipy trust-constr (Iterationen)": [("–" if v is None else str(v)) for v in bs["trust"]]}), hide_index=True, width="stretch")
        st.caption("Barriere: Startpunkt strikt innen, t₀ = 1, Faktor μ = 10, Abbruch bei Lücke 2n/t ≤ 1e-6, Newton-Abbruch Dekrement²/2 ≤ 1e-10, Liniensuche mit 0,99-fachem Weg bis zum Rand und Armijo-Faktor 0,25; "
                   f"3 Zufalls-Seeds je n, Gauß-Klassen, Gauß-Kern σ = 1, C = 1. Einzelwerte {bs['newton_min']} bis {bs['newton_max']} Newton-Schritte; bei n = 80 je nach Einstellung (μ = 2, 10, 50; Abbruchlücke 1e-4, 1e-6, 1e-8) "
                   f"{fmt(bs['abh_min'], 0)} bis {fmt(bs['abh_max'], 0)}. `trust-constr` zählt Iterationen anderer Einheit und trifft den Dualwert nur auf {fmt_g(bs['trust_dual_rel_max'], 2)} relativ; nur als Band.")
    st.markdown(
        f"**Studie (eingefroren, Zähleinheit Aktualisierungsschritte des eigenen SMO).** n = 200, Gauß-Klassen mit 5 % vertauschten Labels, Gauß-Kern σ = 1: "
        f"{', '.join(fmt(v, 0) for v in sc['gauss']['own_mean'])} Schritte bei C = 0,01 / 0,1 / 1 / 10 / 100 / 1000 (Mittel über 5 Zufalls-Seeds; bei C = 1000 je Zufalls-Seed {sc['gauss']['own_min'][-1]} bis {sc['gauss']['own_max'][-1]}). "
        f"Gegen n (C = 1, 20 Zufalls-Seeds, n = 40 bis 1280): Steigung von log(Schritte) gegen log(n) {fmt(sn['slope_all'], 2)} (Bootstrap-90-%-Band [{fmt(sn['band'][0], 2)}; {fmt(sn['band'][1], 2)}] über die 20 Zufalls-Seeds; "
        f"{fmt(sn['slope_a'], 2)} und {fmt(sn['slope_b'], 2)} in zwei Gruppen zu 10; libsvm `n_iter_` {fmt(sn['slope_lib'], 2)}); bei n = 640 zwischen {sn['min'][4]} und {sn['max'][4]} Schritten je Zufalls-Seed. "
        "Das Band der Steigung schließt 1 nicht ein: in diesen Daten (diese Instanzfamilie, C = 1, n = 40 bis 1280) wächst die Schrittzahl schneller als linear, und der Aufwand hängt stark an C; über n = 1280 hinaus ist nichts gesagt. Die Schrittzahl sagt nichts über Sekunden."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.assumptions_md())

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.FORMULAS)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html)."
)
