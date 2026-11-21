"""Plotly-Figuren: alle Achsen fest (`fixedrange`), Beschriftungen als Annotationen, gleicher Maßstab auf beiden Achsen über `scaleanchor` mit automatischem Bereich."""

import numpy as np
import plotly.graph_objects as go

import wmr_constants as C
import wmr_evaluation as ev
import wmr_studies as S


def _box(X, pad: float = 0.25):
    pts = np.asarray(X, float)
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-9)
    return lo - pad * span, hi + pad * span


def _plane_layout(fig, lo, hi, title=None, height=430):
    fig.add_trace(go.Scatter(x=[lo[0], hi[0]], y=[lo[1], hi[1]], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36 if title else 10, b=10), title=dict(text=title, font=dict(size=14)) if title else None,
                      legend=dict(orientation="h", y=-0.12), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True, scaleanchor="y", scaleratio=1, showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def _style(fig, title, height=320, xtitle=None, ytitle=None, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36, b=10), title=dict(text=title, font=dict(size=14)), legend=dict(orientation="h", y=legend_y),
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True, title=xtitle, showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(fixedrange=True, title=ytitle, showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    return fig


def _groups_of(model):
    t = ev._tol_alpha(model.c)
    a = model.alpha
    out = a <= t
    bound = a >= model.c - t
    free = ~out & ~bound
    return out, free, bound


def fig_plane(model, title=None, height=430, grid_n: int = 100):
    """Ebene mit Entscheidungsfunktion f = 0 und den Höhenlinien f = ±1; Punkte in drei Gruppen: α = 0 (klein), 0 < α < C (schwarz umkreist, auf der Marge), α = C (golden umkreist, in oder jenseits der Marge)."""
    X, y = model.X, model.y
    lo, hi = _box(X, pad=0.3)
    gx = np.linspace(lo[0], hi[0], grid_n)
    gy = np.linspace(lo[1], hi[1], grid_n)
    G = np.array([(a, c) for c in gy for a in gx])
    f = model.decision(G).reshape(grid_n, grid_n)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=gx, y=gy, z=f, showscale=False, contours=dict(start=-1, end=1, size=1, coloring="none"), line=dict(color=C.COLOR_LINE, width=1.6),
                             hoverinfo="skip", showlegend=False))
    for lab, col, nm in ((1.0, C.COLOR_POS, "Klasse +1"), (-1.0, C.COLOR_NEG, "Klasse −1")):
        m = y == lab
        fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers", marker=dict(color=col, size=8, line=dict(width=1, color="white")), name=nm,
                                 text=[f"Punkt {i + 1}" for i in np.where(m)[0]], hovertemplate="%{text}<br>(%{x:.2f}; %{y:.2f})<extra></extra>"))
    out, free, bound = _groups_of(model)
    if free.any():
        fig.add_trace(go.Scatter(x=X[free, 0], y=X[free, 1], mode="markers", marker=dict(symbol="circle-open", size=17, line=dict(width=2.2, color=C.COLOR_LINE)),
                                 name="0 < α < C (auf der Marge)", hoverinfo="skip"))
    if bound.any():
        fig.add_trace(go.Scatter(x=X[bound, 0], y=X[bound, 1], mode="markers", marker=dict(symbol="circle-open", size=13, line=dict(width=2.2, color=C.COLOR_BOUND)),
                                 name="α = C (Verletzer oder auf der Marge)", hoverinfo="skip"))
    return _plane_layout(fig, lo, hi, title, height)


def fig_line1d(model, title=None, height=330):
    """Handbeispiel (Punkte auf einer Geraden): y f(x) je Punkt gegen die Lage x; die Marge verlangt y f ≥ 1, darunter liegt der Schlupf ξ."""
    X = model.X[:, 0]
    y = model.y
    f = model.K @ (model.alpha * y) + model.b
    yf = y * f
    xs = np.linspace(X.min() - 0.4, X.max() + 0.4, 100)
    fg = model.decision(xs[:, None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=fg, mode="lines", line=dict(color=C.COLOR_LINE, width=2), name="f(x)"))
    for lev in (-1.0, 1.0):
        fig.add_trace(go.Scatter(x=[xs[0], xs[-1]], y=[lev, lev], mode="lines", line=dict(color=C.COLOR_MARGIN, dash="dash", width=1.2), showlegend=False, hoverinfo="skip"))
    for lab, col, nm in ((1.0, C.COLOR_POS, "Klasse +1"), (-1.0, C.COLOR_NEG, "Klasse −1")):
        m = y == lab
        fig.add_trace(go.Scatter(x=X[m], y=f[m], mode="markers", marker=dict(color=col, size=11, line=dict(width=1, color="white")), name=nm,
                                 hovertemplate="x = %{x}<br>f = %{y:.3f}<extra></extra>"))
    for i in range(len(X)):
        fig.add_annotation(x=X[i], y=f[i], text=f"ξ = {ev.fmt(max(0.0, 1 - yf[i]), 2)}", showarrow=False, yshift=-18, font=dict(size=10))
    return _style(fig, title or "Handbeispiel: f(x) und die Randlinien f = ±1; unter dem Label steht der Schlupf ξ", height, "Lage x", "f(x)")


def fig_c_curve(d: dict, key: str, bayes, group_colors=("#2F6B65", "#4c78a8"), title=None):
    """Mittlerer Testfehler gegen C (Zufalls-Seeds je Gruppe, Streifen = Mittel ± Standardfehler), Bayes-Fehler als Linie."""
    cs = d["c_wirkung"]["C"]
    fig = go.Figure()
    for g, col in zip((1, 2), group_colors):
        a = np.array(S.case_group(d, key, g)["test"])
        mean = a.mean(axis=1)
        sem = np.array([S.se(v) for v in a])
        fig.add_trace(go.Scatter(x=cs + cs[::-1], y=list(mean + sem) + list((mean - sem)[::-1]), fill="toself", fillcolor="rgba(120,120,120,0.15)", line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=cs, y=mean, mode="lines+markers", line=dict(color=col, width=2), name=f"Zufalls-Seed-Gruppe {g} (30)"))
    if bayes is not None:
        fig.add_trace(go.Scatter(x=[cs[0], cs[-1]], y=[bayes, bayes], mode="lines", line=dict(color=C.COLOR_MARGIN, dash="dash"), name=f"Bayes-Fehler {ev.fmt(bayes, 4)}"))
    fig = _style(fig, title or "Testfehler gegen C (eingefroren, 30 Zufalls-Seeds je Gruppe, n = 60)", 330, "C", "mittlerer Testfehler")
    fig.update_xaxes(type="log")
    return fig


def fig_c_live(rows, hand=False):
    """Live-Zeile: Fehler (Test, beim Handbeispiel Training) und KKT-Lücke je C für die gewählte Instanz."""
    cs = [r["c"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cs, y=[r["err"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_NEG, width=2), name="Trainingsfehler" if hand else "Testfehler (2 000 Punkte)"))
    fig.add_trace(go.Scatter(x=cs, y=[max(r["gap"], 1e-12) for r in rows], mode="lines+markers", line=dict(color=C.COLOR_ALT, width=2), name="KKT-Lücke des Lösers", yaxis="y2"))
    fig = _style(fig, "Live: Fehler und KKT-Lücke je C (dieselbe Instanz)", 300, "C", "Fehler")
    fig.update_xaxes(type="log")
    fig.update_layout(yaxis2=dict(title="KKT-Lücke", overlaying="y", side="right", type="log", fixedrange=True))
    return fig


def fig_heat(mean_te, title):
    """Mittlerer Testfehler über dem Raster 6 C × 6 σ (Ringe mit Streuung, eingefroren)."""
    z = np.asarray(mean_te)
    fig = go.Figure(go.Heatmap(z=z, x=[f"σ = {ev.fmt_g(s, 2)}" for s in S_GRID], y=[f"C = {ev.fmt_g(c, 3)}" for c in C_GRID], colorscale="Viridis", reversescale=True, zmin=0.05, zmax=0.45,
                               text=[[ev.fmt(v, 3) for v in row] for row in z], texttemplate="%{text}", showscale=False, hoverinfo="skip"))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=36, b=10), title=dict(text=title, font=dict(size=14)))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True, autorange="reversed")
    return fig


C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
S_GRID = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]


def fig_wahl(rows, title):
    """Gruppierte Balken: optimistischer Fehler (kleinster Fehler auf der Validierungsmenge) gegen ehrlichen Fehler (gleiche Zelle, frische Punkte) je Größe m."""
    ms = [f"m = {r['m']}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ms, y=[r["optimistisch"] for r in rows], name="optimistisch (kleinster Fehler auf der Validierungsmenge)", marker_color=C.COLOR_ALT))
    fig.add_trace(go.Bar(x=ms, y=[r["ehrlich"] for r in rows], name="ehrlich (dieselbe Zelle auf frischen Punkten)", marker_color=C.COLOR_NEG))
    fig = _style(fig, title, 300, "Größe der Validierungsmenge", "mittlerer Fehler")
    fig.update_layout(barmode="group")
    return fig


def fig_loo(res, title=None):
    """Leave-one-out je Punkt: falsch eingeordnet (Kreuz) und Stützvektor (schwarzer Ring)."""
    inst = res["inst"]
    X, y = inst.X, inst.y
    lo, hi = _box(X, pad=0.3)
    fig = go.Figure()
    for lab, col, nm in ((1.0, C.COLOR_POS, "Klasse +1"), (-1.0, C.COLOR_NEG, "Klasse −1")):
        m = y == lab
        fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers", marker=dict(color=col, size=8, line=dict(width=1, color="white")), name=nm, hoverinfo="skip"))
    sv = res["was_sv"]
    fig.add_trace(go.Scatter(x=X[sv, 0], y=X[sv, 1], mode="markers", marker=dict(symbol="circle-open", size=15, line=dict(width=2, color=C.COLOR_LINE)), name="Stützvektor (α > 0)", hoverinfo="skip"))
    err = res["err_solver"]
    fig.add_trace(go.Scatter(x=X[err, 0], y=X[err, 1], mode="markers", marker=dict(symbol="x", size=12, color=C.COLOR_BOUND, line=dict(width=2.5)), name="beim Herausnehmen falsch (b des Lösers)", hoverinfo="skip"))
    return _plane_layout(fig, lo, hi, title or "Leave-one-out: falsch eingeordnete Punkte und Stützvektoren", 400)


def fig_loo_scatter(d: dict):
    """LOO-Fehler gegen Zahl der Stützvektoren der 300 eingefrorenen Instanzen (E1); darunter oder auf der Diagonalen liegt die Schranke."""
    e = d["loo"]["e1"]
    x = [r["nsv"] for r in e]
    yv = [r["loo_solver"] for r in e]
    mx = max(max(x), max(yv)) + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=yv, mode="markers", marker=dict(color=C.COLOR_ALT, size=6, opacity=0.55), name=f"{len(e)} Instanzen", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=C.COLOR_MARGIN, dash="dash"), name="LOO-Fehler = Zahl der Stützvektoren"))
    return _style(fig, "Leave-one-out gegen Stützvektoren (eingefroren, b des Lösers)", 330, "Zahl der Stützvektoren", "LOO-Fehler")


def fig_trace(trace, eps: float, title=None):
    """KKT-Lücke je Aktualisierungsschritt (logarithmisch) mit der Abbruchschwelle eps; der Dualwert steht als zweite Kurve."""
    fig = go.Figure()
    k = max(1, len(trace["step"]) // 2000)  # ausdünnen: höchstens etwa 2 000 Punkte (der Endzustand bleibt)
    idx = np.unique(np.r_[np.arange(0, len(trace["step"]), k), len(trace["step"]) - 1])
    trace = {key: np.asarray(v)[idx] for key, v in trace.items()}
    steps = trace["step"]
    fig.add_trace(go.Scatter(x=steps, y=np.maximum(trace["gap"], 1e-16), mode="lines", line=dict(color=C.COLOR_NEG, width=1.6), name="KKT-Lücke"))
    fig.add_trace(go.Scatter(x=[0, max(int(steps[-1]), 1)], y=[eps, eps], mode="lines", line=dict(color=C.COLOR_MARGIN, dash="dash"), name=f"Abbruch bei {eps:g}"))
    fig.add_trace(go.Scatter(x=steps, y=trace["dual"], mode="lines", line=dict(color=C.COLOR_ALT, width=1.6), name="Dualwert", yaxis="y2"))
    fig = _style(fig, title or "Verlauf des Lösers: KKT-Lücke und Dualwert je Aktualisierungsschritt", 300, "Aktualisierungsschritte", "KKT-Lücke", legend_y=-0.3)
    fig.update_layout(yaxis=dict(title="KKT-Lücke", type="log", fixedrange=True), yaxis2=dict(title="Dualwert", overlaying="y", side="right", fixedrange=True))
    return fig


def fig_steps_c(d: dict):
    """Aktualisierungsschritte gegen C (eingefroren, n = 200, Mittel über 5 Zufalls-Seeds, Streifen = Minimum bis Maximum)."""
    fig = go.Figure()
    for key, nm, col in (("gauss", "Gauß-Klassen, Gauß-Kern σ = 1", C.COLOR_POS), ("ringe", "Ringe mit Streuung, Gauß-Kern σ = 0,5", C.COLOR_ALT)):
        s = S.steps_c(d, key)
        cs = s["C"]
        fig.add_trace(go.Scatter(x=cs + cs[::-1], y=s["own_max"] + s["own_min"][::-1], fill="toself", fillcolor="rgba(120,120,120,0.12)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=cs, y=s["own_mean"], mode="lines+markers", line=dict(color=col, width=2), name=nm))
    fig = _style(fig, "Schritte gegen C (eingefroren, n = 200, 5 Zufalls-Seeds; Streifen: Minimum bis Maximum)", 320, "C", "Aktualisierungsschritte")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return fig


def fig_steps_n(d: dict):
    """Aktualisierungsschritte gegen n (eingefroren, C = 1, 20 Zufalls-Seeds; Streifen: Minimum bis Maximum)."""
    s = S.steps_n(d)
    ns = s["n"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns + ns[::-1], y=s["max"] + s["min"][::-1], fill="toself", fillcolor="rgba(120,120,120,0.15)", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=ns, y=s["mean_all"], mode="lines+markers", line=dict(color=C.COLOR_POS, width=2), name="Mittel über 20 Zufalls-Seeds"))
    fig.add_trace(go.Scatter(x=ns, y=[s["mean_all"][0] * (n / ns[0]) for n in ns], mode="lines", line=dict(color=C.COLOR_MARGIN, dash="dash"), name="Steigung 1 zum Vergleich"))
    fig = _style(fig, "Schritte gegen n (eingefroren, C = 1; Streifen: Minimum bis Maximum)", 320, "n (Trainingspunkte)", "Aktualisierungsschritte")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return fig
