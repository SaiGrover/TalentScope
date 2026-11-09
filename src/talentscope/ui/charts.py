"""Plotly figure builders with one shared style.

Mark rules follow the dataviz guidance: thin bars with 4 px rounded data
ends, 2 px lines, >= 8 px markers with a surface ring, hairline solid grids,
legends for two or more series, values labelled selectively, never two
y-axes, and text in text colours rather than series colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from talentscope.ui import tokens as T

TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family=T.FONT_TEXT, size=12.5, color=T.MUT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=T.CATEGORICAL,
        hoverlabel=dict(bgcolor=T.PANEL_2, bordercolor=T.BORDER_STRONG, font=dict(color=T.TXT, size=12,
                                                                                     family=T.FONT_TEXT)),
        xaxis=dict(gridcolor=T.GRID, zerolinecolor=T.GRID, linecolor=T.GRID, tickfont=dict(color=T.DIM, size=11.5),
                   title=dict(font=dict(color=T.MUT, size=12)), automargin=True),
        yaxis=dict(gridcolor=T.GRID, zerolinecolor=T.GRID, linecolor=T.GRID, tickfont=dict(color=T.DIM, size=11.5),
                   title=dict(font=dict(color=T.MUT, size=12)), automargin=True),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11.5, color=T.MUT), title=dict(text="")),
        bargap=0.35,
        coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=10, tickfont=dict(color=T.DIM, size=11))),
    )
)
pio.templates["talentscope"] = TEMPLATE

PCT = ".0%"


def style(fig: go.Figure, height: int = 340, title: str | None = None, legend: bool | None = None,
          top: int | None = None) -> go.Figure:
    n_named = sum(1 for t in fig.data if getattr(t, "showlegend", None) is not False and t.name)
    show_legend = (n_named >= 2) if legend is None else legend
    fig.update_layout(template="talentscope", height=height, showlegend=show_legend,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=T.FONT_TEXT, size=12.5, color=T.MUT),
                      legend=dict(traceorder="normal"),
                      margin=dict(l=18, r=14, b=8, t=top if top is not None else
                                  (64 if title and show_legend else 40 if title else 34 if show_legend else 10)))
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", y=0.985, yanchor="top", yref="container",
                                     font=dict(family=T.FONT_DISPLAY, size=14.5, color=T.TXT)))
    return fig


def _ring():
    return dict(color=T.PANEL, width=2)


# --------------------------------------------------------------------------- rates
def rate_dot_chart(rates: pd.DataFrame, overall: float, title: str | None = None, height: int | None = None,
                   ordered: bool = False, min_n_note: int = 30, highlight: str | None = None) -> go.Figure:
    """Horizontal dot-and-whisker chart of attrition rate per group with Wilson 95% CI.

    The dashed-free reference line is the overall rate. Small groups (n below
    ``min_n_note``) are drawn hollow so their wide intervals read as uncertain.
    """
    r = rates.copy()
    if not ordered:
        r = r.sort_values("rate")
    labels = [f"{g}  <span style='color:{T.DIM}'>n={n}</span>" for g, n in zip(r["group"], r["n"])]
    small = r["n"] < min_n_note
    colors = [T.LEFT if (highlight is None or g == highlight) else T.NEUTRAL for g in r["group"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=r["rate"], y=labels, mode="markers",
        error_x=dict(type="data", symmetric=False, array=r["ci_high"] - r["rate"], arrayminus=r["rate"] - r["ci_low"],
                     color="rgba(217,99,74,.55)", thickness=2, width=0),
        marker=dict(size=11, color=[("rgba(0,0,0,0)" if s else c) for s, c in zip(small, colors)],
                    line=dict(color=colors, width=2)),
        customdata=np.stack([r["n"], r["left"], r["ci_low"], r["ci_high"], r["lift"]], axis=1),
        hovertemplate="<b>%{y}</b><br>attrition %{x:.1%}<br>95% CI %{customdata[2]:.1%} to %{customdata[3]:.1%}"
                      "<br>%{customdata[1]} of %{customdata[0]} left<br>%{customdata[4]:.2f}x overall<extra></extra>",
        showlegend=False,
    ))
    fig.add_vline(x=overall, line_color=T.BORDER_STRONG, line_width=1)
    fig.add_annotation(x=overall, y=1, yref="paper", yanchor="bottom", text=f"overall {overall:.1%}",
                       showarrow=False, font=dict(size=11, color=T.MUT), xanchor="left", xshift=4)
    xmax = float(min(1.0, max(r["ci_high"].max() * 1.08, overall * 1.3, 0.1)))
    fig.update_xaxes(tickformat=PCT, range=[0, xmax], title=None)
    # plotly draws the first category at the bottom: ordered scales should read top-down
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=labels[::-1] if ordered else labels)
    h = height or max(220, 34 * len(r) + 70)
    return style(fig, h, title, legend=False, top=46 if title else 26)


def stacked_outcome_bars(df: pd.DataFrame, col: str, order: list | None = None, title: str | None = None,
                         horizontal: bool = True, height: int | None = None) -> go.Figure:
    """Headcount per group split into stayed / left (2 px surface gap between segments)."""
    g = df.groupby([col, "Attrition"], observed=True).size().unstack(fill_value=0)
    for c in ("No", "Yes"):
        if c not in g.columns:
            g[c] = 0
    g = g.loc[order] if order is not None else g.loc[g.sum(axis=1).sort_values(ascending=horizontal).index]
    cats = [str(i) for i in g.index]
    fig = go.Figure()
    for key, name, color in (("No", "Stayed", T.STAYED), ("Yes", "Left", T.LEFT)):
        vals = g[key].to_numpy()
        kw = dict(x=vals, y=cats, orientation="h") if horizontal else dict(x=cats, y=vals)
        cat_ax, val_ax = ("y", "x") if horizontal else ("x", "y")
        hover = "%{" + cat_ax + "}<br>" + name + ": %{" + val_ax + ":,}<extra></extra>"
        fig.add_trace(go.Bar(name=name, marker=dict(color=color, line=dict(color=T.PANEL, width=2), cornerradius=4),
                             hovertemplate=hover, **kw))
    fig.update_layout(barmode="stack")
    totals = g.sum(axis=1).to_numpy()
    for c, tot in zip(cats, totals):
        if horizontal:
            fig.add_annotation(x=tot, y=c, text=f"{tot:,}", showarrow=False, xanchor="left", xshift=6,
                               font=dict(size=11, color=T.MUT))
        else:
            fig.add_annotation(x=c, y=tot, text=f"{tot:,}", showarrow=False, yanchor="bottom", yshift=4,
                               font=dict(size=11, color=T.MUT))
    if horizontal:
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(title="employees", range=[0, totals.max() * 1.15])
    else:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(title="employees", range=[0, totals.max() * 1.15])
    h = height or (max(220, 32 * len(cats) + 90) if horizontal else 320)
    return style(fig, h, title, legend=True)


def share_100(df: pd.DataFrame, dims: dict[str, str], levels: list[str], ramp: list[str], title: str | None = None,
              height: int | None = None) -> go.Figure:
    """100% stacked bars of several ordinal dimensions on the same scale (e.g. satisfaction surveys)."""
    fig = go.Figure()
    names = list(dims.values())
    for lvl, color in zip(levels, ramp):
        shares = []
        for col in dims:
            s = df[col].astype(str).value_counts(normalize=True)
            shares.append(float(s.get(lvl, 0.0)))
        fig.add_trace(go.Bar(name=lvl, y=names, x=shares, orientation="h",
                             marker=dict(color=color, line=dict(color=T.PANEL, width=2)),
                             text=[f"{v:.0%}" if v >= 0.09 else "" for v in shares], textposition="inside",
                             insidetextanchor="start", textfont=dict(color="#0b1115", size=11),
                             hovertemplate="%{y}<br>" + lvl + ": %{x:.1%}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=PCT, range=[0, 1])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return style(fig, height or max(200, 46 * len(names) + 90), title, legend=True)


def histogram_by_outcome(df: pd.DataFrame, col: str, title: str | None = None, nbins: int = 30,
                         normalise: bool = True, height: int = 300, xlabel: str | None = None) -> go.Figure:
    fig = go.Figure()
    for key, name, color in (("No", "Stayed", T.STAYED), ("Yes", "Left", T.LEFT)):
        s = df.loc[df["Attrition"] == key, col]
        fig.add_trace(go.Histogram(x=s, name=f"{name} (n={len(s):,})", nbinsx=nbins,
                                   histnorm="percent" if normalise else None, marker=dict(color=color),
                                   opacity=0.62, hovertemplate="%{x}<br>%{y:.1f}" + ("%" if normalise else "") +
                                   "<extra>" + name + "</extra>"))
    fig.update_layout(barmode="overlay", bargap=0.06)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="% of group" if normalise else "employees")
    return style(fig, height, title, legend=True)


def box_by(df: pd.DataFrame, x: str, y: str, title: str | None = None, order: list | None = None,
           height: int = 320, ylabel: str | None = None, by_outcome: bool = True) -> go.Figure:
    fig = go.Figure()
    groups = [(("No", "Stayed", T.STAYED), ("Yes", "Left", T.LEFT))] if by_outcome else [(None, "All", T.BLUE)]
    series = groups[0] if by_outcome else groups
    for key, name, color in series:
        d = df if key is None else df[df["Attrition"] == key]
        fig.add_trace(go.Box(x=d[x].astype(str), y=d[y], name=name, marker=dict(color=color, size=4),
                             line=dict(width=1.5), boxmean=False, fillcolor="rgba(0,0,0,0)",
                             hovertemplate="%{x}<br>%{y:,}<extra>" + name + "</extra>"))
    if by_outcome:
        fig.update_layout(boxmode="group")
    if order is not None:
        fig.update_xaxes(categoryorder="array", categoryarray=[str(o) for o in order])
    fig.update_yaxes(title=ylabel)
    fig.update_xaxes(showgrid=False)
    return style(fig, height, title, legend=by_outcome)


def heatmap(matrix: pd.DataFrame, title: str | None = None, colorscale=None, zmin=None, zmax=None,
            text_fmt: str = ".0%", counts: pd.DataFrame | None = None, height: int | None = None,
            colorbar_title: str = "", hover_label: str = "value") -> go.Figure:
    z = matrix.to_numpy(dtype=float)
    text = np.vectorize(lambda v: "" if np.isnan(v) else format(v, text_fmt))(z)
    custom = counts.reindex(index=matrix.index, columns=matrix.columns).to_numpy() if counts is not None else None
    hover = "%{y} | %{x}<br>" + hover_label + " %{z:" + text_fmt + "}"
    if custom is not None:
        hover += "<br>n = %{customdata}"
    xlabels = [str(c).replace(" ", "<br>", 1) if len(matrix.columns) > 3 and len(str(c)) > 12 else str(c)
               for c in matrix.columns]
    fig = go.Figure(go.Heatmap(
        z=z, x=xlabels, y=[str(i) for i in matrix.index], text=text,
        # no fixed text colour: plotly picks dark or light text per cell for contrast
        texttemplate="%{text}", textfont=dict(size=11), colorscale=colorscale or T.SEQ_CLAY,
        zmin=zmin, zmax=zmax, customdata=custom, hovertemplate=hover + "<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(title=dict(text=colorbar_title, font=dict(size=11, color=T.MUT)), outlinewidth=0, thickness=10,
                      tickformat=text_fmt if "%" in text_fmt else None),
    ))
    fig.update_xaxes(showgrid=False, side="bottom")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    h = height or max(260, 30 * len(matrix.index) + 110)
    return style(fig, h, title, legend=False)


def ranked_bars(labels, values, title: str | None = None, color: str = T.TEAL, fmt: str = ".3f",
                height: int | None = None, xlabel: str | None = None, errors=None, highlight: set | None = None,
                hover_extra: list[str] | None = None) -> go.Figure:
    """Single-series ranked horizontal bars (largest at the top), value at the tip."""
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    colors = [color if (highlight is None or l in highlight) else T.NEUTRAL for l in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
        error_x=dict(type="data", array=errors, color=T.MUT, thickness=1.2, width=3) if errors is not None else None,
        # with error bars the value label would sit on the whisker; the tooltip carries it instead
        text=None if errors is not None else [format(v, fmt) for v in values], textposition="outside",
        textfont=dict(size=11, color=T.MUT),
        cliponaxis=False, customdata=hover_extra,
        hovertemplate="%{y}<br>%{x:" + fmt + "}" + ("<br>%{customdata}" if hover_extra else "") + "<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    xmax = float(np.nanmax(values + (np.asarray(errors) if errors is not None else 0))) if len(values) else 1
    fig.update_xaxes(title=xlabel, range=[min(0, float(np.nanmin(values)) * 1.2), xmax * 1.18 if xmax > 0 else 1])
    fig.update_layout(bargap=0.3)
    return style(fig, height or max(220, 26 * len(labels) + 70), title, legend=False)


def donut_free_split(stayed: int, left: int, height: int = 120) -> go.Figure:
    """Tall segmented bar for one whole split in two (used instead of a donut)."""
    total = stayed + left
    fig = go.Figure()
    for name, val, color in (("Stayed", stayed, T.STAYED), ("Left", left, T.LEFT)):
        fig.add_trace(go.Bar(x=[val], y=[""], orientation="h", name=name,
                             marker=dict(color=color, line=dict(color=T.PANEL, width=2), cornerradius=4),
                             text=[f"{name} {val / total:.1%}"], textposition="inside", insidetextanchor="start",
                             textfont=dict(color="#0b1115", size=12, family=T.FONT_TEXT),
                             hovertemplate=f"{name}: {val:,} ({val / total:.1%})<extra></extra>", width=0.62))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(visible=False, range=[0, total])
    fig.update_yaxes(visible=False)
    return style(fig, height, None, legend=False, top=4)


def curve_chart(curves: list[dict], kind: str = "roc", title: str | None = None, baseline: float | None = None,
                height: int = 360) -> go.Figure:
    """ROC or PR curves. Each curve dict: {name, x, y, color, dash?, width?}."""
    fig = go.Figure()
    if kind == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.BORDER_STRONG, width=1),
                                 name="chance", hoverinfo="skip", showlegend=False))
    elif baseline is not None:
        fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines", line=dict(color=T.BORDER_STRONG, width=1),
                                 name="chance", hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=1, y=baseline, text=f"prevalence {baseline:.1%}", showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=10.5, color=T.DIM))
    for c in curves:
        fig.add_trace(go.Scatter(x=c["x"], y=c["y"], mode="lines", name=c["name"],
                                 line=dict(color=c["color"], width=c.get("width", 2), shape="hv" if kind == "pr" else "linear",
                                           dash=c.get("dash")),
                                 hovertemplate=("FPR %{x:.2f}<br>TPR %{y:.2f}" if kind == "roc" else
                                                "recall %{x:.2f}<br>precision %{y:.2f}") + "<extra>" + c["name"] + "</extra>"))
    if kind == "roc":
        fig.update_xaxes(title="false positive rate", range=[0, 1], tickformat=".1f")
        fig.update_yaxes(title="true positive rate (recall)", range=[0, 1.02], tickformat=".1f")
    else:
        fig.update_xaxes(title="recall", range=[0, 1], tickformat=".1f")
        fig.update_yaxes(title="precision", range=[0, 1.02], tickformat=".1f")
    return style(fig, height, title, legend=True)


def confusion_chart(tn: int, fp: int, fn: int, tp: int, title: str | None = None, height: int = 300) -> go.Figure:
    z = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_tot = z.sum(axis=1, keepdims=True)
    share = np.divide(z, row_tot, out=np.zeros_like(z), where=row_tot > 0)
    text = [[f"<b>{int(z[i, j]):,}</b><br>{share[i, j]:.0%} of row" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=share, x=["Predicted stay", "Predicted leave"], y=["Actually stayed", "Actually left"], text=text,
        texttemplate="%{text}", textfont=dict(size=12.5, color=T.TXT), colorscale=T.SEQ_TEAL, zmin=0, zmax=1,
        showscale=False, xgap=3, ygap=3, hovertemplate="%{y} / %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return style(fig, height, title, legend=False, top=64 if title else 30)


def calibration_chart(series: list[dict], title: str | None = None, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.BORDER_STRONG, width=1),
                             name="perfect calibration", hoverinfo="skip", showlegend=False))
    for s in series:
        fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="lines+markers", name=s["name"],
                                 line=dict(color=s["color"], width=2), marker=dict(size=9, line=_ring()),
                                 hovertemplate="mean predicted %{x:.2f}<br>observed %{y:.2f}<extra>" + s["name"] + "</extra>"))
    top = max([1.0] + [float(np.max(s["y"])) for s in series if len(s["y"])])
    fig.update_xaxes(title="mean predicted probability", range=[0, 1])
    fig.update_yaxes(title="observed attrition rate", range=[0, min(1.02, top * 1.05)])
    return style(fig, height, title, legend=True)


def threshold_chart(sweep: pd.DataFrame, chosen: float, title: str | None = None, height: int = 340,
                    marker: float | None = None) -> go.Figure:
    fig = go.Figure()
    for col, name, color in (("precision", "Precision", T.BLUE), ("recall", "Recall", T.CLAY), ("f1", "F1", T.TEAL)):
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[col], mode="lines", name=name,
                                 line=dict(color=color, width=2),
                                 hovertemplate="threshold %{x:.2f}<br>" + name + " %{y:.3f}<extra></extra>"))
    fig.add_vline(x=chosen, line_color=T.TXT, line_width=1)
    fig.add_annotation(x=chosen, y=1.0, yref="y", text=f"selected {chosen:.2f}", showarrow=False, xanchor="left",
                       xshift=4, font=dict(size=11, color=T.TXT))
    if marker is not None and abs(marker - chosen) > 1e-9:
        fig.add_vline(x=marker, line_color=T.AMBER, line_width=1)
    fig.update_xaxes(title="decision threshold", range=[0, 1])
    fig.update_yaxes(title="score", range=[0, 1.05])
    return style(fig, height, title, legend=True)


def waterfall(contrib: pd.Series, base: float, final: float, space: str, title: str | None = None,
              k: int = 10, height: int | None = None) -> go.Figure:
    """Local explanation: the k largest contributions, the rest folded into 'other features'."""
    c = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    top = c.iloc[:k]
    rest = c.iloc[k:].sum()
    labels = list(top.index) + ([f"{len(c) - k} other features"] if len(c) > k else [])
    vals = list(top.values) + ([rest] if len(c) > k else [])
    unit = "log-odds" if space == "log-odds" else "probability"
    fig = go.Figure(go.Waterfall(
        orientation="h", y=["model average"] + labels + ["this employee"],
        x=[base] + vals + [0], measure=["absolute"] + ["relative"] * len(vals) + ["total"],
        increasing=dict(marker=dict(color=T.LEFT)), decreasing=dict(marker=dict(color=T.STAYED)),
        totals=dict(marker=dict(color=T.BLUE)), connector=dict(line=dict(color=T.BORDER, width=1)),
        text=[f"{base:+.3f}"] + [f"{v:+.3f}" for v in vals] + [f"{final:.3f}"], textposition="outside",
        textfont=dict(size=11, color=T.MUT), cliponaxis=False,
        hovertemplate="%{y}<br>%{x:+.3f} " + unit + "<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title=f"model output ({unit})")
    return style(fig, height or 34 * (len(labels) + 2) + 90, title, legend=False)


def beeswarm(values: np.ndarray, display: np.ndarray, names: list[str], order: list[int], title: str | None = None,
             max_points: int = 400, seed: int = 0, height: int | None = None) -> go.Figure:
    """SHAP summary plot: one row per feature, one dot per employee, colour = feature value (low -> high)."""
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    idx = rng.choice(n, size=min(n, max_points), replace=False)
    fig = go.Figure()
    for rank, j in enumerate(order):
        v = values[idx, j]
        d = display[idx, j].astype(float)
        lo, hi = np.nanpercentile(d, 5), np.nanpercentile(d, 95)
        norm = np.clip((d - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(d, 0.5)
        # jitter proportional to local density, like a beeswarm
        order_v = np.argsort(v)
        jitter = np.zeros_like(v)
        bins = np.digitize(v, np.linspace(v.min(), v.max() + 1e-9, 40))
        for b in np.unique(bins):
            members = np.flatnonzero(bins == b)
            k = len(members)
            jitter[members] = (np.arange(k) - (k - 1) / 2) / max(k, 1) * min(0.8, 0.06 * k)
        fig.add_trace(go.Scattergl(
            x=v, y=rank + jitter, mode="markers", showlegend=False,
            marker=dict(size=5, color=norm, colorscale=[[0, "#2f6f9a"], [0.5, "#7a7f86"], [1, "#e0904f"]], cmin=0, cmax=1,
                        opacity=0.85, showscale=rank == 0,
                        colorbar=dict(title=dict(text="feature value", side="right", font=dict(size=11, color=T.MUT)),
                                      tickvals=[0, 1], ticktext=["low", "high"], thickness=8, len=0.6, outlinewidth=0)),
            customdata=np.stack([d], axis=1), name=names[j],
            hovertemplate=names[j] + "<br>value %{customdata[0]:,.2f}<br>SHAP %{x:+.3f}<extra></extra>",
        ))
        del order_v
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[names[j] for j in order], autorange="reversed",
                     showgrid=False, range=[len(order) - 0.4, -0.6])
    fig.add_vline(x=0, line_color=T.BORDER_STRONG, line_width=1)
    fig.update_xaxes(title="SHAP value (impact on model output)")
    return style(fig, height or 30 * len(order) + 90, title, legend=False)
