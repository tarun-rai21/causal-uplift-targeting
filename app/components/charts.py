"""
Plotly figure builders for the Streamlit app. Pure functions of DataFrames
(no Streamlit, no DB), so they are unit-testable.

Color follows the dataviz method (palette validated with its script):
- Models take categorical slots 1-3 in fixed order, so a model keeps its
  color on every chart and when filtered; validated all-pairs in both modes.
- Segments take a separate, validated set (yellow > magenta > violet > red,
  in segment order) so they never impersonate a model on the same page.
- References (random, X-ceiling) are neutral gray dashes, not series colors.
- Light-mode aqua sits below 3:1 contrast; every chart therefore ships a
  legend and the page offers a table view of the same numbers.
Dark mode uses the dark-surface steps of the same hues, not a flip.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.components.data import MODEL_LABELS, MODEL_ORDER, SEGMENT_ORDER

_MODEL_COLORS = {
    "light": {"baseline": "#2a78d6", "t_learner": "#eb6834", "x_learner": "#1baf7a"},
    "dark": {"baseline": "#3987e5", "t_learner": "#d95926", "x_learner": "#199e70"},
}
_SEGMENT_COLORS = {
    "light": dict(zip(SEGMENT_ORDER, ("#eda100", "#e87ba4", "#6250d6", "#e34948"))),
    "dark": dict(zip(SEGMENT_ORDER, ("#c98500", "#d55181", "#9085e9", "#e66767"))),
}
# surface = Streamlit's default page background per theme: the 2px spacer
# between stacked fills must be the surface color, not transparent.
_INK = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "ref": "#7a7974",
              "surface": "#ffffff"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "grid": "#383835", "ref": "#9a998f",
             "surface": "#0e1117"},
}
_SEGMENT_LABELS = {
    "persuadable": "Persuadable", "sure_thing": "Sure thing",
    "lost_cause": "Lost cause", "sleeping_dog": "Sleeping dog",
}


def model_color(model_type: str, mode: str = "light") -> str:
    return _MODEL_COLORS[mode][model_type]


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, mode: str, y_title: str, x_title: str, height: int = 380):
    ink = _INK[mode]
    fig.update_layout(
        height=height,
        # Generous base margins plus automargin: tight fixed margins clipped
        # tick labels like "1,000" down to "0" (caught by screenshot review).
        margin=dict(l=70, r=20, t=40, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink["text"], size=13),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0, traceorder="normal",
            font=dict(color=ink["muted"]),
        ),
    )
    axis = dict(
        gridcolor=ink["grid"], zerolinecolor=ink["muted"], linecolor=ink["grid"],
        tickfont=dict(color=ink["muted"]), title_font=dict(color=ink["muted"]),
        automargin=True, title_standoff=12,
    )
    fig.update_xaxes(title_text=x_title, tickformat=",", **axis)
    fig.update_yaxes(title_text=y_title, tickformat=",", **axis)
    return fig


def policy_net_chart(views, random_net=None, mode: str = "light", budget_pct=None) -> go.Figure:
    """
    Net revenue vs budget, one line per model. Simulated pilots plot TRUE
    net; real pilots plot the outcome-based estimate with its CI band (the
    only honest number there is). Never both on one chart -- one measure,
    one axis.
    """
    fig = go.Figure()
    has_truth = all("true_net" in v.curve for v in views)
    for v in views:
        color = model_color(v.model_type, mode)
        x = v.curve["budget_pct"] * 100
        if not has_truth:
            band = v.curve.dropna(subset=["ab_net_ci_lower"])
            fig.add_trace(go.Scatter(
                x=pd.concat([band["budget_pct"], band["budget_pct"][::-1]]) * 100,
                y=pd.concat([band["ab_net_ci_upper"], band["ab_net_ci_lower"][::-1]]),
                fill="toself", fillcolor=_rgba(color, 0.12), line=dict(width=0),
                hoverinfo="skip", showlegend=False,
            ))
        y = v.curve["true_net"] if has_truth else v.curve["ab_net_revenue"]
        fig.add_trace(go.Scatter(
            x=x, y=y, name=MODEL_LABELS[v.model_type], mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=8, color=color),
            hovertemplate="%{y:,.0f}<extra>" + MODEL_LABELS[v.model_type] + "</extra>",
        ))
    if random_net is not None and has_truth:
        fig.add_trace(go.Scatter(
            x=views[0].curve["budget_pct"] * 100, y=random_net, name="Random targeting",
            mode="lines", line=dict(color=_INK[mode]["ref"], width=2, dash="dash"),
            hovertemplate="%{y:,.0f}<extra>Random</extra>",
        ))
    if budget_pct is not None:
        fig.add_vline(x=budget_pct * 100, line=dict(color=_INK[mode]["muted"], width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color=_INK[mode]["muted"], width=1))
    title = "True net revenue" if has_truth else "Estimated net revenue (A/B, 95% CI)"
    return _layout(fig, mode, title, "Budget (% of cost to treat everyone)")


def valuation_chart(table: pd.DataFrame, mode: str = "light") -> go.Figure:
    """
    Dot plot at the chosen budget: per model, the model's own claim (hollow),
    the outcome-based A/B estimate with its CI (filled + whisker) and, when
    known, the truth (diamond). Rows are models, colored by model; marker
    SHAPE carries the measure so identity is never color-alone.
    """
    fig = go.Figure()
    models = list(table["model"])
    colors = [model_color(_model_key(m), mode) for m in models]
    # The legend here encodes SHAPE (which measure), so its glyphs are drawn
    # in neutral ink by legend-only traces; otherwise each glyph inherits the
    # first row's model color and wrongly reads as "naive's estimate".
    ink = _INK[mode]["muted"]
    for name, symbol in (
        ("A/B estimate (95% CI)", "circle"), ("Model's own claim", "circle-open"),
        *((("True net", "diamond"),) if "true net" in table else ()),
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], name=name, mode="markers",
            marker=dict(symbol=symbol, size=11, color=ink, line=dict(width=2, color=ink)),
        ))
    fig.add_trace(go.Scatter(
        y=models, x=table["A/B net"], name="A/B estimate (95% CI)", mode="markers",
        showlegend=False,
        marker=dict(symbol="circle", size=11, color=colors),
        error_x=dict(
            type="data", symmetric=False, thickness=2, width=6, color=_INK[mode]["muted"],
            array=table["A/B 95% CI high"] - table["A/B net"],
            arrayminus=table["A/B net"] - table["A/B 95% CI low"],
        ),
        hovertemplate="%{y}: %{x:,.0f}<extra>A/B estimate</extra>",
    ))
    fig.add_trace(go.Scatter(
        y=models, x=table["model-claimed net"], name="Model's own claim", mode="markers",
        showlegend=False,
        marker=dict(symbol="circle-open", size=12, color=colors, line=dict(width=2)),
        hovertemplate="%{y}: %{x:,.0f}<extra>Model's own claim</extra>",
    ))
    if "true net" in table:
        fig.add_trace(go.Scatter(
            y=models, x=table["true net"], name="True net", mode="markers", showlegend=False,
            marker=dict(symbol="diamond", size=12, color=_INK[mode]["text"]),
            hovertemplate="%{y}: %{x:,.0f}<extra>True net</extra>",
        ))
    fig.add_vline(x=0, line=dict(color=_INK[mode]["muted"], width=1))
    fig = _layout(fig, mode, "", "Net revenue at this budget", height=280)
    fig.update_layout(hovermode="closest")
    # Explicitly categorical: the legend-only traces come first with
    # y=[None], which otherwise makes Plotly type the axis as numeric and
    # silently drop every model row (caught in screenshot review).
    fig.update_yaxes(
        type="category", categoryorder="array", categoryarray=models, autorange="reversed"
    )
    return fig


def segment_mix_chart(mixes: dict[str, pd.Series], mode: str = "light") -> go.Figure:
    """Who each policy targets: true-segment shares, one stacked bar per model."""
    fig = go.Figure()
    models = [MODEL_LABELS[m] for m in MODEL_ORDER if m in mixes]
    for seg in SEGMENT_ORDER:
        values = [mixes[m][seg] * 100 for m in MODEL_ORDER if m in mixes]
        fig.add_trace(go.Bar(
            y=models, x=values, name=_SEGMENT_LABELS[seg], orientation="h",
            marker=dict(
                color=_SEGMENT_COLORS[mode][seg], line=dict(width=2, color=_INK[mode]["surface"])
            ),
            hovertemplate="%{y}: %{x:.0f}%<extra>" + _SEGMENT_LABELS[seg] + "</extra>",
        ))
    fig = _layout(fig, mode, "", "Share of targeted users (%)", height=240)
    # Horizontal stacked bars list legend entries in reverse by default;
    # keep them in stacking (segment) order.
    fig.update_layout(barmode="stack", bargap=0.35, hovermode="closest",
                      legend=dict(traceorder="normal"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 100])
    return fig


def qini_chart(curves: pd.DataFrame, mode: str = "light", bands: bool = False) -> go.Figure:
    """
    Qini curves plus the random line. CI bands are opt-in: three overlapping
    translucent bands read as one muddy blob (caught in screenshot review),
    and the intervals are in the table beside the chart anyway.
    """
    fig = go.Figure()
    end = None
    for model_type in MODEL_ORDER:
        c = curves[curves["model_type"] == model_type]
        if c.empty:
            continue
        color = model_color(model_type, mode)
        x = c["pct_targeted"] * 100
        if bands:
            fig.add_trace(go.Scatter(
                x=pd.concat([x, x[::-1]]), y=pd.concat([c["ci_upper"], c["ci_lower"][::-1]]),
                fill="toself", fillcolor=_rgba(color, 0.10), line=dict(width=0),
                hoverinfo="skip", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=x, y=c["cumulative_incremental_gain"], name=MODEL_LABELS[model_type],
            mode="lines", line=dict(color=color, width=2),
            hovertemplate="%{y:,.1f}<extra>" + MODEL_LABELS[model_type] + "</extra>",
        ))
        end = c["cumulative_incremental_gain"].iloc[-1]
    if end is not None:
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[0, end], name="Random targeting", mode="lines",
            line=dict(color=_INK[mode]["ref"], width=2, dash="dash"), hoverinfo="skip",
        ))
    return _layout(fig, mode, "Incremental conversions (Qini)", "Users targeted (%)")


def _model_key(label: str) -> str:
    return {v: k for k, v in MODEL_LABELS.items()}[label]
