"""Reusable Streamlit components and the app's CSS layer.

Native theming (``.streamlit/config.toml``) sets the colours and fonts.
The CSS below adds what config cannot: the background treatment, the card
shape, the KPI tiles, the navigation rail and the method flow. It targets
``st-key-*`` classes produced by keyed containers wherever possible rather
than private Streamlit class names.
"""
from __future__ import annotations

import base64
import html
import itertools
from typing import Iterable

import streamlit as st

from talentscope import config
from talentscope.ui import tokens as T

_card_ids = itertools.count()

MARK_SVG = f"""<svg viewBox="0 0 48 48" aria-hidden="true"><g fill="none" stroke-linecap="round">
<circle cx="24" cy="24" r="19" stroke="{T.TEAL}" stroke-opacity=".55" stroke-width="1.6" stroke-dasharray="4 5"><animateTransform attributeName="transform" type="rotate" from="0 24 24" to="360 24 24" dur="22s" repeatCount="indefinite"/></circle>
<circle cx="24" cy="24" r="12.5" stroke="{T.TXT}" stroke-opacity=".9" stroke-width="2"/>
<path d="M24 3v7M24 38v7M3 24h7M38 24h7" stroke="{T.TEAL}" stroke-width="2"/>
<circle cx="24" cy="20.5" r="3.6" fill="{T.TEAL}" stroke="none"/>
<path d="M17.6 31.2c1.3-3.6 3.7-5.3 6.4-5.3s5.1 1.7 6.4 5.3" stroke="{T.TEAL}" stroke-width="2.2"/>
<circle cx="33.5" cy="15" r="2.3" fill="{T.CLAY}" stroke="none"/></g></svg>"""

GITHUB_SVG = ('<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 '
              '3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94'
              '-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78'
              '-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27'
              '.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75'
              '-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8'
              '-8-8z"/></svg>')


def _svg_uri(svg: str) -> str:
    """SVG as a base64 data URI: survives HTML sanitising, unlike inline <svg>."""
    if "xmlns" not in svg:
        svg = svg.replace("<svg ", "<svg xmlns='http://www.w3.org/2000/svg' ", 1)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def inject_css() -> None:
    dots = _svg_uri("<svg width='22' height='22'><circle cx='1' cy='1' r='1' fill='#a9c4ce' fill-opacity='0.07'/></svg>")
    st.html(f"""
<style>
  .stApp {{
    background:
      radial-gradient(1100px 520px at 8% -6%, rgba(47,166,151,.16), transparent 62%),
      radial-gradient(900px 480px at 100% 0%, rgba(79,134,216,.10), transparent 60%),
      radial-gradient(800px 600px at 50% 120%, rgba(217,99,74,.06), transparent 60%),
      url("{dots}"),
      {T.BG};
    background-attachment: fixed;
  }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stHeaderActionElements"], footer {{ display: none !important; }}
  .block-container {{ max-width: 1680px; padding-top: 1.2rem; }}
  h1, h2, h3, h4 {{ letter-spacing: -.02em; }}
  .stMarkdown p, .stMarkdown li {{ line-height: 1.62; }}

  /* brand bar */
  .ts-top {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
     padding:.35rem .1rem .9rem .1rem; position:relative; }}
  .ts-top:after {{ content:""; position:absolute; left:0; right:0; bottom:0; height:1px;
     background:linear-gradient(90deg, transparent, {T.TEAL} 18%, {T.BLUE} 50%, {T.CLAY} 82%, transparent);
     background-size:220% 100%; animation: ts-sweep 14s linear infinite; opacity:.75; }}
  @keyframes ts-sweep {{ from {{ background-position:0% 0; }} to {{ background-position:220% 0; }} }}
  .ts-brand {{ display:flex; align-items:center; gap:.85rem; }}
  .ts-mark {{ width:46px; height:46px; display:grid; place-items:center; border-radius:14px;
     border:1px solid {T.BORDER}; background:radial-gradient(circle at 30% 25%, rgba(47,166,151,.28), rgba(18,26,32,.6) 70%);
     box-shadow:0 0 30px -10px rgba(47,166,151,.8); }}
  .ts-mark svg {{ width:34px; height:34px; overflow:visible; }}
  .ts-mark .ring {{ transform-origin:24px 24px; animation: ts-spin 22s linear infinite; }}
  @keyframes ts-spin {{ to {{ transform: rotate(360deg); }} }}
  .ts-wm {{ font-family:{T.FONT_DISPLAY}; font-size:1.5rem; font-weight:700; letter-spacing:-.035em; line-height:1.05;
     color:{T.TXT}; }}
  .ts-wm span {{ color:{T.TEAL}; }}
  .ts-tag {{ color:{T.MUT}; font-size:.7rem; letter-spacing:.18em; text-transform:uppercase; font-weight:600;
     margin-top:.25rem; }}
  .ts-right {{ display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; justify-content:flex-end; }}
  .ts-pill {{ display:inline-flex; align-items:center; gap:.35rem; padding:.22rem .6rem; border-radius:999px;
     border:1px solid {T.BORDER}; background:rgba(255,255,255,.03); color:{T.MUT} !important; font-size:.72rem;
     text-decoration:none !important; white-space:nowrap; }}
  .ts-pill b {{ color:{T.TXT}; font-weight:600; }}
  .ts-pill.warn {{ border-color:rgba(217,164,65,.45); background:rgba(217,164,65,.08); }}
  .ts-pill.warn b {{ color:{T.STATUS['warn']}; }}
  a.ts-pill:hover {{ border-color:{T.TEAL}; color:{T.TXT} !important; }}
  .ts-pill svg {{ width:13px; height:13px; }}
  @media (prefers-reduced-motion: reduce) {{ .ts-top:after, .ts-mark .ring {{ animation:none !important; }} }}

  /* navigation rail: three phases, nine stations */
  /* sticky only where there is room: on a phone the stacked rail would cover the screen */
  @media (min-width: 900px) {{
    div[data-testid="stLayoutWrapper"]:has(> div.st-key-ts_rail) {{ position:sticky; top:0; z-index:90; }} }}
  div.st-key-ts_rail {{ background:rgba(11,17,21,.86); backdrop-filter:blur(12px) saturate(1.1);
     border-bottom:1px solid {T.BORDER}; padding:.55rem 0 .65rem 0; margin:0 0 1.1rem 0;
     box-shadow:0 16px 24px -22px rgba(0,0,0,.9); }}
  div[class*="st-key-ts_phase_"] {{ border:1px solid {T.BORDER}; border-radius:14px; padding:.4rem .45rem .45rem .45rem;
     background:rgba(255,255,255,.018); }}
  .ts-phase {{ font-size:.6rem; letter-spacing:.2em; text-transform:uppercase; font-weight:700; color:{T.DIM};
     padding:0 .2rem .3rem .2rem; display:flex; justify-content:space-between; }}
  .ts-phase i {{ font-style:normal; color:var(--c); }}
  div[class*="st-key-ts_nav_"] a {{ border-radius:10px; border:1px solid transparent; padding:.3rem .55rem;
     transition: background .15s ease, border-color .15s ease; width:100%; }}
  div[class*="st-key-ts_nav_"] a:hover {{ background:rgba(255,255,255,.045); border-color:{T.BORDER}; }}
  div[class*="st-key-ts_nav_"] a p {{ font-size:.82rem; color:{T.MUT}; white-space:nowrap; }}
  div[class*="st-key-ts_navon_"] a {{ background:rgba(47,166,151,.14); border-color:rgba(47,166,151,.6); }}
  div[class*="st-key-ts_navon_"] a p {{ color:{T.TXT}; font-weight:600; }}

  /* page header */
  .ts-head {{ margin:.2rem 0 1rem 0; }}
  .ts-eyebrow {{ font-size:.68rem; letter-spacing:.2em; text-transform:uppercase; font-weight:700; color:{T.TEAL}; }}
  .ts-title {{ font-family:{T.FONT_DISPLAY}; font-size:1.85rem; font-weight:700; letter-spacing:-.03em; color:{T.TXT};
     margin:.15rem 0 .25rem 0; line-height:1.15; }}
  .ts-sub {{ color:{T.MUT}; font-size:.93rem; max-width:980px; line-height:1.6; }}

  /* section heading */
  .ts-sec {{ display:flex; align-items:baseline; gap:.7rem; margin:1.6rem 0 .7rem 0; padding-left:.7rem;
     border-left:3px solid {T.TEAL}; }}
  .ts-sec h3 {{ font-family:{T.FONT_DISPLAY}; font-size:1.12rem; font-weight:600; margin:0; color:{T.TXT}; }}
  .ts-sec span {{ color:{T.DIM}; font-size:.83rem; }}

  /* KPI tiles */
  .ts-tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:.7rem; margin:.2rem 0 1rem 0; }}
  .ts-tile {{ position:relative; overflow:hidden; background:linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.012));
     border:1px solid {T.BORDER}; border-radius:14px; padding:.8rem .95rem .75rem .95rem; }}
  .ts-tile:before {{ content:""; position:absolute; top:0; left:0; right:0; height:2px;
     background:linear-gradient(90deg, var(--c), transparent); }}
  .ts-tile .l {{ color:{T.MUT}; font-size:.66rem; letter-spacing:.13em; text-transform:uppercase; font-weight:600; }}
  .ts-tile .v {{ font-family:{T.FONT_DISPLAY}; font-size:1.7rem; font-weight:650; color:{T.TXT}; line-height:1.15;
     margin:.3rem 0 .12rem 0; letter-spacing:-.02em; }}
  .ts-tile .v small {{ font-size:.9rem; color:{T.MUT}; font-weight:500; margin-left:.25rem; }}
  .ts-tile .n {{ color:{T.DIM}; font-size:.75rem; line-height:1.4; }}
  .ts-tile .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--c); margin-right:.35rem; }}

  /* one card shape for every chart and panel */
  div[class*="st-key-card_"] {{ background:{T.PANEL}; border:1px solid {T.BORDER}; border-radius:16px;
     padding:.8rem .9rem .45rem .9rem; }}
  div[class*="st-key-card_"] div[data-testid="stExpander"] {{ border:0; border-top:1px solid {T.BORDER}; border-radius:0;
     background:transparent; }}
  div[class*="st-key-card_"] div[data-testid="stExpander"] details {{ border:0; }}
  div[class*="st-key-card_"] div[data-testid="stExpander"] summary p {{ font-size:.8rem; color:{T.MUT}; }}
  .ts-card-title {{ font-family:{T.FONT_DISPLAY}; font-size:.98rem; font-weight:600; color:{T.TXT}; margin:0 0 .1rem 0; }}
  .ts-card-sub {{ color:{T.DIM}; font-size:.78rem; margin-bottom:.35rem; line-height:1.45; }}
  .ts-panel {{ background:{T.PANEL}; border:1px solid {T.BORDER}; border-radius:16px; padding:.95rem 1.1rem; }}

  /* notes */
  .ts-note {{ border:1px solid {T.BORDER}; border-left:3px solid var(--c); background:rgba(255,255,255,.02);
     border-radius:12px; padding:.6rem .85rem; color:{T.MUT}; font-size:.83rem; line-height:1.55; margin:.2rem 0 .8rem 0; }}
  .ts-note b {{ color:{T.TXT}; }}

  /* findings list */
  .ts-find {{ display:grid; grid-template-columns:28px 1fr; gap:.6rem; padding:.55rem 0; border-bottom:1px dashed {T.BORDER}; }}
  .ts-find:last-child {{ border-bottom:0; }}
  .ts-find .k {{ width:28px; height:28px; border-radius:9px; display:grid; place-items:center; font-weight:700;
     font-size:.8rem; color:{T.CLAY}; border:1px solid rgba(217,99,74,.5); background:rgba(217,99,74,.08); }}
  .ts-find .t {{ color:{T.TXT}; font-size:.86rem; line-height:1.5; }}
  .ts-find .s {{ color:{T.DIM}; font-size:.76rem; }}

  /* meter */
  .ts-meter {{ height:10px; border-radius:999px; background:rgba(255,255,255,.06); position:relative; overflow:visible;
     margin:.5rem 0 1.4rem 0; }}
  .ts-meter .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius:999px; }}
  .ts-meter .tick {{ position:absolute; top:-5px; bottom:-5px; width:2px; background:{T.TXT}; opacity:.6; }}
  .ts-meter .tl {{ position:absolute; top:14px; font-size:.68rem; color:{T.DIM}; transform:translateX(-50%); white-space:nowrap; }}

  /* method flow */
  .ts-flow .step {{ position:relative; display:grid; grid-template-columns:40px 1fr; gap:.9rem; padding-bottom:.95rem; }}
  .ts-flow .step:not(:last-child):after {{ content:""; position:absolute; left:19px; top:44px; bottom:0; width:2px;
     background:linear-gradient(180deg, var(--c), rgba(170,196,206,.12)); }}
  .ts-flow .node {{ width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-weight:700;
     font-size:.85rem; color:var(--c); border:1.5px solid var(--c); background:rgba(255,255,255,.03);
     font-family:{T.FONT_DISPLAY}; }}
  .ts-flow h5 {{ margin:0 0 .15rem 0; font-size:.93rem; color:{T.TXT}; font-family:{T.FONT_DISPLAY}; font-weight:600; }}
  .ts-flow p {{ margin:0; color:{T.MUT}; font-size:.83rem; line-height:1.58; }}
  .ts-flow .chip {{ display:inline-block; margin-top:.35rem; padding:.12rem .55rem; border-radius:999px; font-size:.7rem;
     color:var(--c); border:1px solid var(--c); }}

  /* widgets */
  div[data-testid="stWidgetLabel"] p {{ color:{T.MUT} !important; font-size:.7rem !important; letter-spacing:.11em;
     text-transform:uppercase; font-weight:600; }}
  div[data-testid="stCaptionContainer"] p {{ color:{T.DIM} !important; font-size:.78rem !important; }}
  div[class*="st-key-filters"] {{ background:rgba(255,255,255,.022); border:1px solid {T.BORDER}; border-radius:14px;
     padding:.55rem .8rem .2rem .8rem; margin-bottom:.9rem; }}
  .stButton button[kind="primary"] p, .stFormSubmitButton button[kind="primaryFormSubmit"] p,
  .stDownloadButton button[kind="primary"] p {{ color:#06120f !important; font-weight:650; }}
  div[data-testid="stExpander"] {{ border-color:{T.BORDER}; border-radius:12px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:.25rem; border-bottom:1px solid {T.BORDER}; }}
  .stTabs [data-baseweb="tab"] {{ border-radius:10px 10px 0 0; padding:.4rem .85rem; }}
  .stTabs [aria-selected="true"] {{ background:rgba(47,166,151,.12); }}
  code {{ color:#8fd8cb; background:rgba(47,166,151,.1); }}
</style>""")


def brand_bar(synthetic_rows: int | None = None) -> None:
    rows = f"{synthetic_rows:,} employees" if synthetic_rows else "IBM HR sample"
    st.html(f"""
<div class="ts-top">
  <div class="ts-brand">
    <div class="ts-mark"><img src="{_svg_uri(MARK_SVG)}" alt="TalentScope mark" width="34" height="34"></div>
    <div><div class="ts-wm">Talent<span>Scope</span></div>
      <div class="ts-tag">Employee attrition intelligence</div></div>
  </div>
  <div class="ts-right">
    <span class="ts-pill"><b>{rows}</b> IBM HR Analytics</span>
    <span class="ts-pill warn" title="IBM created this dataset; it does not describe real people.">
      <b>Synthetic data</b> fictional employees</span>
    <a class="ts-pill" href="{config.REPO_URL}" target="_blank" rel="noopener"><img src="{_svg_uri(GITHUB_SVG.replace('currentColor', '#9aabb5'))}" alt="" width="13" height="13"> SaiGrover/TalentScope</a>
  </div>
</div>""")


def nav_rail(phases: list[tuple[str, str, list]], current_title: str) -> None:
    """Pages grouped into the three phases of the workflow; the active page is lit."""
    with st.container(key="ts_rail"):
        cols = st.columns([len(p[2]) + 0.4 for p in phases], gap="small")
        for ci, (col, (phase, colour, pages)) in enumerate(zip(cols, phases)):
            with col, st.container(key=f"ts_phase_{ci}"):
                st.html(f'<div class="ts-phase" style="--c:{colour}"><span>{html.escape(phase)}</span>'
                        f'<i>{"0" + str(ci + 1)}</i></div>')
                with st.container(horizontal=True, gap="xsmall"):
                    for pi, page in enumerate(pages):
                        active = page.title == current_title
                        with st.container(key=f"ts_{'navon' if active else 'nav'}_{ci}_{pi}", width="content"):
                            st.page_link(page, label=page.title, icon=page.icon)


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.html(f'<div class="ts-head"><div class="ts-eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="ts-title">{html.escape(title)}</div>'
            + (f'<div class="ts-sub">{subtitle}</div>' if subtitle else "") + "</div>")


def section(title: str, sub: str = "") -> None:
    st.html(f'<div class="ts-sec"><h3>{html.escape(title)}</h3><span>{sub}</span></div>')


def tiles(items: Iterable[tuple]) -> None:
    """items: (label, value, note, colour[, unit])"""
    out = ['<div class="ts-tiles">']
    for it in items:
        label, value, note, colour = it[:4]
        unit = it[4] if len(it) > 4 else ""
        out.append(f'<div class="ts-tile" style="--c:{colour}"><div class="l"><span class="dot"></span>{html.escape(label)}</div>'
                   f'<div class="v">{value}{f"<small>{unit}</small>" if unit else ""}</div>'
                   f'<div class="n">{note}</div></div>')
    out.append("</div>")
    st.html("".join(out))


def note(text: str, colour: str = T.BLUE) -> None:
    st.html(f'<div class="ts-note" style="--c:{colour}">{text}</div>')


def card(title: str | None = None, sub: str | None = None, key: str | None = None):
    """A panel container. Use as ``with card("Title"):``."""
    c = st.container(key=f"card_{key or next(_card_ids)}")
    if title:
        c.html(f'<div class="ts-card-title">{html.escape(title)}</div>'
               + (f'<div class="ts-card-sub">{sub}</div>' if sub else ""))
    return c


def chart(fig, title: str | None = None, sub: str | None = None, what: str | None = None, why: str | None = None,
          key: str | None = None, alt: str | None = None, on_select=None, selection_mode=None):
    """Every figure sits in a card with an optional two-part interpretation."""
    with card(title, sub, key=key):
        kwargs = dict(width="stretch", theme=None, config={"displayModeBar": False, "displaylogo": False},
                      alt=alt or title or "chart")
        if key:
            kwargs["key"] = f"fig_{key}"
        if on_select is not None:
            kwargs["on_select"] = on_select
            if selection_mode:
                kwargs["selection_mode"] = selection_mode
        event = st.plotly_chart(fig, **kwargs)
        if what:
            with st.expander("Interpretation", icon=":material/insights:"):
                st.markdown(f"**What this shows.** {what}")
                if why:
                    st.markdown(f"**Why it matters.** {why}")
    return event


def findings(items: list[dict]) -> None:
    if not items:
        st.caption("No group in the current selection differs from the rest by more than its confidence interval.")
        return
    out = []
    for i, f in enumerate(items, 1):
        out.append(f'<div class="ts-find"><div class="k">{i}</div><div><div class="t">{html.escape(f["text"])}</div>'
                   f'<div class="s">{f["ratio"]:.1f}x the rest of the selection; intervals do not overlap</div></div></div>')
    st.html("".join(out))


def meter(value: float, threshold: float, low_cut: float, colour: str) -> None:
    pct = max(0.0, min(1.0, value)) * 100
    st.html(f'<div class="ts-meter"><div class="fill" style="width:{pct:.1f}%;background:{colour}"></div>'
            f'<div class="tick" style="left:{low_cut * 100:.1f}%"></div><div class="tl" style="left:{low_cut * 100:.1f}%">'
            f'elevated {low_cut:.0%}</div>'
            f'<div class="tick" style="left:{threshold * 100:.1f}%"></div><div class="tl" style="left:{threshold * 100:.1f}%">'
            f'high {threshold:.0%}</div></div>')


def method_flow(steps: list[tuple[str, str, str]], colour: str = T.TEAL) -> None:
    out = ['<div class="ts-flow">']
    for i, (title, body, chip) in enumerate(steps, 1):
        c = T.CLAY if i == len(steps) else colour
        out.append(f'<div class="step" style="--c:{c}"><div class="node">{i:02d}</div><div><h5>{html.escape(title)}</h5>'
                   f'<p>{body}</p>' + (f'<span class="chip">{html.escape(chip)}</span>' if chip else "") + "</div></div>")
    out.append("</div>")
    st.html("".join(out))


def empty_state(title: str, body: str, icon: str = ":material/info:") -> None:
    with st.container(border=True):
        st.markdown(f"{icon} **{title}**")
        st.caption(body)


def status_badge(status: str) -> str:
    return {"pass": ":green-badge[:material/check: pass]", "warn": ":orange-badge[:material/warning: warn]",
            "fail": ":red-badge[:material/close: fail]"}.get(status, status)
