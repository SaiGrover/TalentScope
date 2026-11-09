"""TalentScope - Employee Attrition Intelligence & Workforce Analytics.

    streamlit run app.py

Entry point: sets the page config, applies the design layer, draws the brand
bar and the navigation rail, then runs the selected page from ``app_pages/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _reload_changed_package() -> None:
    """Streamlit re-executes page scripts but keeps imported modules. When any file under
    src/talentscope changes, drop the package from sys.modules so the next run imports it fresh."""
    stamp = max((p.stat().st_mtime for p in (ROOT / "src" / "talentscope").rglob("*.py")), default=0.0)
    if getattr(sys, "_talentscope_src_mtime", None) not in (None, stamp):
        for name in [m for m in sys.modules if m == "talentscope" or m.startswith("talentscope.")]:
            del sys.modules[name]
    sys._talentscope_src_mtime = stamp


_reload_changed_package()

import streamlit as st  # noqa: E402

from talentscope.ui import components as ui  # noqa: E402
from talentscope.ui import tokens as T  # noqa: E402
from talentscope.ui.state import dataset  # noqa: E402

st.set_page_config(page_title="TalentScope", page_icon=":material/radar:", layout="wide",
                   initial_sidebar_state="collapsed")

P = "app_pages/"
UNDERSTAND = [
    st.Page(P + "1_executive_overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page(P + "2_workforce_explorer.py", title="Workforce", icon=":material/groups:", url_path="workforce"),
    st.Page(P + "3_attrition_intelligence.py", title="Attrition", icon=":material/trending_down:", url_path="attrition"),
    st.Page(P + "4_data_laboratory.py", title="Data lab", icon=":material/biotech:", url_path="data-lab"),
]
MODEL = [
    st.Page(P + "5_model_laboratory.py", title="Model lab", icon=":material/science:", url_path="model-lab"),
    st.Page(P + "6_model_performance.py", title="Performance", icon=":material/monitoring:", url_path="performance"),
    st.Page(P + "7_explainable_ai.py", title="Explainability", icon=":material/psychology:", url_path="explainability"),
]
ACT = [
    st.Page(P + "8_risk_simulator.py", title="Risk simulator", icon=":material/person_search:", url_path="simulator"),
    st.Page(P + "9_reports.py", title="Reports", icon=":material/description:", url_path="reports"),
]

page = st.navigation(UNDERSTAND + MODEL + ACT, position="hidden")

ui.inject_css()
try:
    n_rows = len(dataset())
except FileNotFoundError:
    n_rows = None
ui.brand_bar(n_rows)
ui.nav_rail([("Understand", T.TEAL, UNDERSTAND), ("Model", T.BLUE, MODEL), ("Act", T.CLAY, ACT)], page.title)

if n_rows is None:
    st.error("The dataset file `data/raw/ibm_hr_employee_attrition.csv` is missing. See `data/README.md`.",
             icon=":material/error:")
    st.stop()

page.run()
