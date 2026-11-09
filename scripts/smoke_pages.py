"""Headless smoke run of every page with Streamlit's AppTest (prints one line per page)."""
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["1_executive_overview", "2_workforce_explorer", "3_attrition_intelligence", "4_data_laboratory",
         "5_model_laboratory", "6_model_performance", "7_explainable_ai", "8_risk_simulator", "9_reports"]


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    failures = 0
    for p in PAGES:
        at.switch_page(f"app_pages/{p}.py").run()
        if at.exception:
            failures += 1
            e = at.exception[0]
            print(f"{p:28s} FAIL  {e.message[:400]}")
            print("    " + "\n    ".join(e.stack_trace[-8:]))
        else:
            print(f"{p:28s} ok    charts={len(at.get('plotly_chart'))} tables={len(at.dataframe)} "
                  f"errors={len(at.error)}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
