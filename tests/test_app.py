"""Streamlit smoke tests: every page renders, filters and forms work, and no legacy code remains."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(p.stem for p in (ROOT / "app_pages").glob("*.py"))


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    app.switch_page(f"app_pages/{page}.py").run()
    assert not app.exception, app.exception[0].message if app.exception else ""


def test_nine_pages():
    assert len(PAGES) == 9


def test_department_filter_changes_overview(app):
    app.switch_page("app_pages/1_executive_overview.py").run()
    captions = " ".join(c.value for c in app.caption)
    assert "All 1,470 employees" in captions
    app.multiselect(key="f_department").set_value(["Sales"]).run()
    captions = " ".join(c.value for c in app.caption)
    assert "446 of 1,470 employees" in captions
    app.multiselect(key="f_department").set_value([]).run()


def test_simulator_validates_inputs(app):
    app.switch_page("app_pages/8_risk_simulator.py").run()
    assert not app.exception
    app.number_input(key="sim_YearsAtCompany").set_value(40)
    app.number_input(key="sim_TotalWorkingYears").set_value(5)
    app.form_submit_button[0].click().run()
    text = " ".join(m.value for m in app.markdown)
    assert "cannot exceed" in text


def test_reports_offer_downloads(app):
    app.switch_page("app_pages/9_reports.py").run()
    assert not app.exception
    labels = [b.proto.label for b in app.get("download_button")]
    assert labels.count("Markdown") >= 4 and labels.count("HTML") >= 4


def test_no_flask_or_job_change_leftovers():
    pattern = re.compile(r"flask|enrollee|aug_train|city_development|render_template|gunicorn", re.I)
    for path in list(ROOT.glob("*.py")) + list((ROOT / "app_pages").glob("*.py")) + list((ROOT / "src").rglob("*.py")):
        assert not pattern.search(path.read_text(encoding="utf-8")), path
    assert not (ROOT / "template").exists() and not (ROOT / "pipeline").exists()


def test_no_hardcoded_performance_numbers_in_pages():
    """Model metrics must come from artefacts. Pages may not contain literal metric values like 'AUC 0.81'."""
    pattern = re.compile(r"(PR-AUC|ROC-AUC|F1|recall|precision|accuracy)\W{1,3}0\.\d{2}", re.I)
    for path in (ROOT / "app_pages").glob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path
