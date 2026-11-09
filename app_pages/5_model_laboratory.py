"""Model laboratory: configure, launch and save reproducible experiments."""
import streamlit as st

from talentscope import config
from talentscope.modeling import MODEL_SPECS, STRATEGIES, ExperimentConfig
from talentscope.modeling.artifacts import delete_run
from talentscope.ui import components as ui
from talentscope.ui import jobs
from talentscope.ui import tokens as T
from talentscope.ui.state import bundle, run_dirs, run_label

ui.page_header(
    "05 · Model", "Model laboratory",
    "Choose algorithms and imbalance strategies, set the search budget and launch a run. Training happens in the "
    "background on the training split only; you can keep using the other pages while it works. Every run is saved "
    "with its configuration, seeds, fold scores and model, and can be opened on Performance and Explainability.")

c_form, c_side = st.columns([1.55, 1], gap="large")

with c_form:
    with st.form("lab_form", border=False):
        with ui.card("Algorithms and imbalance handling", key="lab_algos"):
            models = st.pills("Algorithms", list(MODEL_SPECS), default=["dummy", "logreg", "xgboost"],
                              format_func=lambda k: MODEL_SPECS[k].name, selection_mode="multi", key="lab_models")
            strategies = st.pills("Class-imbalance strategies", list(STRATEGIES), default=["none", "class_weight", "smotenc"],
                                  format_func=lambda k: STRATEGIES[k].name, selection_mode="multi", key="lab_strats")
            st.caption("The dummy baseline runs once, without rebalancing. Every other algorithm is crossed with every "
                       "selected strategy.")
        with ui.card("SMOTENC and oversampling", "applied inside each training fold only", key="lab_smote"):
            s1, s2 = st.columns(2)
            ratio = s1.slider("Minority : majority after resampling", 0.3, 1.0, 1.0, 0.05, key="lab_ratio",
                              help="1.0 balances the classes. Lower values add fewer synthetic leavers.")
            k_nn = s2.slider("SMOTENC neighbours (k)", 1, 10, 5, key="lab_k",
                             help="Synthetic leavers are interpolated between a leaver and one of its k nearest leavers.")
        with ui.card("Search and evaluation", key="lab_search"):
            s1, s2, s3 = st.columns(3)
            n_iter = s1.slider("Search candidates per configuration", 1, 40, 8, key="lab_iter",
                               help="RandomizedSearchCV candidates, scored by PR-AUC with stratified K-fold.")
            folds = s2.select_slider("CV folds", [3, 4, 5, 10], value=5, key="lab_folds")
            repeats = s3.select_slider("CV repeats", [1, 2, 3], value=1, key="lab_rep",
                                       help="Repeated stratified CV used to compare the tuned configurations.")
            s4, s5, s6 = st.columns(3)
            thr_metric = s4.segmented_control("Threshold rule", ["f1", "f2"], default="f1", key="lab_thr",
                                              format_func=lambda m: {"f1": "Maximise F1", "f2": "Maximise F2"}[m],
                                              help="F2 weighs recall twice as much as precision.")
            rule = s5.segmented_control("Model selection", ["one_se", "best"], default="one_se", key="lab_rule",
                                        format_func=lambda m: {"one_se": "1-SE rule", "best": "Best mean"}[m],
                                        help="1-SE picks the simplest configuration within one standard error of the best.")
            seed = s6.number_input("Random seed", 0, 99999, config.RANDOM_SEED, key="lab_seed",
                                   help="Changing the seed changes the train/test split too.")
            st.caption("Primary metric: average precision (PR-AUC). Thresholds come from out-of-fold training "
                       "predictions. The held-out 20% is scored once, for the selected configuration only.")
        submitted = st.form_submit_button("Launch experiment", type="primary", icon=":material/play_arrow:",
                                          width="stretch")

    if submitted:
        bad = []
        if not models:
            bad.append("choose at least one algorithm")
        if not strategies:
            bad.append("choose at least one strategy")
        if models and set(models) == {"dummy"}:
            bad.append("add a real algorithm; the dummy baseline cannot be selected")
        if bad:
            st.error("Cannot launch: " + "; ".join(bad) + ".", icon=":material/error:")
        else:
            cfg = ExperimentConfig(models=list(models), strategies=list(strategies), n_iter=int(n_iter),
                                   cv_folds=int(folds), cv_repeats=int(repeats), seed=int(seed),
                                   smote_sampling_strategy=float(ratio), smote_k_neighbors=int(k_nn),
                                   threshold_metric=thr_metric or "f1", selection_rule=rule or "one_se",
                                   bootstrap_resamples=500, name="lab",
                                   # one process: a worker pool inside the web server would starve the UI and,
                                   # on Streamlit Community Cloud, exceed the memory limit
                                   n_jobs=1)
            try:
                job = jobs.submit(cfg)
                st.session_state["lab_job"] = job.id
                st.toast("Experiment started", icon=":material/rocket_launch:")
            except RuntimeError as exc:
                st.warning(str(exc), icon=":material/hourglass_top:")

with c_side:
    n_combos = len(ExperimentConfig(models=list(st.session_state.get("lab_models") or []),
                                    strategies=list(st.session_state.get("lab_strats") or [])).combos())
    fits = n_combos * (st.session_state.get("lab_iter", 8) * st.session_state.get("lab_folds", 5)
                       + st.session_state.get("lab_folds", 5) * st.session_state.get("lab_rep", 1))
    ui.tiles([("Configurations", f"{n_combos}", "model x strategy pairs", T.BLUE),
              ("Model fits", f"{fits:,}", "search + comparison", T.AMBER)])

    @st.fragment(run_every=1.5)
    def job_panel():
        job = jobs.get(st.session_state.get("lab_job")) or jobs.latest()
        with ui.card("Training status", key="lab_status"):
            if job is None:
                st.caption("No experiment launched in this session.")
                return
            icon = {"running": ":material/progress_activity:", "queued": ":material/hourglass_top:",
                    "done": ":material/check_circle:", "failed": ":material/error:"}[job.status]
            st.markdown(f"{icon} **{job.status.capitalize()}** · run `{job.id}`")
            st.progress(min(1.0, job.progress), text=job.message)
            if job.status == "failed":
                st.error(job.error, icon=":material/error:")
            if job.status == "done" and job.run_dir is not None:
                if st.button("Open results on Performance", type="primary", icon=":material/monitoring:",
                             key="lab_open", width="stretch"):
                    st.session_state["active_run"] = str(job.run_dir)
                    st.switch_page("app_pages/6_model_performance.py")
            with st.expander("Log"):
                st.code("\n".join(job.log[-40:]) or "(empty)", language=None)

    job_panel()

    with ui.card("Saved experiments", key="lab_runs"):
        runs = run_dirs()
        if not runs:
            st.caption("None yet.")
        for p in runs:
            b = bundle(p)
            if b is None:
                continue
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown(f"**{run_label(p)}**  \n{b.label} · CV PR-AUC {b.selection['cv_pr_auc_mean']:.3f} · "
                        f"test PR-AUC {b.test_metrics['pr_auc']:.3f}")
            if p.resolve() != config.DEFAULT_EXPERIMENT_DIR.resolve():
                if c2.button("Delete", key=f"del_{p.name}", icon=":material/delete:"):
                    delete_run(p)
                    if st.session_state.get("active_run") == str(p):
                        st.session_state.pop("active_run")
                    st.rerun()
            else:
                c2.caption("committed")

ui.section("The protocol behind every run", "identical for the default model and for lab runs")
ui.method_flow([
    ("Split once, stratified", "80% train, 20% test, stratified on attrition with the run's seed. The test rows are "
     "set aside before any imputer, scaler, encoder or sampler is fitted.", "train 1,176 · test 294"),
    ("Pipeline per fold", "impute → resample (optional) → round & scale → one-hot → model, as one imblearn pipeline. "
     "Cross-validation refits the whole chain on each training fold, so SMOTENC never sees a validation row.",
     "no leakage"),
    ("Randomised search", "Each algorithm × strategy is tuned with RandomizedSearchCV over a small, meaningful space, "
     "scored by PR-AUC on stratified folds of the training set.", "PR-AUC"),
    ("Repeated CV comparison", "The tuned configurations are re-scored on fresh repeated stratified folds (a different "
     "seed from the search). Fold scores give mean ± SD; averaged out-of-fold probabilities give curves.",
     "mean ± SD"),
    ("Select model and threshold", "Highest mean PR-AUC, or the simplest configuration within one standard error. The "
     "decision threshold maximises F1 (or F2) on out-of-fold training predictions.", "training data only"),
    ("Score the test set once", "Refit the selected pipeline on all training rows and score the untouched test set. "
     "Stratified bootstrap gives 95% intervals. Everything is written to disk.", "held-out estimate"),
], T.BLUE)
