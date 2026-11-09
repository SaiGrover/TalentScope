"""TalentScope - Employee Attrition Intelligence & Workforce Analytics.

The package is organised by responsibility:

- ``data``            loading, dataset card, schema and validation
- ``preprocessing``   leakage-safe feature pipelines and the stratified split
- ``modeling``        model registry, resampling strategies and the experiment runner
- ``evaluation``      metrics, bootstrap intervals, curves and threshold selection
- ``explainability``  SHAP and permutation importance on fitted pipelines
- ``analytics``       workforce and attrition analytics used by the dashboard
- ``reporting``       downloadable reports and model cards built from artefacts
- ``ui``              design tokens, Plotly styling and reusable Streamlit components

The Streamlit app, the notebooks and the test-suite all import from here so
that every number shown anywhere comes from one implementation.
"""

__version__ = "2.0.0"
