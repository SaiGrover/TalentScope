"""SHAP and permutation-importance explanations of fitted TalentScope pipelines."""
from .shap_tools import (
    ShapExplanation,
    aggregate_by_source,
    build_explainer,
    explain,
    explain_bundle,
    load_explanations,
    output_space,
    permutation_table,
)

__all__ = [
    "ShapExplanation", "build_explainer", "explain", "explain_bundle", "aggregate_by_source",
    "permutation_table", "load_explanations", "output_space",
]
