"""Workforce and attrition analytics. Pure functions over the prepared frame."""
from .workforce import (
    DIMENSIONS,
    apply_filters,
    association_ranking,
    correlation_matrix,
    cramers_v_matrix,
    crosstab_rates,
    headline_findings,
    numeric_comparison,
    outlier_table,
    rate_by,
    workforce_kpis,
)

__all__ = [
    "DIMENSIONS", "apply_filters", "workforce_kpis", "rate_by", "crosstab_rates", "association_ranking",
    "numeric_comparison", "headline_findings", "correlation_matrix", "cramers_v_matrix", "outlier_table",
]
