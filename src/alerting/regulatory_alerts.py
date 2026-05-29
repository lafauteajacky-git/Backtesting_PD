from __future__ import annotations

import pandas as pd


def status_from_metric(value: float, orange: float, red: float, higher_is_worse: bool = True) -> str:
    """Generic traffic-light helper for regulatory enrichment modules."""
    if pd.isna(value):
        return "grey"
    if higher_is_worse:
        if value >= red:
            return "red"
        if value >= orange:
            return "orange"
    else:
        if value <= red:
            return "red"
        if value <= orange:
            return "orange"
    return "green"


def assign_rds_status(psi: float, thresholds: dict) -> str:
    cfg = thresholds.get("rds", thresholds.get("stability", {}))
    return status_from_metric(psi, cfg.get("psi_orange", 0.10), cfg.get("psi_red", 0.25))


def assign_eligibility_status(exclusion_rate: float, thresholds: dict) -> str:
    cfg = thresholds.get("eligibility", {})
    return status_from_metric(exclusion_rate, cfg.get("exclusion_rate_orange", 0.05), cfg.get("exclusion_rate_red", 0.15))


def assign_migration_status(downgrade_rate: float, thresholds: dict) -> str:
    cfg = thresholds.get("migration", {})
    return status_from_metric(downgrade_rate, cfg.get("downgrade_rate_orange", 0.25), cfg.get("downgrade_rate_red", 0.40))


def assign_pd_adjustment_status(impact: float, thresholds: dict) -> str:
    cfg = thresholds.get("pd_adjustments", {})
    return status_from_metric(impact, cfg.get("impact_orange", 0.01), cfg.get("impact_red", 0.03))


def assign_low_default_status(observations: int, defaults: int, thresholds: dict) -> str:
    cfg = thresholds.get("low_default", {})
    if observations < cfg.get("min_observations", 30):
        return "grey"
    if defaults < cfg.get("min_defaults", 5):
        return "orange"
    return "green"


def assign_population_shift_status(max_share_change: float, thresholds: dict) -> str:
    cfg = thresholds.get("population_shift", {})
    return status_from_metric(max_share_change, cfg.get("share_change_orange", 0.10), cfg.get("share_change_red", 0.20))
