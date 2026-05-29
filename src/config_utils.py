from __future__ import annotations

from copy import deepcopy


def apply_threshold_profile(thresholds: dict, profile_name: str = "standard") -> dict:
    """Return thresholds with the selected profile applied."""
    configured = deepcopy(thresholds)
    profiles = configured.get("threshold_profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown threshold profile: {profile_name}")

    profile = profiles[profile_name]
    binomial = configured.setdefault("calibration_tests", {}).setdefault("binomial", {})
    if "calibration_tests" in profile and "binomial" in profile["calibration_tests"]:
        binomial.update(profile["calibration_tests"]["binomial"])

    for section in ["discrimination", "stability", "minimum_volume"]:
        if section in profile:
            configured.setdefault(section, {}).update(profile[section])

    configured["active_threshold_profile"] = profile_name
    return configured


def validate_thresholds(thresholds: dict) -> list[str]:
    """Return human-readable validation errors for the threshold configuration."""
    required_paths = [
        ("calibration_tests", "binomial", "alpha_orange"),
        ("calibration_tests", "binomial", "alpha_red"),
        ("minimum_volume", "min_observations"),
        ("minimum_volume", "min_defaults"),
        ("minimum_volume", "min_non_defaults"),
        ("discrimination", "auc_orange"),
        ("discrimination", "auc_red"),
        ("stability", "psi_orange"),
        ("stability", "psi_red"),
    ]
    errors = []
    for path in required_paths:
        cursor = thresholds
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                errors.append("Missing threshold: " + ".".join(path))
                break
            cursor = cursor[key]

    if not errors:
        alpha_orange = thresholds["calibration_tests"]["binomial"]["alpha_orange"]
        alpha_red = thresholds["calibration_tests"]["binomial"]["alpha_red"]
        if alpha_red > alpha_orange:
            errors.append("calibration_tests.binomial.alpha_red must be <= alpha_orange")
        if thresholds["stability"]["psi_red"] < thresholds["stability"]["psi_orange"]:
            errors.append("stability.psi_red must be >= psi_orange")
    return errors
