import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_PATH = Path("data/generated/pd_observations.csv")
RATING_GRADES = ["A", "B", "C", "D", "E", "F", "G"]
RETAIL_SEGMENTS = ["Mortgage", "Consumer Finance", "Credit Card", "Auto Loan"]
CORPORATE_SEGMENTS = ["Large Corporate", "SME", "Specialized Lending"]
GRADE_ORDER = {grade: index + 1 for index, grade in enumerate(RATING_GRADES)}


def _grade_pd_profile(grades: np.ndarray, portfolio: str) -> np.ndarray:
    retail_profile = {
        "A": 0.002,
        "B": 0.006,
        "C": 0.014,
        "D": 0.030,
        "E": 0.060,
        "F": 0.120,
        "G": 0.220,
    }
    corporate_profile = {
        "A": 0.001,
        "B": 0.003,
        "C": 0.007,
        "D": 0.014,
        "E": 0.030,
        "F": 0.060,
        "G": 0.110,
    }
    profile = retail_profile if portfolio == "Retail" else corporate_profile
    return np.array([profile[grade] for grade in grades])


def _simulate_portfolio(
    rng: np.random.Generator,
    portfolio: str,
    n_observations: int,
    id_offset: int,
    observation_start_year: int,
    observation_years: int,
) -> pd.DataFrame:
    if portfolio == "Retail":
        segments = RETAIL_SEGMENTS
        grade_probabilities = [0.10, 0.22, 0.28, 0.20, 0.12, 0.06, 0.02]
        products = ["Term Loan", "Revolving Credit", "Mortgage", "Lease"]
        model_id = "PD_RET_12M"
        default_multiplier = 1.02
        score_noise = 35
        ead_shape = (9.2, 0.65)
    else:
        segments = CORPORATE_SEGMENTS
        grade_probabilities = [0.18, 0.26, 0.24, 0.16, 0.10, 0.04, 0.02]
        products = ["Corporate Loan", "Credit Line", "Trade Finance", "Project Finance"]
        model_id = "PD_CORP_12M"
        default_multiplier = 0.55
        score_noise = 28
        ead_shape = (13.2, 0.85)

    observation_ids = np.arange(id_offset, id_offset + n_observations).astype(object)
    obligor_ids = np.array(
        [f"{portfolio[:3].upper()}_OBL_{value:07d}" for value in rng.integers(1, n_observations // 2, n_observations)]
    )
    facility_ids = np.array(
        [f"{portfolio[:3].upper()}_FAC_{value:08d}" for value in rng.integers(1, n_observations, n_observations)]
    )
    rating_grades = rng.choice(RATING_GRADES, size=n_observations, p=grade_probabilities)
    base_pd = _grade_pd_profile(rating_grades, portfolio)
    pd_estimate = np.clip(base_pd * rng.lognormal(mean=0.0, sigma=0.18, size=n_observations), 0.0001, 0.75)
    observed_pd = np.clip(pd_estimate * default_multiplier, 0.0001, 0.95)
    default_flag = rng.binomial(1, observed_pd).astype(float)

    observation_start = np.datetime64(f"{observation_start_year}-01-01")
    observation_days = rng.integers(0, 365 * observation_years, size=n_observations)
    observation_dates = observation_start + observation_days.astype("timedelta64[D]")
    origination_offsets = rng.integers(30, 365 * 5, size=n_observations).astype("timedelta64[D]")
    origination_dates = observation_dates - origination_offsets

    default_dates = np.full(n_observations, np.datetime64("NaT"), dtype="datetime64[ns]")
    default_offsets = rng.integers(1, 366, size=n_observations).astype("timedelta64[D]")
    default_dates[default_flag == 1] = observation_dates[default_flag == 1] + default_offsets[default_flag == 1]

    score = np.clip(
        820 - (pd_estimate * 1600) + rng.normal(0, score_noise, size=n_observations),
        250,
        900,
    )

    frame = pd.DataFrame(
        {
            "observation_id": observation_ids,
            "obligor_id": obligor_ids,
            "facility_id": facility_ids,
            "portfolio": portfolio,
            "segment": rng.choice(segments, size=n_observations),
            "product_type": rng.choice(products, size=n_observations),
            "model_id": model_id,
            "model_version": rng.choice(["v1.0", "v1.1"], size=n_observations, p=[0.35, 0.65]),
            "observation_date": pd.to_datetime(observation_dates),
            "origination_date": pd.to_datetime(origination_dates),
            "performance_window_months": 12,
            "rating_grade": rating_grades,
            "score": np.round(score, 0),
            "pd_estimate": np.round(pd_estimate, 6),
            "ead_at_observation": np.round(rng.lognormal(*ead_shape, size=n_observations), 2),
            "default_flag_12m": default_flag,
            "default_date": pd.to_datetime(default_dates),
            "exclusion_flag": rng.choice([0, 1], size=n_observations, p=[0.985, 0.015]),
            "exclusion_reason": None,
        }
    )
    frame.loc[frame["exclusion_flag"] == 1, "exclusion_reason"] = rng.choice(
        ["Fraud", "Restructuring", "Incomplete history"],
        size=int(frame["exclusion_flag"].sum()),
    )
    return _enrich_regulatory_schema(frame, rng, portfolio, observation_start_year, observation_years)


def _bucketize(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype(object)


def _enrich_regulatory_schema(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    portfolio: str,
    observation_start_year: int,
    observation_years: int,
) -> pd.DataFrame:
    """Add regulatory, rating-system and migration fields to the synthetic PD dataset."""
    output = frame.copy()
    n = len(output)
    observation_dates = pd.to_datetime(output["observation_date"], errors="coerce")
    observation_year = observation_dates.dt.year.astype("Int64")
    output["snapshot_id"] = observation_year.astype(str).replace("<NA>", "unknown").map(lambda year: f"SNAP_{year}")
    output["as_of_date"] = pd.to_datetime(observation_year.astype(str) + "-12-31", errors="coerce")
    output["observation_year"] = observation_year
    output["previous_observation_date"] = observation_dates - pd.DateOffset(years=1)

    output["regulatory_exposure_class"] = portfolio
    output["irb_approach"] = "AIRB" if portfolio == "Retail" else rng.choice(["AIRB", "FIRB"], size=n, p=[0.35, 0.65])
    output["application_scope_flag"] = rng.choice([1, 0], size=n, p=[0.975, 0.025])
    output["model_applicability_flag"] = rng.choice([1, 0], size=n, p=[0.965, 0.035])
    status = rng.choice(["active", "closed", "matured", "sold", "defaulted"], size=n, p=[0.90, 0.035, 0.035, 0.01, 0.02])
    output["exposure_status_at_observation"] = status
    output["performing_status_at_observation"] = np.where(
        status == "defaulted",
        "defaulted",
        rng.choice(["performing", "non_performing", "forborne"], size=n, p=[0.94, 0.035, 0.025]),
    )
    output["default_flag_at_observation"] = (output["exposure_status_at_observation"] == "defaulted").astype(int)
    output["data_complete_12m_flag"] = rng.choice([1, 0], size=n, p=[0.975, 0.025])
    output["maturity_before_12m_flag"] = rng.choice([0, 1], size=n, p=[0.965, 0.035])
    output["closure_date"] = pd.NaT
    closing_mask = output["exposure_status_at_observation"].isin(["closed", "matured", "sold"])
    closure_offsets = rng.integers(10, 330, size=n).astype("timedelta64[D]")
    output.loc[closing_mask, "closure_date"] = observation_dates.loc[closing_mask].to_numpy() + closure_offsets[closing_mask]

    output["rating_system_id"] = "RS_RET_01" if portfolio == "Retail" else "RS_CORP_01"
    output["rating_method"] = rng.choice(
        ["scorecard", "expert_judgement", "hybrid", "statistical_model"],
        size=n,
        p=[0.45, 0.12, 0.18, 0.25] if portfolio == "Retail" else [0.15, 0.42, 0.28, 0.15],
    )
    output["model_use_type"] = rng.choice(
        ["application", "behavioural", "corporate_rating", "monitoring"],
        size=n,
        p=[0.32, 0.42, 0.04, 0.22] if portfolio == "Retail" else [0.08, 0.10, 0.58, 0.24],
    )
    output["model_philosophy"] = rng.choice(["PIT", "TTC", "Hybrid"], size=n, p=[0.42, 0.28, 0.30] if portfolio == "Retail" else [0.20, 0.52, 0.28])
    output["pd_type"] = rng.choice(
        ["regulatory_12m", "origination_12m", "behavioural_12m", "monitoring_12m"],
        size=n,
        p=[0.45, 0.15, 0.28, 0.12] if portfolio == "Retail" else [0.58, 0.08, 0.08, 0.26],
    )
    output["pd_horizon_months"] = 12
    output["calibration_pool_id"] = np.where(output["portfolio"].eq("Retail"), "POOL_RET", "POOL_CORP")
    assignment_offsets = rng.integers(0, 760, size=n).astype("timedelta64[D]")
    output["rating_assignment_date"] = observation_dates.to_numpy() - assignment_offsets
    review_offsets = rng.integers(0, 420, size=n).astype("timedelta64[D]")
    output["last_review_date"] = observation_dates.to_numpy() - review_offsets
    output["rating_age_months"] = np.maximum(((observation_dates - output["rating_assignment_date"]).dt.days / 30.4).round(0), 0)
    output["stale_rating_flag"] = (output["rating_age_months"] > 15).astype(int)

    output["grade_order"] = output["rating_grade"].map(GRADE_ORDER).astype("Int64")
    output["master_scale_grade"] = output["rating_grade"].map(lambda grade: f"MS_{grade}" if pd.notna(grade) else pd.NA)
    output["score_bucket"] = _bucketize(output["score"], [0, 500, 600, 700, 800, 1000], ["S1", "S2", "S3", "S4", "S5"])
    output["pd_bucket"] = _bucketize(output["pd_estimate"], [0, 0.005, 0.015, 0.04, 0.10, 1.0], ["PD1", "PD2", "PD3", "PD4", "PD5"])
    year_center = observation_start_year + (max(observation_years, 1) - 1) / 2
    cycle = (observation_year.astype("float").fillna(year_center) - year_center) / max(observation_years - 1, 1)
    pit_multiplier = 1.0 + 0.42 * cycle
    ttc_multiplier = 1.0 + 0.08 * cycle
    hybrid_multiplier = 1.0 + 0.22 * cycle
    philosophy_multiplier = np.select(
        [
            output["model_philosophy"].eq("PIT"),
            output["model_philosophy"].eq("TTC"),
            output["model_philosophy"].eq("Hybrid"),
        ],
        [pit_multiplier, ttc_multiplier, hybrid_multiplier],
        default=1.0,
    )
    output["pd_estimate"] = np.clip(pd.to_numeric(output["pd_estimate"], errors="coerce") * philosophy_multiplier, 0.0001, 1.0)

    raw_noise = rng.normal(0.0, 0.08, size=n)
    output["pd_raw"] = np.clip(output["pd_estimate"].astype(float) * (1 + raw_noise), 0.0001, 1.0)
    output["pd_calibrated"] = np.clip(output["pd_raw"] * rng.lognormal(0.0, 0.05, size=n), 0.0001, 1.0)
    output["margin_of_conservatism_type"] = rng.choice(
        ["none", "data_quality", "methodology", "representativeness", "conservatism_overlay"],
        size=n,
        p=[0.72, 0.08, 0.08, 0.07, 0.05],
    )
    output["margin_of_conservatism"] = np.where(output["margin_of_conservatism_type"].eq("none"), 0.0, rng.uniform(0.0002, 0.006, size=n))
    floor_values = np.where(output["portfolio"].eq("Retail"), 0.0005, 0.0003)
    output["pd_floor_value"] = floor_values
    output["pd_before_floor"] = np.clip(output["pd_calibrated"] + output["margin_of_conservatism"], 0.0001, 1.0)
    output["pd_after_floor"] = np.maximum(output["pd_before_floor"], output["pd_floor_value"])
    output["pd_floor_applied_flag"] = (output["pd_after_floor"] > output["pd_before_floor"]).astype(int)
    output["pd_regulatory"] = np.clip(output["pd_after_floor"], 0.0001, 1.0)

    grade_shift = rng.choice([-2, -1, 0, 1, 2], size=n, p=[0.05, 0.16, 0.58, 0.16, 0.05])
    previous_order = np.clip(output["grade_order"].fillna(4).astype(int) - grade_shift, 1, len(RATING_GRADES))
    new_customer = rng.choice([0, 1], size=n, p=[0.92, 0.08])
    exited_customer = rng.choice([0, 1], size=n, p=[0.94, 0.06])
    output["previous_grade_order"] = previous_order
    output.loc[new_customer == 1, "previous_grade_order"] = pd.NA
    inverse_grade = {value: key for key, value in GRADE_ORDER.items()}
    output["previous_rating_grade"] = output["previous_grade_order"].map(inverse_grade)
    output["previous_score"] = np.clip(output["score"].astype(float) + rng.normal(0, 35, size=n), 250, 900)
    output["previous_pd_estimate"] = np.clip(output["pd_estimate"].astype(float) * rng.lognormal(0, 0.20, size=n), 0.0001, 1.0)
    output.loc[new_customer == 1, ["previous_score", "previous_pd_estimate"]] = np.nan
    output["previous_pd_bucket"] = _bucketize(output["previous_pd_estimate"], [0, 0.005, 0.015, 0.04, 0.10, 1.0], ["PD1", "PD2", "PD3", "PD4", "PD5"])
    output["notch_change"] = output["grade_order"].astype("float") - output["previous_grade_order"].astype("float")
    output["new_customer_flag"] = new_customer
    output["exited_customer_flag"] = exited_customer
    output["rating_change_direction"] = np.select(
        [new_customer == 1, exited_customer == 1, output["notch_change"] > 0, output["notch_change"] < 0],
        ["new", "exited", "worse", "better"],
        default="stable",
    )
    output["rating_migration"] = np.select(
        [new_customer == 1, exited_customer == 1, output["notch_change"] > 0, output["notch_change"] < 0],
        ["new", "exited", "downgrade", "upgrade"],
        default="stable",
    )

    output["country_code"] = rng.choice(["FR", "DE", "IT", "ES", "BE", "NL"], size=n, p=[0.58, 0.12, 0.10, 0.08, 0.07, 0.05])
    output["country_of_risk"] = output["country_code"]
    output["sector_code"] = rng.choice(["A", "C", "F", "G", "J", "K", "L", "M"], size=n)
    output["borrower_size_class"] = rng.choice(["micro", "small", "medium", "large"], size=n, p=[0.35, 0.30, 0.22, 0.13] if portfolio == "Retail" else [0.04, 0.23, 0.38, 0.35])
    output["secured_flag"] = rng.choice([1, 0], size=n, p=[0.62, 0.38] if portfolio == "Retail" else [0.48, 0.52])
    output["collateral_type"] = np.where(output["secured_flag"].eq(1), rng.choice(["real_estate", "financial", "guarantee", "other"], size=n), "none")
    output["ltv_bucket"] = np.where(output["secured_flag"].eq(1), rng.choice(["0-50", "50-80", "80-100", ">100"], size=n, p=[0.32, 0.42, 0.20, 0.06]), "not_secured")
    output["origination_channel"] = rng.choice(["branch", "broker", "digital", "relationship_manager"], size=n)
    output["watchlist_flag"] = rng.choice([0, 1], size=n, p=[0.91, 0.09])
    output["forbearance_flag_at_observation"] = (output["performing_status_at_observation"] == "forborne").astype(int)
    output["npe_flag_at_observation"] = output["performing_status_at_observation"].isin(["non_performing", "defaulted"]).astype(int)
    output["days_past_due_at_observation"] = np.where(output["npe_flag_at_observation"].eq(1), rng.integers(1, 120, size=n), rng.integers(0, 20, size=n))

    default_reasons = rng.choice(["past_due_90", "unlikely_to_pay", "bankruptcy", "restructuring", "other"], size=n, p=[0.45, 0.30, 0.06, 0.13, 0.06])
    output["default_reason"] = np.where(output["default_flag_12m"].eq(1), default_reasons, None)
    output["past_due_90_flag_12m"] = np.where(output["default_flag_12m"].eq(1), (output["default_reason"] == "past_due_90").astype(int), 0)
    output["utp_flag_12m"] = np.where(output["default_flag_12m"].eq(1), output["default_reason"].isin(["unlikely_to_pay", "bankruptcy", "restructuring"]).astype(int), 0)
    output["max_days_past_due_12m"] = np.where(output["default_flag_12m"].eq(1), rng.integers(90, 260, size=n), rng.integers(0, 60, size=n))
    output["months_to_default"] = np.where(output["default_flag_12m"].eq(1), rng.integers(1, 13, size=n), np.nan)
    output["default_exit_date"] = pd.NaT
    cured = output["default_flag_12m"].eq(1) & (rng.random(n) < 0.38)
    output.loc[cured, "default_exit_date"] = pd.to_datetime(output.loc[cured, "default_date"]) + pd.to_timedelta(rng.integers(30, 360, size=int(cured.sum())), unit="D")
    output["cure_flag"] = cured.astype(int)

    output["exclusion_category"] = np.where(output["exclusion_flag"].eq(1), rng.choice(["data_quality", "out_of_scope", "defaulted_at_observation", "incomplete_performance_window", "closed_before_horizon", "model_not_applicable", "other"], size=n), "none")
    output.loc[output["default_flag_at_observation"].eq(1), ["exclusion_flag", "exclusion_category"]] = [1, "defaulted_at_observation"]
    output.loc[output["application_scope_flag"].eq(0), ["exclusion_flag", "exclusion_category"]] = [1, "out_of_scope"]
    output.loc[output["model_applicability_flag"].eq(0), ["exclusion_flag", "exclusion_category"]] = [1, "model_not_applicable"]
    output.loc[output["data_complete_12m_flag"].eq(0), ["exclusion_flag", "exclusion_category"]] = [1, "incomplete_performance_window"]
    output.loc[output["maturity_before_12m_flag"].eq(1), ["exclusion_flag", "exclusion_category"]] = [1, "closed_before_horizon"]
    output["exclusion_reason"] = np.where(output["exclusion_flag"].eq(1), output["exclusion_category"], output["exclusion_reason"])
    output["exclusion_materiality"] = np.where(output["exclusion_flag"].eq(1), rng.choice(["low", "medium", "high"], size=n, p=[0.45, 0.38, 0.17]), "none")
    output["exclusion_applied_by_rule"] = output["exclusion_flag"].astype(int)
    output["exclusion_rule_id"] = np.where(output["exclusion_flag"].eq(1), "RULE_" + output["exclusion_category"].astype(str).str.upper(), None)

    return output


def _inject_data_quality_anomalies(observations: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    anomalous = observations.copy()
    indexes = rng.choice(anomalous.index, size=80, replace=False)

    anomalous.loc[indexes[0:5], "observation_id"] = np.nan
    anomalous.loc[indexes[5:10], "observation_id"] = anomalous.loc[indexes[10], "observation_id"]
    anomalous.loc[indexes[10:18], "pd_estimate"] = np.nan
    anomalous.loc[indexes[18:25], "pd_estimate"] = rng.choice([-0.01, 0.0, 1.2], size=7)
    anomalous.loc[indexes[25:32], "default_flag_12m"] = np.nan
    anomalous.loc[indexes[32:38], "observation_date"] = pd.NaT
    anomalous.loc[indexes[38:43], "portfolio"] = np.nan
    anomalous.loc[indexes[43:48], "rating_grade"] = np.nan
    anomalous.loc[indexes[48:55], "performance_window_months"] = rng.choice([0, 6, 18], size=7)

    default_rows = anomalous.index[anomalous["default_flag_12m"] == 1].to_numpy()
    if len(default_rows) >= 8:
        missing_default_date_rows = rng.choice(default_rows, size=8, replace=False)
        anomalous.loc[missing_default_date_rows, "default_date"] = pd.NaT

    return anomalous


def generate_pd_observations(
    retail_observations: int = 30000,
    corporate_observations: int = 5000,
    seed: int = 42,
    include_anomalies: bool = True,
    observation_start_year: int = 2019,
    observation_years: int = 5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    retail = _simulate_portfolio(
        rng,
        "Retail",
        retail_observations,
        1,
        observation_start_year,
        observation_years,
    )
    corporate = _simulate_portfolio(
        rng,
        "Corporate",
        corporate_observations,
        retail_observations + 1,
        observation_start_year,
        observation_years,
    )
    observations = pd.concat([retail, corporate], ignore_index=True)

    if include_anomalies:
        observations = _inject_data_quality_anomalies(observations, rng)

    return observations.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fictitious 12-month PD observations.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    parser.add_argument("--retail", type=int, default=30000, help="Retail observation count.")
    parser.add_argument("--corporate", type=int, default=5000, help="Corporate observation count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--start-year", type=int, default=2019, help="First observation year.")
    parser.add_argument("--years", type=int, default=5, help="Number of observation years.")
    parser.add_argument("--no-anomalies", action="store_true", help="Disable deliberate DQ anomalies.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observations = generate_pd_observations(
        retail_observations=args.retail,
        corporate_observations=args.corporate,
        seed=args.seed,
        include_anomalies=not args.no_anomalies,
        observation_start_year=args.start_year,
        observation_years=args.years,
    )
    observations.to_csv(output_path, index=False)
    print(f"Generated {len(observations):,} observations at {output_path}")


if __name__ == "__main__":
    main()
