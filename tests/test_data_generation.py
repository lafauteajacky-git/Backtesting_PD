from src.data_generation.generate_pd_observations import generate_pd_observations


def test_generator_returns_required_columns_and_minimum_volumes():
    observations = generate_pd_observations(
        retail_observations=30000,
        corporate_observations=5000,
        seed=7,
        include_anomalies=False,
    )

    required_columns = {
        "observation_id",
        "obligor_id",
        "facility_id",
        "portfolio",
        "segment",
        "product_type",
        "model_id",
        "model_version",
        "observation_date",
        "performance_window_months",
        "rating_grade",
        "score",
        "pd_estimate",
        "ead_at_observation",
        "default_flag_12m",
        "default_date",
        "exclusion_flag",
        "exclusion_reason",
    }

    assert required_columns.issubset(observations.columns)
    assert len(observations) == 35000
    assert (observations["portfolio"] == "Retail").sum() == 30000
    assert (observations["portfolio"] == "Corporate").sum() == 5000
    assert observations["observation_date"].dt.year.nunique() >= 5


def test_generator_injects_data_quality_anomalies():
    observations = generate_pd_observations(
        retail_observations=30000,
        corporate_observations=5000,
        seed=7,
        include_anomalies=True,
    )

    assert observations["observation_id"].isna().sum() > 0
    assert observations["pd_estimate"].isna().sum() > 0
    assert (~observations["pd_estimate"].dropna().between(0, 1, inclusive="right")).sum() > 0
