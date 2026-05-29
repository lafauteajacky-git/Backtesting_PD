from pathlib import Path

import pandas as pd


DATE_COLUMNS = [
    "observation_date",
    "default_date",
    "origination_date",
    "as_of_date",
    "previous_observation_date",
    "rating_assignment_date",
    "last_review_date",
    "closure_date",
    "default_exit_date",
]


def load_observations(path: str | Path) -> pd.DataFrame:
    """Load generated PD observations from a CSV file."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    columns = pd.read_csv(data_path, nrows=0).columns
    date_columns = [column for column in DATE_COLUMNS if column in columns]
    observations = pd.read_csv(data_path, parse_dates=date_columns)
    if "origination_date" not in observations.columns:
        observations["origination_date"] = pd.NaT
    return observations


def load_thresholds(path: str | Path = "config/thresholds.yaml") -> dict:
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
