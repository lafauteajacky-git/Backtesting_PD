from src.data_generation.demo_scenarios import REQUIRED_COLUMNS


def validate_observation_schema(columns) -> list[str]:
    """Return missing required columns for the generated observation dataset."""
    available = set(columns)
    return [column for column in REQUIRED_COLUMNS if column not in available]
