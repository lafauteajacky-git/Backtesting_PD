import pandas as pd


def combine_statuses(statuses: list[str]) -> str:
    """Combine traffic-light statuses into one global status."""
    usable = [status for status in statuses if pd.notna(status)]
    interpretable = [status for status in usable if status != "grey"]
    if "red" in interpretable:
        return "red"
    if "orange" in interpretable:
        return "orange"
    if not interpretable:
        return "grey"
    return "green"
