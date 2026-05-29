import pandas as pd

from src.pd_backtesting.metrics import metrics_by_portfolio, portfolio_metrics


def test_portfolio_metrics():
    observations = pd.DataFrame(
        {
            "portfolio": ["Retail", "Retail", "Corporate"],
            "pd_estimate": [0.01, 0.03, 0.02],
            "default_flag_12m": [0, 1, 0],
        }
    )

    metrics = portfolio_metrics(observations)

    assert metrics["observations"] == 3
    assert metrics["defaults"] == 1
    assert metrics["mean_pd"] == 0.02
    assert metrics["observed_default_rate"] == 1 / 3


def test_metrics_by_portfolio_returns_one_row_per_portfolio():
    observations = pd.DataFrame(
        {
            "portfolio": ["Retail", "Retail", "Corporate"],
            "pd_estimate": [0.01, 0.03, 0.02],
            "default_flag_12m": [0, 1, 0],
        }
    )

    summary = metrics_by_portfolio(observations)

    assert set(summary["portfolio"]) == {"Retail", "Corporate"}
    assert summary["observations"].sum() == 3
