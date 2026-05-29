from pathlib import Path

import base64
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy.stats import chi2, chi2_contingency

from src.alerting.calibration_alerts import assign_status, build_calibration_alerts
from src.alerting.discrimination_alerts import build_discrimination_alerts
from src.alerting.global_status import combine_statuses
from src.alerting.monotonicity_alerts import build_monotonicity_alert
from src.alerting.stability_alerts import build_stability_alerts
from src.config_utils import apply_threshold_profile, validate_thresholds
from src.data_generation.demo_scenarios import REQUIRED_COLUMNS, generate_demo_scenario, scenario_catalog
from src.data_quality.checks import run_data_quality_checks
from src.data_quality.exclusion_analysis import (
    calculate_exclusion_ead_impact,
    generate_exclusion_findings,
    identify_material_exclusions,
    summarize_exclusions_by_category,
    summarize_exclusions_by_materiality,
)
from src.pd_backtesting.aggregation import aggregate_pd_metrics, build_standard_aggregations
from src.pd_backtesting.data_loader import load_observations, load_thresholds
from src.pd_backtesting.discrimination import calculate_discrimination_metrics
from src.pd_backtesting.eligibility import build_population_waterfall, summarize_exclusions, summarize_population_eligibility
from src.pd_backtesting.interpretability import add_interpretability
from src.pd_backtesting.low_default import calculate_multi_year_default_summary, identify_low_default_segments, recommend_ldp_treatment
from src.pd_backtesting.migration import build_rating_migration_matrix, calculate_notch_migration_distribution, identify_material_migration_patterns, summarize_migration
from src.pd_backtesting.metrics import portfolio_metrics
from src.pd_backtesting.pd_adjustments import analyze_moc_impact, analyze_pd_floor_impact, compare_raw_calibrated_regulatory_pd, summarize_pd_components
from src.pd_backtesting.philosophy import (
    analyze_pd_volatility_by_philosophy,
    compare_pd_by_philosophy,
    compare_pit_ttc_behaviour,
    compare_rds_by_philosophy,
    generate_philosophy_commentary,
    summarize_model_philosophy,
)
from src.pd_backtesting.population_shift import diagnose_population_change, summarize_new_and_exited_customers
from src.pd_backtesting.rds import diagnose_rds_change, summarize_rds_stability
from src.pd_backtesting import stat_tests
from src.pd_backtesting.stat_tests import binomial_calibration_test
from src.pd_backtesting.stability import add_observation_year, calculate_psi
from src.pd_backtesting.validation import validate_observation_schema
from src.reporting.demo_narrative import generate_demo_narrative
from src.reporting.data_dictionary import build_data_dictionary
from src.reporting.findings import generate_validation_findings
from src.reporting.test_mapping import build_test_mapping


DATA_PATH = Path("data/generated/pd_observations.csv")
CONFIG_PATH = Path("config/thresholds.yaml")
AURIA_LOGO_PATH = Path("app/assets/auria_logo.png")
AURIA_COLORS = {
    "navy": "#0b2b46",
    "navy_2": "#102f4a",
    "ink": "#061a2d",
    "peach": "#f1a986",
    "peach_2": "#f7c6ae",
    "cream": "#f8f4ef",
    "cream_2": "#fffaf5",
    "grey": "#6d7885",
    "mist": "#d8e0e7",
    "sage": "#7f9c90",
    "rose": "#c96f6f",
}
AURIA_SEQUENCE = [
    AURIA_COLORS["navy"],
    AURIA_COLORS["peach"],
    AURIA_COLORS["sage"],
    AURIA_COLORS["navy_2"],
    AURIA_COLORS["rose"],
    AURIA_COLORS["grey"],
    AURIA_COLORS["peach_2"],
]
AURIA_CONTINUOUS_SCALE = [
    [0.0, "#fffaf5"],
    [0.25, "#f7c6ae"],
    [0.5, "#f1a986"],
    [0.75, "#45647a"],
    [1.0, "#0b2b46"],
]
STATUS_COLORS = {"green": "#6f9d7a", "orange": "#f1a986", "red": "#c96f6f", "grey": "#8a96a3"}
DEMO_DEFAULTS = {
    "scenario_choice": "rds_stable_population",
    "threshold_profile": "standard",
    "retail_n": 30000,
    "corporate_n": 5000,
    "start_year": 2019,
    "years_count": 5,
    "dq_level": "low",
    "random_seed": 42,
}


def configure_plotly_theme() -> None:
    """Configure a global Plotly theme aligned with the Auria visual identity."""
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,250,245,0.72)",
        font={"family": "Inter, Arial, sans-serif", "color": AURIA_COLORS["ink"], "size": 13},
        title={"font": {"color": AURIA_COLORS["ink"], "size": 18}},
        colorway=AURIA_SEQUENCE,
        xaxis={
            "gridcolor": "rgba(11,43,70,0.08)",
            "linecolor": "rgba(11,43,70,0.18)",
            "zerolinecolor": "rgba(11,43,70,0.10)",
            "title": {"font": {"color": AURIA_COLORS["grey"]}},
        },
        yaxis={
            "gridcolor": "rgba(11,43,70,0.08)",
            "linecolor": "rgba(11,43,70,0.18)",
            "zerolinecolor": "rgba(11,43,70,0.10)",
            "title": {"font": {"color": AURIA_COLORS["grey"]}},
        },
        legend={
            "bgcolor": "rgba(255,255,255,0.68)",
            "bordercolor": "rgba(11,43,70,0.10)",
            "borderwidth": 1,
        },
        margin={"l": 36, "r": 24, "t": 56, "b": 36},
    )
    pio.templates["auria"] = template
    pio.templates.default = "auria"
    px.defaults.template = "auria"
    px.defaults.color_discrete_sequence = AURIA_SEQUENCE
    px.defaults.color_continuous_scale = AURIA_CONTINUOUS_SCALE


def inject_auria_theme() -> None:
    """Apply Auria Advisory-inspired visual styling to Streamlit."""
    st.markdown(
        """
        <style>
        :root {
          --auria-navy: #0b2b46;
          --auria-navy-2: #102f4a;
          --auria-ink: #061a2d;
          --auria-peach: #f1a986;
          --auria-peach-2: #f7c6ae;
          --auria-cream: #f8f4ef;
          --auria-cream-2: #fffaf5;
          --auria-grey: #6d7885;
          --auria-line: rgba(11, 43, 70, 0.14);
          --auria-card: rgba(255, 255, 255, 0.86);
          --shadow-soft: 0 24px 60px rgba(11, 43, 70, 0.16);
          --shadow-card: 0 18px 44px rgba(11, 43, 70, 0.10);
        }

        .stApp {
          color: var(--auria-ink);
          background:
            radial-gradient(circle at 10% 2%, rgba(241, 169, 134, 0.24), transparent 28rem),
            radial-gradient(circle at 96% 12%, rgba(11, 43, 70, 0.12), transparent 24rem),
            linear-gradient(180deg, var(--auria-cream-2), var(--auria-cream));
        }

        .stApp::before {
          content: "";
          position: fixed;
          inset: 0;
          pointer-events: none;
          background-image:
            linear-gradient(rgba(11,43,70,0.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(11,43,70,0.035) 1px, transparent 1px);
          background-size: 44px 44px;
          mask-image: linear-gradient(180deg, rgba(0,0,0,0.70), transparent 72%);
          z-index: 0;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.15rem;
          padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
          color: var(--auria-ink);
          letter-spacing: 0;
        }

        h1 {
          font-size: clamp(2rem, 4vw, 3.65rem) !important;
          line-height: 1.02 !important;
          font-weight: 850 !important;
          margin-top: 0.35rem !important;
        }

        h2, h3 {
          font-weight: 800 !important;
        }

        div[data-testid="stToolbar"],
        header[data-testid="stHeader"] {
          background: transparent;
        }

        section[data-testid="stSidebar"] {
          background: rgba(255, 250, 245, 0.88);
          border-right: 1px solid rgba(11, 43, 70, 0.10);
          backdrop-filter: blur(16px);
        }

        div[data-testid="stMetric"] {
          border: 1px solid var(--auria-line);
          background: var(--auria-card);
          border-radius: 22px;
          padding: 16px 18px;
          box-shadow: var(--shadow-card);
          min-height: 118px;
        }

        div[data-testid="stMetricLabel"] p {
          color: var(--auria-grey);
          font-size: 0.78rem;
          font-weight: 800;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }

        div[data-testid="stMetricValue"] {
          color: var(--auria-navy);
          font-weight: 850;
        }

        .stButton > button,
        .stDownloadButton > button,
        button[kind="secondary"],
        button[kind="primary"] {
          border-radius: 999px !important;
          border: 1px solid rgba(11, 43, 70, 0.18) !important;
          background: rgba(255, 255, 255, 0.78) !important;
          color: var(--auria-navy) !important;
          font-weight: 800 !important;
          box-shadow: 0 4px 14px rgba(11, 43, 70, 0.06);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
          border-color: rgba(241, 169, 134, 0.62) !important;
          box-shadow: 0 13px 28px rgba(11, 43, 70, 0.12);
          transform: translateY(-1px);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        textarea,
        input {
          background: rgba(255, 255, 255, 0.78) !important;
          border-color: rgba(11, 43, 70, 0.16) !important;
          color: var(--auria-ink) !important;
          border-radius: 14px !important;
          box-shadow: none !important;
        }

        div[data-baseweb="select"]:focus-within > div,
        div[data-baseweb="input"]:focus-within > div,
        div[data-baseweb="base-input"]:focus-within {
          border-color: rgba(241, 169, 134, 0.82) !important;
          box-shadow: 0 0 0 3px rgba(241, 169, 134, 0.18) !important;
        }

        span[data-baseweb="tag"] {
          background: var(--auria-navy) !important;
          color: white !important;
          border-radius: 999px !important;
          border: 1px solid rgba(11, 43, 70, 0.16) !important;
          font-weight: 800 !important;
        }

        span[data-baseweb="tag"] span,
        span[data-baseweb="tag"] svg {
          color: white !important;
          fill: white !important;
        }

        div[data-baseweb="popover"],
        ul[data-testid="stVirtualDropdown"] {
          background: var(--auria-cream-2) !important;
          border: 1px solid rgba(11,43,70,0.12) !important;
          box-shadow: var(--shadow-card) !important;
        }

        [data-testid="stSlider"] [role="slider"] {
          background: var(--auria-navy) !important;
          border-color: var(--auria-peach) !important;
        }

        [data-testid="stSlider"] div[style*="background"] {
          color: var(--auria-navy) !important;
        }

        div[data-testid="stTabs"] [role="tablist"] {
          gap: 8px;
          border-bottom: 0;
          background: rgba(255,255,255,0.50);
          border: 1px solid var(--auria-line);
          border-radius: 999px;
          padding: 8px;
          box-shadow: 0 8px 24px rgba(11,43,70,0.06);
        }

        div[data-testid="stTabs"] [role="tab"] {
          border-radius: 999px;
          color: var(--auria-navy);
          font-weight: 800;
          padding: 10px 14px;
        }

        div[data-testid="stTabs"] [aria-selected="true"] {
          background: var(--auria-navy);
          color: white;
        }

        div[data-testid="stDataFrame"] {
          border: 1px solid var(--auria-line);
          border-radius: 18px;
          overflow: hidden;
          box-shadow: 0 8px 24px rgba(11,43,70,0.06);
        }

        .js-plotly-plot .plotly .modebar {
          background: rgba(255,250,245,0.78) !important;
          border-radius: 999px;
        }

        .auria-topbar {
          display: flex;
          align-items: center;
          justify-content: flex-start;
          min-height: 58px;
          padding: 0;
          border: 0;
          border-radius: 0;
          background: transparent;
          box-shadow: none;
          margin-bottom: 12px;
        }

        .auria-brand {
          display: flex;
          align-items: center;
          gap: 18px;
          min-width: 260px;
          color: var(--auria-navy);
          font-weight: 900;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          font-size: 0.92rem;
        }

        .auria-brand img {
          display: block;
          width: 210px;
          max-width: 32vw;
          height: auto;
        }

        .auria-nav {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 22px;
          color: var(--auria-navy);
          font-size: 0.82rem;
          font-weight: 800;
          white-space: nowrap;
        }

        .auria-nav a {
          color: var(--auria-navy);
          opacity: 0.82;
          text-decoration: none;
          transition: 0.18s ease;
        }

        .auria-nav a:hover,
        .auria-nav a:focus-visible {
          opacity: 1;
          color: var(--auria-peach);
        }

        .auria-top-actions {
          display: flex;
          align-items: center;
          gap: 10px;
          justify-content: flex-end;
        }

        .auria-top-pill {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border: 1px solid rgba(11, 43, 70, 0.18);
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.76);
          color: var(--auria-navy);
          padding: 11px 16px;
          font-size: 0.82rem;
          font-weight: 900;
          text-decoration: none;
        }

        .auria-top-pill.primary {
          color: white;
          background: var(--auria-navy);
          border-color: var(--auria-navy);
        }

        .auria-hero {
          position: relative;
          overflow: hidden;
          border-radius: 34px;
          padding: 30px 32px;
          margin: 16px 0 24px 0;
          background:
            radial-gradient(circle at 88% 18%, rgba(241, 169, 134, 0.34), transparent 17rem),
            radial-gradient(circle at 18% 82%, rgba(255,255,255,0.14), transparent 20rem),
            linear-gradient(135deg, #071d31, var(--auria-navy));
          color: white;
          box-shadow: var(--shadow-soft);
        }

        .auria-hero-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
          gap: 24px;
          align-items: stretch;
        }

        .auria-kicker {
          color: var(--auria-peach-2);
          font-size: 0.78rem;
          font-weight: 900;
          letter-spacing: 0.10em;
          text-transform: uppercase;
          margin-bottom: 8px;
        }

        .auria-hero h2 {
          color: white;
          font-size: clamp(1.4rem, 2.5vw, 2.35rem);
          line-height: 1.08;
          margin: 0 0 10px 0;
        }

        .auria-hero p {
          color: rgba(255,255,255,0.76);
          margin: 0.35rem 0;
        }

        .auria-hero-card {
          border: 1px solid rgba(255,255,255,0.16);
          background: rgba(255,255,255,0.08);
          border-radius: 24px;
          padding: 18px;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }

        .auria-hero-card .metric-line {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 14px;
          padding: 10px 0;
          border-bottom: 1px solid rgba(255,255,255,0.12);
        }

        .auria-hero-card .metric-line:last-child {
          border-bottom: 0;
        }

        .auria-hero-card .metric-label {
          color: rgba(255,255,255,0.66);
          font-size: 0.80rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .auria-hero-card .metric-value {
          color: white;
          font-size: 1rem;
          font-weight: 900;
          text-align: right;
        }

        .auria-hero h1 {
          color: white !important;
          font-size: clamp(2.15rem, 4.2vw, 4.35rem) !important;
          line-height: 1.01 !important;
          margin: 0 0 12px 0 !important;
          max-width: 980px;
        }

        .auria-hero .lead {
          max-width: 860px;
          color: rgba(255,255,255,0.82);
          font-size: 1.02rem;
          line-height: 1.65;
        }

        .auria-run {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          margin-top: 16px;
          padding: 9px 13px;
          border: 1px solid rgba(255,255,255,0.22);
          border-radius: 999px;
          background: rgba(255,255,255,0.08);
          color: rgba(255,255,255,0.92);
          font-size: 0.82rem;
          font-weight: 850;
        }

        .auria-section-header {
          border: 1px solid var(--auria-line);
          background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,250,245,0.74));
          border-radius: 24px;
          padding: 18px 20px;
          margin: 8px 0 18px 0;
          box-shadow: var(--shadow-card);
        }

        .auria-section-header h2,
        .auria-section-header h3 {
          margin: 0 0 6px 0 !important;
          color: var(--auria-navy) !important;
        }

        .auria-section-header p {
          margin: 0;
          color: var(--auria-grey);
          line-height: 1.55;
        }

        .auria-kpi-card {
          min-height: 128px;
          border: 1px solid var(--auria-line);
          border-radius: 20px;
          background: rgba(255,255,255,0.88);
          box-shadow: var(--shadow-card);
          padding: 18px 18px 14px;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
        }

        .auria-kpi-label {
          color: var(--auria-grey);
          font-size: 0.76rem;
          font-weight: 900;
          letter-spacing: 0.06em;
          text-transform: uppercase;
        }

        .auria-kpi-value {
          color: var(--auria-navy);
          font-size: clamp(1.45rem, 2.1vw, 2.18rem);
          line-height: 1.04;
          font-weight: 900;
          overflow-wrap: anywhere;
          margin-top: 12px;
        }

        .auria-kpi-caption {
          color: var(--auria-grey);
          font-size: 0.82rem;
          line-height: 1.35;
          margin-top: 10px;
        }

        .auria-status-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
          gap: 12px;
          margin: 14px 0 20px;
        }

        .auria-status-card {
          border: 1px solid var(--auria-line);
          border-radius: 20px;
          background: rgba(255,255,255,0.86);
          box-shadow: var(--shadow-card);
          padding: 14px 14px;
          display: grid;
          grid-template-columns: 12px minmax(0, 1fr) auto;
          gap: 12px;
          align-items: center;
        }

        .auria-status-dot {
          width: 11px;
          height: 44px;
          border-radius: 999px;
        }

        .auria-status-title {
          color: var(--auria-navy);
          font-size: 0.95rem;
          font-weight: 900;
          margin-bottom: 3px;
        }

        .auria-status-message {
          color: var(--auria-grey);
          font-size: 0.80rem;
          line-height: 1.35;
        }

        .auria-status-pill {
          color: white;
          border-radius: 999px;
          padding: 6px 9px;
          font-size: 0.72rem;
          font-weight: 900;
          text-transform: uppercase;
        }

        div[data-testid="stPlotlyChart"] {
          border: 1px solid var(--auria-line);
          border-radius: 20px;
          background: rgba(255,255,255,0.82);
          box-shadow: var(--shadow-card);
          padding: 12px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
          border: 1px solid var(--auria-line) !important;
          border-radius: 26px !important;
          background: rgba(255,255,255,0.78) !important;
          box-shadow: var(--shadow-card) !important;
          padding: 8px !important;
        }

        .auria-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 14px;
        }

        .auria-chip {
          display: inline-flex;
          border: 1px solid rgba(255,255,255,0.18);
          background: rgba(255,255,255,0.10);
          color: white;
          border-radius: 999px;
          padding: 7px 11px;
          font-size: 0.78rem;
          font-weight: 800;
        }

        .auria-filter-panel {
          border: 1px solid var(--auria-line);
          background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(255,250,245,0.78));
          border-radius: 26px;
          padding: 20px 22px;
          margin: 14px 0 10px 0;
          box-shadow: var(--shadow-card);
        }

        .auria-filter-shell {
          border: 1px solid rgba(11, 43, 70, 0.18);
          background: rgba(255, 255, 255, 0.64);
          border-radius: 28px;
          padding: 16px 18px 18px 18px;
          margin: 0 0 24px 0;
          box-shadow: 0 16px 42px rgba(11,43,70,0.08);
        }

        .auria-context-panel {
          border: 1px solid rgba(11,43,70,0.12);
          background: rgba(255,255,255,0.72);
          border-radius: 24px;
          padding: 18px 20px;
          margin: 0 0 18px 0;
          box-shadow: 0 10px 28px rgba(11,43,70,0.07);
        }

        .auria-context-panel p {
          margin: 0.35rem 0;
          color: var(--auria-grey);
        }

        .auria-filter-title {
          color: var(--auria-navy);
          font-weight: 900;
          letter-spacing: 0.03em;
          text-transform: uppercase;
          font-size: 0.82rem;
          margin-bottom: 2px;
        }

        .auria-muted {
          color: var(--auria-grey);
          font-size: 0.90rem;
          margin-bottom: 12px;
        }

        .auria-contact-panel {
          margin: 32px 0 22px;
          position: relative;
          overflow: hidden;
          border-radius: 42px;
          background:
            radial-gradient(circle at 88% 18%, rgba(241, 169, 134, 0.34), transparent 15rem),
            linear-gradient(135deg, #071d31, var(--auria-navy));
          color: white;
          box-shadow: var(--shadow-soft);
        }

        .auria-contact-panel::before {
          content: "";
          position: absolute;
          width: 280px;
          height: 280px;
          border: 1.5px solid rgba(241, 169, 134, 0.45);
          border-radius: 50%;
          right: -85px;
          top: -95px;
        }

        .auria-contact-grid {
          position: relative;
          z-index: 1;
          display: grid;
          grid-template-columns: 1.15fr 0.85fr;
          gap: 34px;
          align-items: center;
          padding: clamp(28px, 5vw, 52px);
        }

        .auria-contact-panel h2 {
          margin: 0;
          color: white;
          font-family: Georgia, "Times New Roman", serif;
          font-size: clamp(34px, 5vw, 62px);
          line-height: 0.96;
          letter-spacing: 0;
        }

        .auria-contact-panel .lead {
          margin: 18px 0 0;
          max-width: 680px;
          color: rgba(255, 255, 255, 0.76);
          font-size: 17px;
          line-height: 1.7;
        }

        .auria-contact-links {
          border: 1px solid rgba(255,255,255,0.15);
          background: rgba(255,255,255,0.08);
          backdrop-filter: blur(12px);
          border-radius: 30px;
          padding: 22px;
          display: grid;
          gap: 12px;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.14);
        }

        .auria-contact-links a {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 14px;
          min-height: 42px;
          border-radius: 999px;
          padding: 8px 14px;
          background: rgba(255, 255, 255, 0.94);
          color: var(--auria-navy);
          text-decoration: none;
          font-weight: 900;
        }

        .auria-contact-logo {
          display: inline-grid;
          place-items: center;
          width: 28px;
          height: 28px;
          border-radius: 50%;
          background: var(--auria-navy);
          color: white;
          flex: 0 0 auto;
        }

        .auria-contact-logo svg {
          width: 16px;
          height: 16px;
          fill: currentColor;
        }

        .auria-contact-link-main {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .auria-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 14px;
          flex-wrap: wrap;
          margin: 18px 0 0;
          padding: 18px 0 6px;
          color: var(--auria-grey);
          font-size: 0.88rem;
        }

        .auria-footer strong {
          color: var(--auria-navy);
        }

        @media (max-width: 900px) {
          .auria-hero-grid { grid-template-columns: 1fr; }
          .auria-topbar { align-items: flex-start; border-radius: 24px; flex-direction: column; padding: 16px; }
          .auria-nav, .auria-top-actions { align-items: flex-start; flex-direction: column; gap: 10px; }
          .auria-contact-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def auria_logo_data_uri() -> str:
    """Return the local Auria logo as a data URI for HTML rendering."""
    if not AURIA_LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(AURIA_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_auria_topbar() -> None:
    logo = auria_logo_data_uri()
    logo_html = f'<img src="{logo}" alt="Auria Advisory" />' if logo else "<strong>Auria Advisory</strong>"
    st.markdown(
        f"""
        <div class="auria-topbar">
          <a class="auria-brand" href="#" aria-label="Auria Advisory, retour au haut de page">
            {logo_html}
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auria_contact_footer() -> None:
    """Render the RegWatch-inspired Auria contact block at the bottom of the app."""
    st.markdown(
        """
        <section class="auria-contact-panel" id="contact" aria-labelledby="contactTitle">
          <div class="auria-contact-grid">
            <div>
              <p class="auria-kicker">Auria Advisory</p>
              <h2 id="contactTitle">Nous contacter</h2>
              <p class="lead">
                Pour structurer une trajectoire de validation réglementaire PD, échanger sur vos enjeux
                de gouvernance modèle, de backtesting ou de mise en conformité, parlons de votre contexte.
              </p>
            </div>
            <div class="auria-contact-links" aria-label="Liens de contact Auria Advisory">
              <a href="https://auria-advisory.fr/" target="_blank" rel="noopener" aria-label="Ouvrir le site internet Auria Advisory dans un nouvel onglet">
                <span class="auria-contact-link-main">
                  <span class="auria-contact-logo" aria-hidden="true">
                    <svg viewBox="0 0 24 24" focusable="false">
                      <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm6.93 9h-3.18a15.4 15.4 0 0 0-1.2-5.06A8.04 8.04 0 0 1 18.93 11ZM12 4.04c.83 1.2 1.5 3.28 1.7 6.96h-3.4c.2-3.68.87-5.76 1.7-6.96ZM4.26 13H8.3c.16 2.1.55 3.9 1.15 5.06A8.04 8.04 0 0 1 4.26 13Zm4.04-2H5.07a8.04 8.04 0 0 1 4.38-5.06A15.4 15.4 0 0 0 8.3 11Zm3.7 8.96c-.83-1.2-1.5-3.28-1.7-6.96h3.4c-.2 3.68-.87 5.76-1.7 6.96Zm2.55-1.9c.6-1.16.99-2.96 1.15-5.06h4.04a8.04 8.04 0 0 1-5.19 5.06Z"/>
                    </svg>
                  </span>
                  <span>Site internet</span>
                </span>
                <span>auria-advisory.fr</span>
              </a>
              <a href="https://www.linkedin.com/company/auria-advisory/" target="_blank" rel="noopener" aria-label="Ouvrir la page LinkedIn Auria Advisory dans un nouvel onglet">
                <span class="auria-contact-link-main">
                  <span class="auria-contact-logo" aria-hidden="true">
                    <svg viewBox="0 0 24 24" focusable="false">
                      <path d="M20.45 20.45h-3.56v-5.58c0-1.33-.03-3.04-1.85-3.04-1.85 0-2.13 1.44-2.13 2.94v5.68H9.35V9h3.42v1.56h.05a3.75 3.75 0 0 1 3.37-1.85c3.61 0 4.27 2.37 4.27 5.46v6.28ZM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12Zm1.78 13.02H3.56V9h3.56v11.45Z"/>
                    </svg>
                  </span>
                  <span>LinkedIn</span>
                </span>
                <span>Suivre nos actualités</span>
              </a>
            </div>
          </div>
        </section>
        <footer class="auria-footer" role="contentinfo">
          <span><strong>Auria Advisory</strong> · Plateforme de démonstration — validation réglementaire PD, systèmes de notation et supervision prudentielle.</span>
          <span>Version démonstrateur · Mai 2026</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def cached_load_observations(path: str, file_mtime: float) -> pd.DataFrame:
    return load_observations(path)


@st.cache_data
def cached_generate_scenario(
    scenario: str,
    retail: int,
    corporate: int,
    start_year: int,
    years: int,
    dq_level: str,
    seed: int,
) -> pd.DataFrame:
    return generate_demo_scenario(scenario, retail, corporate, start_year, years, dq_level, seed)


def format_percent(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def format_number(value: float) -> str:
    return "" if pd.isna(value) else f"{value:,.2f}".replace(",", " ")


def render_kpi_card(label: str, value: str, caption: str = "") -> None:
    """Render a stable Auria-style KPI card for the executive dashboard."""
    st.markdown(
        f"""
        <div class="auria-kpi-card">
          <div class="auria-kpi-label">{html.escape(str(label))}</div>
          <div class="auria-kpi-value">{html.escape(str(value))}</div>
          <div class="auria-kpi-caption">{html.escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, body: str, kicker: str | None = None) -> None:
    """Render a compact explanatory section header."""
    kicker_html = f'<div class="auria-kicker" style="color:#f1a986">{html.escape(kicker)}</div>' if kicker else ""
    st.markdown(
        f"""
        <div class="auria-section-header">
          {kicker_html}
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_rates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in [
        "pd_mean",
        "ead_weighted_pd",
        "odr",
        "calibration_gap",
        "p_value",
        "p_value_two_sided",
        "p_value_one_sided_high",
        "ci_lower",
        "ci_upper",
        "auc",
        "gini",
        "ks",
        "psi",
        "bad_rate",
    ]:
        if column in output.columns:
            output[column] = output[column].map(format_percent)
    for column in ["expected_defaults", "ead_total", "calibration_ratio"]:
        if column in output.columns:
            output[column] = output[column].map(format_number)
    return output


def thresholds_limitations_text(profile: str) -> str:
    return (
        f"Profil actif : {profile}. Les seuils du démonstrateur sont paramétrables dans "
        "`config/thresholds.yaml`. Ils servent à illustrer une démarche de validation et ne "
        "constituent pas des seuils réglementaires universels."
    )


def methodology_one_liners() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("ODR", "Le taux de défaut observé mesure la fréquence réelle des défauts sur le périmètre."),
            ("PD moyenne", "La PD moyenne résume le niveau de risque estimé par le modèle."),
            ("Défauts attendus", "La somme des PD donne le nombre de défauts attendu sous le modèle."),
            ("Calibration gap", "L'écart ODR - PD moyenne indique une sur- ou sous-estimation du niveau de risque."),
            ("Calibration ratio", "Le ratio ODR / PD moyenne mesure l'écart relatif entre observé et estimé."),
            ("Test binomial", "Le test binomial indique si l'écart de défauts est statistiquement significatif."),
            ("AUC", "L'AUC mesure la capacité du modèle à classer les défauts avant les non-défauts."),
            ("Gini", "Le Gini est une transformation de l'AUC : 2 x AUC - 1."),
            ("KS", "Le KS mesure la séparation maximale entre défauts et non-défauts."),
            ("PSI", "Le PSI mesure la dérive de distribution entre deux périodes."),
            ("Monotonie des grades", "La monotonie vérifie que le taux de défaut augmente avec le risque du grade."),
            ("Traffic light", "Les statuts vert, orange, rouge et gris priorisent les signaux de validation."),
        ],
        columns=["indicateur", "explication_simple"],
    )


def regulatory_anchors() -> list[dict]:
    """Return key regulatory references for PD backtesting demonstrations."""
    return [
        {
            "organisme": "Union europeenne",
            "titre": "Reglement (UE) n deg 575/2013, CRR - Article 185, validation des estimations internes",
            "date": "26 juin 2013",
            "lien": "https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32013R0575",
            "utilite": (
                "Socle reglementaire IRB. L'article 185 impose notamment la validation reguliere "
                "des estimations internes et la comparaison des taux de defaut realises avec les PD estimees."
            ),
        },
        {
            "organisme": "EBA",
            "titre": "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures - EBA/GL/2017/16",
            "date": "20 novembre 2017",
            "lien": "https://www.eba.europa.eu/publications-and-media/press-releases/eba-publishes-final-guidelines-estimation-risk-parameters",
            "utilite": (
                "Reference centrale sur l'estimation et la calibration des parametres PD sous approche IRB. "
                "Elle couvre les principes de calibration, les revues regulieres et l'usage des estimations."
            ),
        },
        {
            "organisme": "EBA",
            "titre": "Guidelines on the application of the definition of default - EBA/GL/2016/07",
            "date": "28 septembre 2016",
            "lien": "https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-application-definition?version=2016",
            "utilite": (
                "Cadre de reference pour le defaut reglementaire : jours de retard, unlikeliness to pay, "
                "retour au statut non defaute, application groupe et specificites Retail."
            ),
        },
        {
            "organisme": "EBA",
            "titre": "Supervisory handbook on the validation of IRB rating systems - EBA/REP/2023/29",
            "date": "7 aout 2023, document mis a jour le 10 aout 2023",
            "lien": "https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/supervisory-handbook-validation",
            "utilite": (
                "Precise les attentes de supervision sur la fonction de validation, la performance modele, "
                "la quantification du risque, la discrimination, la gouvernance et la documentation."
            ),
        },
        {
            "organisme": "BCE",
            "titre": "ECB Guide to internal models, revised guide",
            "date": "28 juillet 2025",
            "lien": "https://www.bankingsupervision.europa.eu/press/pr/date/2025/html/ssm.pr250728~2b36305822.en.html",
            "utilite": (
                "Exprime les attentes de la BCE pour les modeles internes des etablissements supervises, "
                "dont les modeles de risque de credit, la validation interne, l'audit interne et la quantification PD/LGD."
            ),
        },
    ]


def render_methodology_text() -> None:
    """Render concise methodology definitions as narrative text blocks."""
    for _, row in methodology_one_liners().iterrows():
        st.markdown(f"**{row['indicateur']}**  \n{row['explication_simple']}")


def render_regulatory_anchors() -> None:
    """Render regulatory anchors as readable source blocks."""
    st.subheader("Ancrages reglementaires")
    st.markdown(
        "Ces references positionnent le demonstrateur dans le cadre europeen IRB. "
        "Elles ne remplacent pas une analyse normative complete, mais donnent les points d'ancrage utiles "
        "pour expliquer le backtesting PD en rendez-vous."
    )
    for anchor in regulatory_anchors():
        st.markdown(
            f"""
            **{anchor["organisme"]} - {anchor["titre"]}**  
            Date de publication : {anchor["date"]}  
            Source : [{anchor["lien"]}]({anchor["lien"]})  
            Utilite pour le demonstrateur : {anchor["utilite"]}
            """
        )


def retail_corporate_comparison(
    summary: pd.DataFrame,
    calibration_alerts: pd.DataFrame,
    discrimination_alerts: pd.DataFrame,
    stability_alerts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for portfolio in ["Retail", "Corporate"]:
        row = summary[summary["portfolio"] == portfolio]
        cal = calibration_alerts[(calibration_alerts.get("aggregation_level") == "portfolio") & (calibration_alerts.get("perimeter") == portfolio)]
        disc = discrimination_alerts[(discrimination_alerts.get("aggregation_level") == "portfolio") & (discrimination_alerts.get("perimeter") == portfolio)]
        stab = stability_alerts[(stability_alerts.get("aggregation_level") == "portfolio") & (stability_alerts.get("perimeter") == portfolio)] if not stability_alerts.empty else pd.DataFrame()
        observations = int(row["observations"].sum()) if not row.empty else 0
        defaults = int(row["defaults"].sum()) if not row.empty else 0
        robustness = "Interprétable" if observations >= 30 and defaults >= 5 else "Défauts insuffisants"
        statuses = [
            cal["status"].iloc[0] if not cal.empty else "grey",
            disc["status"].iloc[0] if not disc.empty else "grey",
            stab["status"].iloc[0] if not stab.empty else "grey",
        ]
        rows.append(
            {
                "portfolio": portfolio,
                "observations": observations,
                "defaults": defaults,
                "robustesse_statistique": robustness,
                "statuts_principaux": ", ".join(statuses),
                "limites_interpretation": "Faibles défauts possibles" if portfolio == "Corporate" else "Volumes généralement robustes",
                "recommandations": "Documenter les faibles volumes" if robustness != "Interprétable" else "Analyser les alertes par grade et millésime",
            }
        )
    return pd.DataFrame(rows)


def build_model_summary(
    observations: pd.DataFrame,
    calibration_alerts: pd.DataFrame,
    discrimination_alerts: pd.DataFrame,
    stability_alerts: pd.DataFrame,
    current_period,
) -> pd.DataFrame:
    current = add_observation_year(observations)
    current = current[current["observation_year"] == current_period]
    rows = []
    has_calibration = {"aggregation_level", "perimeter", "status"}.issubset(calibration_alerts.columns)
    has_discrimination = {"aggregation_level", "perimeter", "status"}.issubset(discrimination_alerts.columns)
    has_stability = {"aggregation_level", "perimeter", "status"}.issubset(stability_alerts.columns)
    for (model_id, portfolio), group in current.groupby(["model_id", "portfolio"], dropna=False):
        metrics = portfolio_metrics(group)
        calibration = (
            calibration_alerts[(calibration_alerts["aggregation_level"] == "portfolio") & (calibration_alerts["perimeter"] == portfolio)]
            if has_calibration
            else pd.DataFrame()
        )
        discrimination = (
            discrimination_alerts[(discrimination_alerts["aggregation_level"] == "portfolio") & (discrimination_alerts["perimeter"] == portfolio)]
            if has_discrimination
            else pd.DataFrame()
        )
        stability = (
            stability_alerts[(stability_alerts["aggregation_level"] == "portfolio") & (stability_alerts["perimeter"] == portfolio)]
            if has_stability
            else pd.DataFrame()
        )
        calibration_status = calibration["status"].iloc[0] if not calibration.empty else "grey"
        discrimination_status = discrimination["status"].iloc[0] if not discrimination.empty else "grey"
        stability_status = stability["status"].iloc[0] if not stability.empty else "grey"
        rows.append(
            {
                "model_id": model_id,
                "portfolio": portfolio,
                "periode_analysee": current_period,
                "observations": metrics["observations"],
                "defaults": metrics["observed_defaults"],
                "odr": metrics["odr"],
                "pd_mean": metrics["pd_mean"],
                "calibration_status": calibration_status,
                "auc": discrimination["auc"].iloc[0] if not discrimination.empty else pd.NA,
                "gini": discrimination["gini"].iloc[0] if not discrimination.empty else pd.NA,
                "ks": discrimination["ks"].iloc[0] if not discrimination.empty else pd.NA,
                "discrimination_status": discrimination_status,
                "psi": stability["psi"].iloc[0] if not stability.empty else pd.NA,
                "stability_status": stability_status,
                "statut_global": combine_statuses([calibration_status, discrimination_status, stability_status]),
            }
        )
    return pd.DataFrame(rows)


def status_counts(*frames: pd.DataFrame) -> dict[str, int]:
    statuses = pd.concat([frame["status"] for frame in frames if not frame.empty and "status" in frame.columns], ignore_index=True)
    return {
        "red": int((statuses == "red").sum()),
        "orange": int((statuses == "orange").sum()),
        "grey": int((statuses == "grey").sum()),
    }


def status_distribution(*frames: pd.DataFrame) -> pd.DataFrame:
    """Return a complete status distribution for visual executive charts."""
    statuses = pd.concat([frame["status"] for frame in frames if not frame.empty and "status" in frame.columns], ignore_index=True)
    if statuses.empty:
        statuses = pd.Series(["grey"])
    counts = statuses.value_counts().reindex(["green", "orange", "red", "grey"], fill_value=0).reset_index()
    counts.columns = ["status", "count"]
    return counts


def data_quality_executive_status(dq_results: pd.DataFrame, thresholds: dict) -> str:
    """Summarise data quality checks into one executive traffic-light status."""
    if dq_results.empty or "failure_rate" not in dq_results.columns:
        return "grey"
    max_failure_rate = float(dq_results["failure_rate"].fillna(0).max())
    if max_failure_rate == 0:
        return "green"
    dq_threshold = thresholds.get("data_quality", {}).get("max_error_rate", 0.01)
    return traffic_from_rate(max_failure_rate, dq_threshold)


def build_executive_theme_status(
    dq_results: pd.DataFrame,
    calibration_alerts: pd.DataFrame,
    discrimination_alerts: pd.DataFrame,
    stability_alerts: pd.DataFrame,
    monotonicity_alerts: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:
    """Build one status row per main analysis family."""
    rows = [
        {
            "theme": "Data quality",
            "status": data_quality_executive_status(dq_results, thresholds),
            "message": "Complétude, unicité et cohérence des champs clés.",
        },
        {
            "theme": "Calibration PD",
            "status": first_status(calibration_alerts),
            "message": "Comparaison entre PD estimée et défauts observés.",
        },
        {
            "theme": "Discrimination",
            "status": first_status(discrimination_alerts),
            "message": "Pouvoir de séparation défauts / non-défauts.",
        },
        {
            "theme": "Stabilité RDS",
            "status": first_status(stability_alerts),
            "message": "Stabilité des distributions entre périodes.",
        },
        {
            "theme": "Ordonnancement",
            "status": first_status(monotonicity_alerts),
            "message": "Monotonie des défauts observés par grade.",
        },
    ]
    return pd.DataFrame(rows)


def build_portfolio_executive_view(observations: pd.DataFrame) -> pd.DataFrame:
    """Compute compact portfolio-level PD/ODR values for the executive dashboard."""
    rows = []
    if "portfolio" not in observations.columns:
        return pd.DataFrame(rows)
    for portfolio, group in observations.groupby("portfolio", dropna=False):
        metrics = portfolio_metrics(group)
        rows.append(
            {
                "portfolio": portfolio,
                "observations": metrics["observations"],
                "defaults": metrics["observed_defaults"],
                "PD moyenne": metrics["pd_mean"],
                "ODR": metrics["odr"],
                "Défauts attendus": metrics["expected_defaults"],
                "Défauts observés": metrics["observed_defaults"],
            }
        )
    return pd.DataFrame(rows)


def render_status_strip(theme_status: pd.DataFrame) -> None:
    """Render a compact visual strip of traffic-light statuses."""
    cards = []
    for _, row in theme_status.iterrows():
        status = str(row["status"])
        color = STATUS_COLORS.get(status, STATUS_COLORS["grey"])
        cards.append(
            f"""
            <div class="auria-status-card">
              <div class="auria-status-dot" style="background:{color}"></div>
              <div>
                <div class="auria-status-title">{html.escape(str(row["theme"]))}</div>
                <div class="auria-status-message">{html.escape(str(row["message"]))}</div>
              </div>
              <div class="auria-status-pill" style="background:{color}">{html.escape(status)}</div>
            </div>
            """
        )
    st.markdown(f'<div class="auria-status-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def filter_dataframe(frame: pd.DataFrame, portfolios, segments, model_ids, years) -> pd.DataFrame:
    filtered = frame.copy()
    if portfolios:
        filtered = filtered[filtered["portfolio"].isin(portfolios)]
    if segments:
        filtered = filtered[filtered["segment"].isin(segments)]
    if model_ids:
        filtered = filtered[filtered["model_id"].isin(model_ids)]
    if years:
        filtered = filtered[filtered["observation_year"].isin(years)]
    return filtered


def ensure_pd_12m_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Backfill 12-month PD metadata for files generated before PIT/TTC enrichment."""
    output = frame.copy()
    if "pd_type" not in output.columns:
        output["pd_type"] = "regulatory_12m"
    if "pd_horizon_months" not in output.columns:
        output["pd_horizon_months"] = 12
    return output


def init_demo_state(profile_names: list[str]) -> None:
    """Initialise demo parameters in Streamlit session state."""
    for key, value in DEMO_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    if st.session_state["threshold_profile"] not in profile_names:
        st.session_state["threshold_profile"] = profile_names[0]


def demo_settings_body(catalog: dict, profile_names: list[str]) -> None:
    """Render demo settings controls in a dialog or fallback container."""
    with st.form("demo_settings_form"):
        scenario_keys = ["current_file"] + list(catalog)
        scenario_choice = st.selectbox(
            "Scenario",
            scenario_keys,
            index=scenario_keys.index(st.session_state["scenario_choice"])
            if st.session_state["scenario_choice"] in scenario_keys
            else 0,
            format_func=lambda key: "Fichier courant" if key == "current_file" else catalog[key].label,
        )
        threshold_profile = st.selectbox(
            "Profil de seuils",
            profile_names,
            index=profile_names.index(st.session_state["threshold_profile"])
            if st.session_state["threshold_profile"] in profile_names
            else 0,
        )
        cols = st.columns(2)
        retail_n = cols[0].number_input("Observations Retail", min_value=1000, value=int(st.session_state["retail_n"]), step=1000)
        corporate_n = cols[1].number_input("Observations Corporate", min_value=500, value=int(st.session_state["corporate_n"]), step=500)
        cols = st.columns(2)
        start_year = cols[0].number_input("Annee de debut", min_value=2000, max_value=2030, value=int(st.session_state["start_year"]), step=1)
        years_count = cols[1].number_input("Nombre d'annees", min_value=2, max_value=10, value=int(st.session_state["years_count"]), step=1)
        dq_level = st.selectbox(
            "Niveau anomalies DQ",
            ["none", "low", "medium", "high"],
            index=["none", "low", "medium", "high"].index(st.session_state["dq_level"]),
        )
        random_seed = st.number_input("Random seed", min_value=1, value=int(st.session_state["random_seed"]), step=1)
        submitted = st.form_submit_button("Appliquer les parametres")

    if submitted:
        st.session_state["scenario_choice"] = scenario_choice
        st.session_state["threshold_profile"] = threshold_profile
        st.session_state["retail_n"] = int(retail_n)
        st.session_state["corporate_n"] = int(corporate_n)
        st.session_state["start_year"] = int(start_year)
        st.session_state["years_count"] = int(years_count)
        st.session_state["dq_level"] = dq_level
        st.session_state["random_seed"] = int(random_seed)
        st.rerun()


if hasattr(st, "dialog"):
    @st.dialog("Parametres demo")
    def demo_settings_dialog(catalog: dict, profile_names: list[str]) -> None:
        demo_settings_body(catalog, profile_names)
else:
    def demo_settings_dialog(catalog: dict, profile_names: list[str]) -> None:
        with st.expander("Parametres demo", expanded=True):
            demo_settings_body(catalog, profile_names)


def render_scope_presentation(
    observations: pd.DataFrame,
    scenario_label: str,
    scenario_description: str,
    expected_observation: str,
    threshold_profile: str,
) -> None:
    """Display the analysed perimeter as the main Auria-style hero."""
    years = sorted(observations["observation_year"].dropna().unique().tolist())
    portfolios = sorted(observations["portfolio"].dropna().unique().tolist())
    models = sorted(observations["model_id"].dropna().unique().tolist())
    segments = sorted(observations["segment"].dropna().unique().tolist())
    period_label = f"{years[0]}-{years[-1]}" if years else "Non disponible"
    default_count = int(observations["default_flag_12m"].fillna(0).sum()) if "default_flag_12m" in observations else 0
    obs_label = f"{len(observations):,}".replace(",", " ")
    default_label = f"{default_count:,}".replace(",", " ")
    portfolio_label = ", ".join(map(str, portfolios)) if portfolios else "NA"
    model_label = ", ".join(map(str, models[:3])) if models else "NA"
    if len(models) > 3:
        model_label += f" +{len(models) - 3}"
    segment_label = f"{len(segments)} segment(s)" if segments else "NA"
    st.markdown(
        f"""
        <div class="auria-hero">
          <div class="auria-hero-grid">
            <div>
              <div class="auria-kicker">Auria Advisory | Backtesting & Rating Systems</div>
              <h1>Validation réglementaire PD 12 mois</h1>
              <p class="lead">
                Démonstrateur de validation et monitoring des systèmes de notation crédit :
                qualité des données, population éligible, stabilité RDS, calibration, discrimination,
                migrations, low-default portfolios et lecture PIT / TTC / Hybrid.
              </p>
              <div class="auria-run">Scénario actif : {html.escape(str(scenario_label))}</div>
              <p><b style="color:#f7c6ae">Contexte :</b> {html.escape(str(scenario_description))}</p>
              <p><b style="color:#f7c6ae">À observer :</b> {html.escape(str(expected_observation))}</p>
              <div class="auria-chips">
                <span class="auria-chip">Période {html.escape(str(period_label))}</span>
                <span class="auria-chip">Profil seuils {html.escape(str(threshold_profile))}</span>
                <span class="auria-chip">{obs_label} observations</span>
                <span class="auria-chip">{default_label} défauts</span>
              </div>
            </div>
            <div class="auria-hero-card">
              <div class="auria-kicker">Périmètre de démonstration</div>
              <div class="metric-line">
                <span class="metric-label">Portefeuilles</span>
                <span class="metric-value">{html.escape(portfolio_label)}</span>
              </div>
              <div class="metric-line">
                <span class="metric-label">Modèles</span>
                <span class="metric-value">{html.escape(model_label)}</span>
              </div>
              <div class="metric-line">
                <span class="metric-label">Segmentation</span>
                <span class="metric-value">{html.escape(segment_label)}</span>
              </div>
              <div class="metric-line">
                <span class="metric-label">Usage</span>
                <span class="metric-value">RDV client / validation modèle</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app_context_intro() -> None:
    """Render a business-oriented introduction for the enriched PD demonstrator."""
    st.markdown(
        """
        <div class="auria-context-panel">
          <p><b>Démonstrateur de validation et monitoring des systèmes de notation PD</b></p>
          <p>
            L'application couvre le backtesting réglementaire PD 12 mois et l'analyse des systèmes de notation :
            qualité des données, population réglementaire éligible, calibration, discrimination, stabilité RDS,
            migrations de ratings, low-default portfolios, exclusions, floors, marges de conservatisme et lecture PIT / TTC.
          </p>
          <p>
            Les résultats sont calculés sur données fictives Retail et Corporate, avec des scénarios conçus pour illustrer
            les principaux constats de validation modèle et les limites d'interprétation liées aux volumes ou aux changements de population.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_filters(
    observations: pd.DataFrame,
) -> tuple[list, list, list, list, str]:
    """Render the analysis filters in the page body."""
    year_values = sorted(observations["observation_year"].dropna().unique().tolist())
    portfolio_options = sorted(observations["portfolio"].dropna().unique())
    segment_options = sorted(observations["segment"].dropna().unique())
    model_options = sorted(observations["model_id"].dropna().unique())
    st.session_state["filter_portfolios"] = [
        value for value in st.session_state.get("filter_portfolios", []) if value in portfolio_options
    ]
    st.session_state["filter_segments"] = [
        value for value in st.session_state.get("filter_segments", []) if value in segment_options
    ]
    st.session_state["filter_model_ids"] = [
        value for value in st.session_state.get("filter_model_ids", []) if value in model_options
    ]
    st.session_state["filter_years"] = [
        value for value in st.session_state.get("filter_years", year_values) if value in year_values
    ] or year_values

    with st.container(border=True):
        st.markdown("### Filtres d'analyse")
        st.caption(
            "Ces filtres délimitent le périmètre analysé dans tous les onglets : population, qualité, stabilité, "
            "calibration, discrimination, alertes et exports."
        )
        row1 = st.columns([1.2, 1.2, 1.1])
        portfolios = row1[0].multiselect("Portfolio", portfolio_options, key="filter_portfolios")
        segments = row1[1].multiselect("Segment", segment_options, key="filter_segments")
        model_ids = row1[2].multiselect("Model ID", model_options, key="filter_model_ids")
        row2 = st.columns([1.4, 1])
        years = row2[0].multiselect("Période", year_values, key="filter_years")
        aggregation_level = row2[1].selectbox(
            "Niveau d'agrégation",
            ["portfolio", "segment", "rating_grade", "portfolio_segment", "portfolio_rating_grade"],
            key="filter_aggregation_level",
        )
        if st.button("Réinitialiser les filtres", type="secondary"):
            st.session_state["filter_portfolios"] = []
            st.session_state["filter_segments"] = []
            st.session_state["filter_model_ids"] = []
            st.session_state["filter_years"] = year_values
            st.session_state["filter_aggregation_level"] = "portfolio"
            st.rerun()
    return portfolios, segments, model_ids, years, aggregation_level


def traffic_from_rate(rate: float, threshold: float) -> str:
    """Return a simple traffic-light status from an anomaly rate."""
    if pd.isna(rate):
        return "grey"
    if rate <= threshold:
        return "green"
    if rate <= threshold * 2:
        return "orange"
    return "red"


def compliance_pie(labels: list[str], values: list[int], title: str) -> go.Figure:
    """Build an Auria-colored compliance pie chart."""
    return px.pie(
        names=labels,
        values=values,
        title=title,
        color=labels,
        color_discrete_map={
            "Conforme": AURIA_COLORS["navy"],
            "Non conforme": AURIA_COLORS["peach"],
            "OK": AURIA_COLORS["navy"],
            "KO": AURIA_COLORS["peach"],
        },
        hole=0.48,
    )


def variable_kind(frame: pd.DataFrame, column: str) -> str:
    """Classify a variable as numeric-continuous or discrete/categorical for charts."""
    if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].dropna().nunique() > 12:
        return "continuous"
    return "discrete"


def chi2_categorical_tests(frame: pd.DataFrame, target: str = "default_flag_12m") -> pd.DataFrame:
    """Run chi-square independence tests between categorical variables and the default flag."""
    candidates = [
        column
        for column in ["portfolio", "segment", "product_type", "model_id", "model_version", "rating_grade", "exclusion_flag"]
        if column in frame.columns and column != target
    ]
    rows = []
    valid = frame[frame[target].isin([0, 1])] if target in frame.columns else pd.DataFrame()
    for column in candidates:
        subset = valid[[column, target]].dropna()
        if subset[column].nunique() < 2 or subset[target].nunique() < 2:
            rows.append(
                {
                    "variable": column,
                    "test": "Khi 2",
                    "statistic": pd.NA,
                    "p_value": pd.NA,
                    "interpretation": "Non interprétable",
                }
            )
            continue
        contingency = pd.crosstab(subset[column], subset[target])
        statistic, p_value, _, _ = chi2_contingency(contingency)
        rows.append(
            {
                "variable": column,
                "test": "Khi 2",
                "statistic": statistic,
                "p_value": p_value,
                "interpretation": "Association significative" if p_value < 0.05 else "Pas de signal significatif",
            }
        )
    return pd.DataFrame(rows)


def bad_rate_by_decile(
    frame: pd.DataFrame,
    variable: str = "pd_estimate",
    defaults_column: str = "default_flag_12m",
) -> pd.DataFrame:
    """Compute default rate by decile of a risk variable."""
    data = frame[[variable, defaults_column]].copy()
    data[variable] = pd.to_numeric(data[variable], errors="coerce")
    data = data[data[variable].notna() & data[defaults_column].isin([0, 1])]
    if data.empty or data[variable].nunique() < 2:
        return pd.DataFrame(columns=["decile", "observations", "defaults", "bad_rate", "mean_value"])

    risk_value = data[variable]
    if variable == "score":
        risk_value = -risk_value
    data["_risk_value"] = risk_value
    data["decile"] = pd.qcut(data["_risk_value"], q=min(10, data["_risk_value"].nunique()), labels=False, duplicates="drop") + 1
    result = (
        data.groupby("decile", dropna=False)
        .agg(
            observations=(defaults_column, "size"),
            defaults=(defaults_column, "sum"),
            bad_rate=(defaults_column, "mean"),
            mean_value=(variable, "mean"),
        )
        .reset_index()
    )
    result["decile"] = result["decile"].astype(int)
    return result


def run_hosmer_lemeshow_test(
    observations: pd.DataFrame,
    n_buckets: int = 10,
    min_observations: int = 30,
    min_defaults: int = 5,
) -> dict:
    """Run Hosmer-Lemeshow through the PD stat module with a local fallback."""
    if hasattr(stat_tests, "hosmer_lemeshow_test"):
        return stat_tests.hosmer_lemeshow_test(
            observations,
            n_buckets=n_buckets,
            min_observations=min_observations,
            min_defaults=min_defaults,
        )

    frame = observations[
        observations["pd_estimate"].notna()
        & observations["pd_estimate"].between(0, 1, inclusive="right")
        & observations["default_flag_12m"].isin([0, 1])
    ].copy()
    observed_defaults = int(frame["default_flag_12m"].sum()) if not frame.empty else 0
    if len(frame) < min_observations or observed_defaults < min_defaults or frame["pd_estimate"].nunique() < 2:
        return {
            "test_interpretable": False,
            "hl_statistic": pd.NA,
            "p_value": pd.NA,
            "hl_p_value": pd.NA,
            "hl_buckets": pd.DataFrame(),
        }

    frame["bucket"] = pd.qcut(
        frame["pd_estimate"],
        q=min(n_buckets, frame["pd_estimate"].nunique(), len(frame)),
        labels=False,
        duplicates="drop",
    )
    if frame["bucket"].nunique(dropna=True) < 2:
        return {
            "test_interpretable": False,
            "hl_statistic": pd.NA,
            "p_value": pd.NA,
            "hl_p_value": pd.NA,
            "hl_buckets": pd.DataFrame(),
        }
    frame["bucket"] = frame["bucket"].astype(int) + 1
    buckets = (
        frame.groupby("bucket")
        .agg(
            observations=("default_flag_12m", "size"),
            observed_defaults=("default_flag_12m", "sum"),
            expected_defaults=("pd_estimate", "sum"),
            pd_mean=("pd_estimate", "mean"),
        )
        .reset_index()
    )
    buckets["observed_non_defaults"] = buckets["observations"] - buckets["observed_defaults"]
    buckets["expected_non_defaults"] = buckets["observations"] - buckets["expected_defaults"]
    statistic = (
        ((buckets["observed_defaults"] - buckets["expected_defaults"]) ** 2)
        / buckets["expected_defaults"].clip(lower=1e-9)
        + ((buckets["observed_non_defaults"] - buckets["expected_non_defaults"]) ** 2)
        / buckets["expected_non_defaults"].clip(lower=1e-9)
    ).sum()
    degrees = max(len(buckets) - 2, 1)
    p_value = float(chi2.sf(statistic, degrees))
    buckets["odr"] = buckets["observed_defaults"] / buckets["observations"]
    buckets["calibration_gap"] = buckets["odr"] - buckets["pd_mean"]
    return {
        "test_interpretable": True,
        "hl_statistic": float(statistic),
        "p_value": p_value,
        "hl_p_value": p_value,
        "hl_degrees_freedom": int(degrees),
        "hl_buckets": buckets,
    }


def alert_row(
    theme: str,
    sub_section: str,
    test_name: str,
    objective: str,
    indicator: str,
    status: str,
    comment: str,
) -> dict:
    """Build one traffic-light synthesis row."""
    return {
        "theme": theme,
        "sous_partie": sub_section,
        "test": test_name,
        "objectif": objective,
        "indicateur": indicator,
        "status": status,
        "commentaire": comment,
    }


def first_status(frame: pd.DataFrame, default: str = "grey") -> str:
    """Combine statuses from a frame, returning grey when no status exists."""
    if frame.empty or "status" not in frame.columns:
        return default
    return combine_statuses(frame["status"].dropna().tolist())


def build_alert_synthesis(
    filtered: pd.DataFrame,
    thresholds: dict,
    dq_summary: pd.DataFrame,
    calibration_alerts: pd.DataFrame,
    discrimination_portfolio: pd.DataFrame,
    stability_global: pd.DataFrame,
    stability_segment: pd.DataFrame,
    monotonicity_alert: dict,
    periods: list,
) -> pd.DataFrame:
    """Build the Alertes tab synthesis using the same structure as analysis tabs."""
    rows = []
    dq_lookup = {row["controle"]: row for _, row in dq_summary.iterrows()} if not dq_summary.empty else {}
    for control, sub_section, objective in [
        ("Données manquantes", "Data quality", "Valider la complétude des champs nécessaires au backtesting."),
        ("Doublons observation_id", "Data quality", "Vérifier l'unicité des observations analysées."),
        ("PD hors bornes", "Data quality", "Contrôler que les PD sont des probabilités exploitables."),
        ("Date défaut manquante", "Data quality", "Vérifier la cohérence entre défaut observé et date de défaut."),
    ]:
        source = dq_lookup.get(control, {})
        rows.append(
            alert_row(
                "Data quality",
                sub_section,
                control,
                objective,
                source.get("indicateur", "Taux d'anomalie"),
                source.get("traffic_light", "grey"),
                f"Valeur: {source.get('valeur', 'n/a')} | seuil: {source.get('seuil', 'n/a')}",
            )
        )

    numeric_columns = filtered.select_dtypes(include="number").columns.tolist()
    categorical_columns = filtered.select_dtypes(include=["object", "category"]).columns.tolist()
    rows.extend(
        [
            alert_row(
                "Data quality",
                "Statistique descriptive univariée",
                "Distribution des variables",
                "Contrôler la forme des distributions continues et les répartitions discrètes.",
                "Graphique dynamique par variable",
                "green" if len(filtered) > 0 else "grey",
                "Disponible sur le périmètre filtré.",
            ),
            alert_row(
                "Data quality",
                "Statistique descriptive univariée",
                "Monitoring temporel des variables",
                "Vérifier si les distributions évoluent lorsque plusieurs périodes sont sélectionnées.",
                "Evolution par année",
                "green" if len(periods) >= 2 else "grey",
                "Non pertinent avec une seule période sélectionnée." if len(periods) < 2 else "Monitoring disponible.",
            ),
            alert_row(
                "Data quality",
                "Statistique descriptive multivariée",
                "Matrice de corrélation",
                "Identifier les dépendances entre variables numériques.",
                "Corrélations numériques",
                "green" if len(numeric_columns) >= 2 else "grey",
                "Nécessite au moins deux variables numériques.",
            ),
            alert_row(
                "Data quality",
                "Statistique descriptive multivariée",
                "Tests du Khi 2",
                "Tester l'association entre variables catégorielles et défaut 12 mois.",
                "p-value par variable catégorielle",
                "green" if categorical_columns and "default_flag_12m" in filtered.columns else "grey",
                "Disponible lorsque les variables catégorielles et le flag défaut sont présents.",
            ),
        ]
    )

    grade_counts = filtered["rating_grade"].fillna("Missing").value_counts() if "rating_grade" in filtered.columns else pd.Series(dtype=int)
    low_grade_count = int((grade_counts < thresholds["monotonicity"]["min_observations_per_grade"]).sum()) if not grade_counts.empty else 0
    rows.extend(
        [
            alert_row(
                "Stabilité Rating / Score Distribution (RDS)",
                "Distribution pour la période sélectionnée",
                "PSI",
                "Comparer la distribution courante à une période de référence.",
                "PSI global",
                first_status(stability_global),
                "PSI non calculable si aucune période de référence distincte n'est disponible." if stability_global.empty else "Synthèse du PSI global.",
            ),
            alert_row(
                "Stabilité Rating / Score Distribution (RDS)",
                "Distribution pour la période sélectionnée",
                "Distribution des rating grades",
                "Décrire la répartition de la population par grade.",
                "Bar chart par grade",
                "green" if not grade_counts.empty else "grey",
                "Distribution disponible sur le périmètre filtré.",
            ),
            alert_row(
                "Stabilité Rating / Score Distribution (RDS)",
                "Distribution pour la période sélectionnée",
                "Concentration par grade",
                "Identifier les grades concentrant une part élevée de population.",
                "% population par grade",
                "green" if not grade_counts.empty else "grey",
                "Lecture descriptive sans seuil réglementaire universel.",
            ),
            alert_row(
                "Stabilité Rating / Score Distribution (RDS)",
                "Distribution pour la période sélectionnée",
                "Grades peu peuplés",
                "Identifier les classes statistiquement fragiles.",
                f"{low_grade_count} grade(s) sous le seuil",
                "orange" if low_grade_count else ("green" if not grade_counts.empty else "grey"),
                "À interpréter avec prudence si des grades sont peu peuplés.",
            ),
            alert_row(
                "Stabilité Rating / Score Distribution (RDS)",
                "Distribution pour la période sélectionnée",
                "Distribution des buckets de PD",
                "Contrôler la dispersion des PD estimées.",
                "Histogramme PD",
                "green" if filtered.get("pd_estimate", pd.Series(dtype=float)).notna().any() else "grey",
                "Disponible si les PD sont renseignées.",
            ),
            alert_row(
                "Stabilité Rating / Score Distribution (RDS)",
                "Distribution pour la période sélectionnée",
                "Distribution des scores",
                "Comparer la distribution des scores à une référence.",
                "Histogramme score N vs référence",
                "green" if "score" in filtered.columns and len(periods) >= 2 else "grey",
                "Nécessite la variable score et au moins deux périodes.",
            ),
            alert_row(
                "Stabilité Rating / Score Distribution (RDS)",
                "Migration",
                "Migration des grades",
                "Mesurer les passages entre grades de référence et grades courants.",
                "Matrice de migration simplifiée",
                "green" if len(periods) >= 2 else "grey",
                "Nécessite deux périodes distinctes.",
            ),
            alert_row(
                "Stabilité Rating / Score Distribution (RDS)",
                "Migration",
                "Changement de mix produit / segment",
                "Vérifier si la population change structurellement.",
                "Evolution des poids",
                "green" if len(periods) >= 2 else "grey",
                "Disponible avec plusieurs périodes sélectionnées.",
            ),
            alert_row(
                "Stabilité Rating / Score Distribution (RDS)",
                "Migration",
                "Synthèse PSI par segment",
                "Prioriser les segments présentant une dérive de population.",
                "PSI par segment",
                first_status(stability_segment),
                "PSI segment non calculable sans période de référence." if stability_segment.empty else "Synthèse des statuts PSI par segment.",
            ),
        ]
    )

    calibration_global = calibration_alerts[calibration_alerts.get("aggregation_level") == "global"] if not calibration_alerts.empty else pd.DataFrame()
    calibration_status = first_status(calibration_global if not calibration_global.empty else calibration_alerts)
    rows.extend(
        [
            alert_row("Calibration PD", "Mesure du niveau de risque observé et estimé", "ODR", "Calculer le taux de défaut observé.", "ODR", "green", "Indicateur calculé sur le périmètre filtré."),
            alert_row("Calibration PD", "Mesure du niveau de risque observé et estimé", "PD moyenne estimée", "Mesurer le niveau moyen de risque estimé.", "PD moyenne", "green", "Indicateur calculé sur le périmètre filtré."),
            alert_row("Calibration PD", "Mesure du niveau de risque observé et estimé", "PD moyenne pondérée EAD", "Pondérer le risque par exposition.", "PD pondérée EAD", "green" if "ead_at_observation" in filtered.columns else "grey", "Nécessite EAD à l'observation."),
            alert_row("Calibration PD", "Comparaison défauts attendus vs observés", "Défauts attendus", "Comparer la somme des PD au nombre observé.", "Somme des PD", "green", "Indicateur calculé sur le périmètre filtré."),
            alert_row("Calibration PD", "Comparaison défauts attendus vs observés", "Défauts observés vs attendus", "Visualiser l'écart de niveau.", "Observé / attendu", calibration_status, "Statut issu des tests de calibration."),
            alert_row("Calibration PD", "Comparaison défauts attendus vs observés", "Calibration gap", "Mesurer l'écart absolu ODR - PD moyenne.", "Gap", calibration_status, "À lire avec le test statistique."),
            alert_row("Calibration PD", "Comparaison défauts attendus vs observés", "Calibration ratio", "Mesurer l'écart relatif ODR / PD moyenne.", "Ratio", calibration_status, "À lire avec le test statistique."),
            alert_row("Calibration PD", "Significativité statistique des écarts", "Test binomial bilatéral", "Tester tout écart significatif.", "p-value bilatérale", calibration_status, "Statut aligné sur les seuils de p-value."),
            alert_row("Calibration PD", "Significativité statistique des écarts", "Test binomial unilatéral haut", "Détecter la sous-estimation du risque.", "p-value unilatérale haute", calibration_status, "Test prioritaire pour signal de sous-estimation."),
            alert_row("Calibration PD", "Significativité statistique des écarts", "Intervalle de confiance ODR", "Encadrer l'incertitude statistique autour de l'ODR.", "IC Wilson / Clopper-Pearson", calibration_status, "Non interprétable si volume ou défauts insuffisants."),
            alert_row("Calibration PD", "Significativité statistique des écarts", "Hosmer-Lemeshow par buckets", "Tester la calibration par groupes ordonnés de risque.", "p-value HL", calibration_status, "Calculé dans l'onglet Calibration PD."),
            alert_row("Calibration PD", "Calibration temporelle et vintage", "Backtesting par année / vintage", "Suivre la calibration dans le temps et par millésime.", "ODR vs PD par année/vintage", "green" if len(periods) >= 2 else "grey", "Nécessite plusieurs périodes sélectionnées."),
            alert_row("Calibration PD", "Philosophie de rating", "Lecture PIT / TTC et low-default", "Contextualiser les écarts au regard de la philosophie du modèle.", "Commentaire méthodologique", "green", "Lecture qualitative dans le MVP."),
        ]
    )

    discrimination_status = first_status(discrimination_portfolio)
    monotonicity_status = monotonicity_alert.get("status", "grey")
    rows.extend(
        [
            alert_row("Discrimination", "Mesure globale du pouvoir discriminant", "AUC / ROC / Gini", "Mesurer la capacité à séparer défauts et non-défauts.", "AUC, ROC, Gini", discrimination_status, "Statut issu des seuils de discrimination."),
            alert_row("Discrimination", "Mesure globale du pouvoir discriminant", "KS statistic", "Mesurer l'écart maximal entre défauts et non-défauts.", "KS", discrimination_status, "Statut issu des seuils de discrimination."),
            alert_row("Discrimination", "Visualisation de la concentration du risque", "Courbe CAP", "Visualiser la concentration des défauts dans les classes risquées.", "Courbe CAP", discrimination_status, "Disponible si le test est interprétable."),
            alert_row("Discrimination", "Visualisation de la concentration du risque", "Bad rate par décile", "Vérifier la concentration progressive du risque.", "Bad rate par décile", "green" if filtered["default_flag_12m"].isin([0, 1]).any() else "grey", "Lecture descriptive par buckets de risque."),
            alert_row("Discrimination", "Ordonnancement du risque", "Ordonnancement des grades", "Vérifier que les taux de défaut augmentent avec le risque.", "Monotonie des grades", monotonicity_status, monotonicity_alert.get("comment", "")),
            alert_row("Discrimination", "Ordonnancement du risque", "Monotonie des taux observés", "Identifier les inversions entre grades adjacents.", "Nombre de violations", monotonicity_status, monotonicity_alert.get("comment", "")),
            alert_row("Discrimination", "Suivi temporel de la discrimination", "Evolution AUC / Gini / KS", "Identifier une dégradation temporelle du pouvoir discriminant.", "Graphique et tableau temporels", "green" if len(periods) >= 2 else "grey", "Non pertinent avec une seule période sélectionnée."),
        ]
    )
    return pd.DataFrame(rows)


def render_alert_synthesis_section(alerts: pd.DataFrame, theme: str) -> None:
    """Render one themed alert synthesis section."""
    theme_alerts = alerts[alerts["theme"] == theme].copy()
    if theme_alerts.empty:
        st.info(f"Aucune alerte disponible pour {theme}.")
        return
    st.markdown(f"### {theme}")
    traffic = theme_alerts.groupby(["sous_partie", "status"], as_index=False).size()
    st.plotly_chart(
        px.bar(
            traffic,
            x="sous_partie",
            y="size",
            color="status",
            color_discrete_map=STATUS_COLORS,
            labels={"sous_partie": "Sous-partie", "size": "Nombre de tests", "status": "Statut"},
        ),
        use_container_width=True,
    )
    for sub_section in theme_alerts["sous_partie"].drop_duplicates():
        st.markdown(f"#### {sub_section}")
        section_columns = ["test", "objectif", "indicateur", "status", "commentaire"]
        st.dataframe(theme_alerts[theme_alerts["sous_partie"] == sub_section][section_columns], use_container_width=True, hide_index=True)


def discrimination_over_time(frame: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Compute AUC, Gini and KS by observation year."""
    rows = []
    for year, group in frame.groupby("observation_year", dropna=False):
        metrics = calculate_discrimination_metrics(
            group,
            thresholds["minimum_volume"]["min_observations"],
            thresholds["minimum_volume"]["min_defaults"],
            thresholds["minimum_volume"]["min_non_defaults"],
        )
        rows.append(
            {
                "observation_year": year,
                "observations": metrics["observations"],
                "defaults": metrics["defaults"],
                "auc": metrics["auc"],
                "gini": metrics["gini"],
                "ks": metrics["ks"],
                "is_interpretable": metrics["is_interpretable"],
            }
        )
    return pd.DataFrame(rows).sort_values("observation_year")


def render_test_card(title: str, description: str, status: str) -> None:
    """Render a compact test explanation card."""
    color = STATUS_COLORS.get(status, "#7f7f7f")
    st.markdown(
        f"""
        <div style="border:1px solid rgba(11,43,70,0.14);background:rgba(255,255,255,0.76);
                    border-radius:22px;padding:16px 18px;margin:10px 0 12px 0;
                    box-shadow:0 12px 30px rgba(11,43,70,0.08)">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:12px">
            <div style="font-weight:900;color:var(--auria-navy);font-size:1.02rem">{title}</div>
            <div style="border-radius:999px;background:{color};color:white;padding:6px 10px;
                        font-size:.76rem;font-weight:900;text-transform:uppercase">{status}</div>
          </div>
          <div style="color:#6d7885;margin-top:6px">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cartography_presentation(thresholds: dict, parameter_name: str = "PD") -> None:
    """Render the test cartography as a theoretical presentation."""
    st.markdown("## Data quality")
    st.markdown("### Données manquantes")
    render_test_card(
        "Analyse des données manquantes",
        "Définition : mesure du taux de valeurs absentes par champ. Objectif : identifier les variables critiques qui peuvent fragiliser les calculs, les agrégations ou l'interprétation validation.",
        "inclus",
    )
    st.markdown(
        f"""
        **Formule** : nombre de valeurs manquantes / nombre d'observations.

        **Seuil théorique démo** : vert si le taux reste inférieur ou égal à {format_percent(thresholds["data_quality"]["max_error_rate"])} ; orange ou rouge si le taux augmente.
        """
    )
    render_test_card(
        "Duplication des données sur les ID",
        "Définition : contrôle des identifiants d'observation répétés. Objectif : éviter qu'une exposition soit comptée plusieurs fois dans les indicateurs de backtesting.",
        "inclus",
    )
    st.markdown(
        f"""
        **Formule** : nombre de lignes avec `observation_id` dupliqué / nombre total de lignes.

        **Seuil théorique démo** : cible proche de 0 %, seuil d'alerte aligné sur le seuil de data quality.
        """
    )

    st.markdown("### Cohérence métier")
    render_test_card(
        "PD hors bornes",
        "Définition : contrôle que chaque PD respecte 0 < PD <= 1. Objectif : garantir que les tests statistiques utilisent des probabilités valides.",
        "inclus",
    )
    st.markdown("**Formule** : part des lignes dont la PD est manquante, nulle, négative ou supérieure à 1.")
    render_test_card(
        "Default date manquante si flag défaut = 1",
        "Définition : contrôle de cohérence entre flag de défaut et date de défaut. Objectif : vérifier la traçabilité temporelle de la fenêtre 12 mois.",
        "inclus",
    )
    st.markdown("**Formule** : défauts avec date manquante / total des défauts observés.")

    st.markdown("## Stabilité Rating / Score Distribution (RDS)")
    st.markdown(
        f"""
        **Définition** : comparaison de distributions entre une période de référence et une période courante.

        **Objectif** : détecter une dérive de population sur les ratings, scores ou buckets de PD.

        **Formule PSI** : somme sur les buckets de `(part_courante - part_reference) x ln(part_courante / part_reference)`.

        **Seuils théoriques démo** : vert si PSI < {format_percent(thresholds["stability"]["psi_orange"])}, orange entre {format_percent(thresholds["stability"]["psi_orange"])} et {format_percent(thresholds["stability"]["psi_red"])}, rouge au-delà.
        """
    )

    st.markdown(f"## Calibration du paramêtre {parameter_name}")
    st.markdown(
        f"""
        **Objectif général** : vérifier si le niveau moyen du paramêtre estimé reste cohérent avec les réalisations observées.

        - **ODR** : défauts observés / observations.
        - **PD moyenne** : moyenne arithmétique des PD individuelles.
        - **Défauts attendus** : somme des PD individuelles.
        - **Calibration gap** : ODR - PD moyenne.
        - **Calibration ratio** : ODR / PD moyenne.
        - **Test binomial** : test statistique comparant le nombre de défauts observés au nombre attendu sous l'hypothèse PD moyenne.

        **Seuils théoriques démo** : orange si p-value < {format_percent(thresholds["calibration_tests"]["binomial"]["alpha_orange"])}, rouge si p-value < {format_percent(thresholds["calibration_tests"]["binomial"]["alpha_red"])}.
        """
    )

    st.markdown("## Discrimination")
    st.markdown(
        f"""
        **Objectif général** : vérifier si le modèle classe correctement les emprunteurs risqués avant les emprunteurs moins risqués.

        - **AUC** : probabilité qu'un défaut ait un score de risque supérieur à un non-défaut.
        - **Gini** : `2 x AUC - 1`.
        - **KS** : écart maximal entre distributions cumulées des défauts et non-défauts.
        - **ROC / CAP** : visualisations du pouvoir de classement.

        **Seuils théoriques démo** : AUC verte si >= {format_percent(thresholds["discrimination"]["auc_orange"])}, orange entre {format_percent(thresholds["discrimination"]["auc_red"])} et {format_percent(thresholds["discrimination"]["auc_orange"])}, rouge en-dessous.
        """
    )


if hasattr(st, "dialog"):
    @st.dialog("Cartographie des tests")
    def test_cartography_dialog(thresholds: dict) -> None:
        render_cartography_presentation(thresholds)

    @st.dialog("Ancrages réglementaires")
    def regulatory_anchors_dialog() -> None:
        render_regulatory_anchors()
else:
    def test_cartography_dialog(thresholds: dict) -> None:
        with st.expander("Cartographie des tests", expanded=True):
            render_cartography_presentation(thresholds)

    def regulatory_anchors_dialog() -> None:
        with st.expander("Ancrages réglementaires", expanded=True):
            render_regulatory_anchors()


def render_data_dictionary_body(frame: pd.DataFrame) -> None:
    """Render the PD base data dictionary."""
    dictionary = build_data_dictionary(frame)
    st.markdown(
        "Ce dictionnaire documente les champs présents dans la base PD filtrée : libellé métier, catégorie, "
        "type, description, caractéristiques, complétude et exemples de valeurs."
    )
    categories = ["Toutes"] + sorted(dictionary["categorie"].dropna().unique().tolist())
    selected_category = st.selectbox("Catégorie", categories, key="data_dictionary_category")
    search = st.text_input("Rechercher un champ ou une description", key="data_dictionary_search")
    view = dictionary.copy()
    if selected_category != "Toutes":
        view = view[view["categorie"] == selected_category]
    if search:
        search_lower = search.lower()
        mask = view.apply(lambda row: search_lower in " ".join(row.astype(str).tolist()).lower(), axis=1)
        view = view[mask]
    st.metric("Champs documentés", len(view))
    st.dataframe(display_rates(view), use_container_width=True, hide_index=True)


if hasattr(st, "dialog"):
    @st.dialog("Dictionnaire de données")
    def data_dictionary_dialog(frame: pd.DataFrame) -> None:
        render_data_dictionary_body(frame)
else:
    def data_dictionary_dialog(frame: pd.DataFrame) -> None:
        with st.expander("Dictionnaire de données", expanded=True):
            render_data_dictionary_body(frame)


def render_stability_rds_section(filtered: pd.DataFrame, thresholds: dict, periods: list) -> None:
    """Render Rating / Score Distribution stability analysis."""
    st.subheader("Stabilité Rating / Score Distribution (RDS)")
    st.caption(
        "Cette section vérifie si la distribution des ratings, scores ou buckets de PD reste comparable entre deux périodes."
    )
    st.markdown(
        "Objectif : identifier une dérive de population susceptible d'expliquer un changement de performance, "
        "de calibration ou de discrimination du modèle."
    )

    if not periods:
        st.warning("Aucune période disponible sur le périmètre filtré.")
        return

    analysis_period = st.selectbox(
        "Période d'analyse RDS",
        periods,
        index=len(periods) - 1,
        key="rds_analysis_period",
    )
    reference_candidates = [period for period in periods if period != analysis_period]
    reference_period = reference_candidates[0] if reference_candidates else analysis_period
    if reference_candidates:
        reference_period = st.selectbox(
            "Période de référence RDS",
            reference_candidates,
            index=0,
            key="rds_reference_period",
        )

    current = filtered[filtered["observation_year"] == analysis_period]
    reference = filtered[filtered["observation_year"] == reference_period]

    st.markdown(f"### Distribution pour la période {analysis_period}")
    st.markdown(
        "Cette partie décrit la population de la période sélectionnée : stabilité globale, répartition des grades, "
        "classes peu peuplées, buckets de PD et distribution des scores."
    )

    if analysis_period == reference_period:
        st.warning("Le PSI nécessite deux périodes distinctes. Sélectionnez au moins deux périodes dans le filtre Période.")
    else:
        psi_global = build_stability_alerts(filtered, thresholds, reference_period, analysis_period)
        st.markdown("#### 1. PSI")
        st.dataframe(display_rates(psi_global), use_container_width=True, hide_index=True)

    st.markdown("#### 2. Distribution des rating grades")
    grade_counts = (
        current["rating_grade"]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    grade_counts.columns = ["rating_grade", "count"]
    st.plotly_chart(
        px.bar(
            grade_counts,
            x="rating_grade",
            y="count",
            text_auto=True,
            color_discrete_sequence=[AURIA_COLORS["navy"]],
            labels={"rating_grade": "Rating grade", "count": "Effectif"},
        ),
        use_container_width=True,
    )

    st.markdown("#### 2bis. Concentration par grade")
    total_current = max(len(current), 1)
    grade_counts["population_share"] = grade_counts["count"] / total_current
    st.dataframe(
        display_rates(grade_counts[["rating_grade", "count", "population_share"]]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 2ter. Grades peu peuplés")
    grade_threshold = thresholds["monotonicity"]["min_observations_per_grade"]
    low_populated = grade_counts[grade_counts["count"] < grade_threshold].copy()
    low_populated["seuil_minimum"] = grade_threshold
    low_populated["interpretation"] = "Classe statistiquement fragile"
    if low_populated.empty:
        st.success("Aucun grade peu peuplé sur la période sélectionnée.")
    else:
        st.dataframe(low_populated, use_container_width=True, hide_index=True)

    st.markdown("#### 3. Distribution des buckets de PD")
    pd_values = pd.to_numeric(current["pd_estimate"], errors="coerce").dropna()
    if pd_values.empty:
        st.warning("Aucune PD exploitable pour construire les buckets.")
    else:
        st.plotly_chart(
            px.histogram(
                current,
                x="pd_estimate",
                nbins=20,
                color="portfolio" if "portfolio" in current.columns else None,
                labels={"pd_estimate": "PD estimée"},
            ),
            use_container_width=True,
        )

    st.markdown("#### 4. Distribution des scores")
    if "score" not in filtered.columns:
        st.warning("La variable score n'est pas disponible.")
    elif analysis_period == reference_period:
        st.warning("La comparaison score période N vs référence nécessite deux périodes distinctes.")
    else:
        score_compare = pd.concat(
            [
                reference.assign(_periode=f"Référence {reference_period}"),
                current.assign(_periode=f"Période {analysis_period}"),
            ],
            ignore_index=True,
        )
        st.plotly_chart(
            px.histogram(
                score_compare,
                x="score",
                color="_periode",
                nbins=35,
                barmode="overlay",
                opacity=0.62,
                labels={"score": "Score", "_periode": "Période"},
                color_discrete_sequence=[AURIA_COLORS["navy"], AURIA_COLORS["peach"]],
            ),
            use_container_width=True,
        )

    st.markdown("### Migration")
    st.markdown(
        "Cette partie vérifie si les contreparties changent de grade ou si la structure de population évolue par produit, segment ou portefeuille."
    )

    st.markdown("#### 1. Migration des grades")
    if analysis_period == reference_period:
        st.warning("La matrice de migration nécessite une période de référence distincte.")
    else:
        ref_grade = (
            reference.sort_values("observation_date")
            .dropna(subset=["obligor_id", "rating_grade"])
            .drop_duplicates("obligor_id", keep="last")[["obligor_id", "rating_grade"]]
            .rename(columns={"rating_grade": "grade_reference"})
        )
        cur_grade = (
            current.sort_values("observation_date")
            .dropna(subset=["obligor_id", "rating_grade"])
            .drop_duplicates("obligor_id", keep="last")[["obligor_id", "rating_grade"]]
            .rename(columns={"rating_grade": "grade_courant"})
        )
        migration = ref_grade.merge(cur_grade, on="obligor_id", how="inner")
        if migration.empty:
            st.warning("Aucune contrepartie commune entre les deux périodes pour construire une matrice de migration.")
        else:
            matrix = pd.crosstab(migration["grade_reference"], migration["grade_courant"], normalize="index")
            st.plotly_chart(
                px.imshow(
                    matrix,
                    text_auto=".1%",
                    color_continuous_scale=AURIA_CONTINUOUS_SCALE,
                    labels={"x": "Grade courant", "y": "Grade référence", "color": "Part"},
                    title="Matrice de migration simplifiée",
                ),
                use_container_width=True,
            )

    st.markdown("#### 2. Changement de mix produit / segment")
    mix_dimension = st.selectbox(
        "Dimension de mix",
        [column for column in ["portfolio", "product_type", "segment"] if column in filtered.columns],
        index=2 if "segment" in filtered.columns else 0,
        key="rds_mix_dimension",
    )
    mix = (
        filtered.assign(_mix=filtered[mix_dimension].fillna("Missing").astype(str))
        .groupby(["observation_year", "_mix"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    mix["share"] = mix["count"] / mix.groupby("observation_year")["count"].transform("sum")
    st.plotly_chart(
        px.area(
            mix,
            x="observation_year",
            y="share",
            color="_mix",
            labels={"observation_year": "Période", "share": "Poids", "_mix": mix_dimension},
        ),
        use_container_width=True,
    )

    st.markdown("#### Synthèse PSI par segment")
    if len(periods) < 2:
        st.warning("Aucune période de référence distincte n'est disponible pour le PSI.")
    else:
        stability_segment_view = build_stability_alerts(
            filtered, thresholds, reference_period, analysis_period, "rating_grade", ["segment"]
        )

    st.markdown("### Stabilité par philosophie de modèle")
    st.markdown(
        "Cette lecture compare la stabilité des distributions par philosophie déclarée. Elle aide à interpréter "
        "les dérives RDS sans conclure automatiquement qu'une philosophie est meilleure qu'une autre."
    )
    if analysis_period == reference_period:
        st.warning("La stabilité par philosophie nécessite une période de référence distincte.")
    else:
        philosophy_rds = compare_rds_by_philosophy(filtered, reference_period, analysis_period)
        if philosophy_rds.empty:
            st.warning("PSI/RDS par philosophie non calculable sur le périmètre filtré.")
        else:
            st.dataframe(display_rates(philosophy_rds), use_container_width=True, hide_index=True)
            st.plotly_chart(
                px.bar(
                    philosophy_rds,
                    x="model_philosophy",
                    y="psi",
                    color="status",
                    color_discrete_map=STATUS_COLORS,
                    labels={"model_philosophy": "Philosophie", "psi": "PSI", "status": "Statut"},
                    title="PSI rating grade par philosophie",
                ),
                use_container_width=True,
            )
    if "model_philosophy" in current.columns:
        distribution_variable = "pd_bucket" if "pd_bucket" in current.columns else "rating_grade"
        philosophy_distribution = (
            current.assign(
                _philosophy=current["model_philosophy"].fillna("Unknown").astype(str),
                _bucket=current[distribution_variable].fillna("Missing").astype(str),
            )
            .groupby(["_philosophy", "_bucket"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        philosophy_distribution["share"] = philosophy_distribution["count"] / philosophy_distribution.groupby("_philosophy")["count"].transform("sum")
        st.plotly_chart(
            px.bar(
                philosophy_distribution,
                x="_bucket",
                y="share",
                color="_philosophy",
                barmode="group",
                labels={"_bucket": distribution_variable, "share": "Part", "_philosophy": "Philosophie"},
                title=f"Distribution {distribution_variable} par philosophie",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(
                stability_segment_view.sort_values("psi", ascending=False),
                x="perimeter",
                y="psi",
                color="status",
                color_discrete_map=STATUS_COLORS,
                labels={"perimeter": "Segment", "psi": "PSI", "status": "Statut"},
            ),
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="Validation PD & Rating Systems", layout="wide")
    configure_plotly_theme()
    inject_auria_theme()
    render_auria_topbar()

    raw_thresholds = load_thresholds(CONFIG_PATH)
    profile_names = sorted(raw_thresholds.get("threshold_profiles", {"standard": {}}))

    catalog = scenario_catalog()
    init_demo_state(profile_names)

    scenario_choice = st.session_state["scenario_choice"]
    threshold_profile = st.session_state["threshold_profile"]
    retail_n = st.session_state["retail_n"]
    corporate_n = st.session_state["corporate_n"]
    start_year = st.session_state["start_year"]
    years_count = st.session_state["years_count"]
    dq_level = st.session_state["dq_level"]
    random_seed = st.session_state["random_seed"]

    try:
        thresholds = apply_threshold_profile(raw_thresholds, threshold_profile)
    except ValueError as exc:
        st.error(f"Seuils YAML invalides : {exc}")
        st.stop()
    threshold_errors = validate_thresholds(thresholds)
    if threshold_errors:
        st.error("Seuils YAML invalides : " + " ; ".join(threshold_errors))
        st.stop()

    try:
        if scenario_choice == "current_file":
            if not DATA_PATH.exists():
                st.error("Aucun fichier charge. Generez un scenario ou creez data/generated/pd_observations.csv.")
                st.stop()
            observations = cached_load_observations(str(DATA_PATH), DATA_PATH.stat().st_mtime)
            scenario_description = "Fichier courant charge depuis data/generated/pd_observations.csv."
            expected_observation = "Les resultats dependent du fichier charge."
        else:
            observations = cached_generate_scenario(
                scenario_choice,
                int(retail_n),
                int(corporate_n),
                int(start_year),
                int(years_count),
                dq_level,
                int(random_seed),
            )
            scenario_description = catalog[scenario_choice].description
            expected_observation = catalog[scenario_choice].expected_observation
    except Exception as exc:
        st.error(f"Erreur lors du chargement ou de la generation des donnees : {exc}")
        st.stop()

    observations = ensure_pd_12m_fields(observations)
    missing_columns = validate_observation_schema(observations.columns)
    if missing_columns:
        st.error("Colonnes obligatoires absentes : " + ", ".join(missing_columns))
        st.stop()
    if observations.empty:
        st.error("Le dataset est vide.")
        st.stop()

    observations = add_observation_year(observations)
    scenario_label = "Fichier courant" if scenario_choice == "current_file" else catalog[scenario_choice].label
    render_scope_presentation(
        observations,
        scenario_label,
        scenario_description,
        expected_observation,
        threshold_profile,
    )
    action_cols = st.columns([0.16, 0.22, 0.24, 0.38])
    if action_cols[0].button("Parametres demo", type="secondary"):
        demo_settings_dialog(catalog, profile_names)
    if action_cols[1].button("Cartographie des tests", type="secondary"):
        test_cartography_dialog(thresholds)
    if action_cols[2].button("Ancrages réglementaires", type="secondary"):
        regulatory_anchors_dialog()

    portfolios, segments, model_ids, years, aggregation_level = render_analysis_filters(observations)

    filtered = filter_dataframe(observations, portfolios, segments, model_ids, years)
    if filtered.empty:
        st.error("Les filtres selectionnes vident la population. Ajustez la sidebar.")
        st.stop()
    if filtered["default_flag_12m"].fillna(0).sum() == 0:
        st.warning("Aucun defaut observe dans la population filtree : certains tests seront gris.")

    periods = sorted(filtered["observation_year"].dropna().unique().tolist())
    reference_period = periods[0] if periods else None
    current_period = periods[-1] if periods else None

    expected_window = thresholds["data_quality"]["expected_performance_window_months"]
    global_metrics = portfolio_metrics(filtered)
    aggregations = build_standard_aggregations(filtered)
    calibration_alerts = add_interpretability(build_calibration_alerts(filtered, thresholds), thresholds, "calibration")
    discrimination_global = calculate_discrimination_metrics(
        filtered,
        thresholds["minimum_volume"]["min_observations"],
        thresholds["minimum_volume"]["min_defaults"],
        thresholds["minimum_volume"]["min_non_defaults"],
    )
    discrimination_portfolio = add_interpretability(
        build_discrimination_alerts(filtered, thresholds, ["portfolio"]), thresholds, "discrimination"
    )
    discrimination_segment = add_interpretability(
        build_discrimination_alerts(filtered, thresholds, ["segment"]), thresholds, "discrimination"
    )
    stability_global = pd.DataFrame()
    stability_portfolio = pd.DataFrame()
    stability_segment = pd.DataFrame()
    if reference_period is not None and current_period is not None and reference_period != current_period:
        stability_global = add_interpretability(
            build_stability_alerts(filtered, thresholds, reference_period, current_period), thresholds, "stability"
        )
        stability_portfolio = add_interpretability(
            build_stability_alerts(filtered, thresholds, reference_period, current_period, "rating_grade", ["portfolio"]),
            thresholds,
            "stability",
        )
        stability_segment = add_interpretability(
            build_stability_alerts(filtered, thresholds, reference_period, current_period, "rating_grade", ["segment"]),
            thresholds,
            "stability",
        )
    dq_results = run_data_quality_checks(filtered, expected_window)
    eligibility_summary = summarize_population_eligibility(filtered, thresholds)
    eligibility_exclusions = summarize_exclusions(filtered)
    eligibility_waterfall = build_population_waterfall(filtered)
    exclusion_category = summarize_exclusions_by_category(filtered)
    exclusion_materiality = summarize_exclusions_by_materiality(filtered)
    exclusion_ead_impact = calculate_exclusion_ead_impact(filtered)
    material_exclusions = identify_material_exclusions(filtered, thresholds)
    exclusion_findings = generate_exclusion_findings(filtered)
    monotonicity_results, monotonicity_alert = build_monotonicity_alert(filtered, thresholds)
    monotonicity_alerts = pd.DataFrame([monotonicity_alert])
    rds_summary = pd.DataFrame()
    rds_drivers = pd.DataFrame()
    migration_summary = summarize_migration(filtered, thresholds)
    migration_matrices = build_rating_migration_matrix(filtered)
    notch_distribution = calculate_notch_migration_distribution(filtered)
    migration_patterns = identify_material_migration_patterns(filtered)
    philosophy_summary = summarize_model_philosophy(filtered)
    philosophy_pd_summary = compare_pd_by_philosophy(filtered)
    philosophy_behaviour = compare_pit_ttc_behaviour(filtered)
    philosophy_volatility = analyze_pd_volatility_by_philosophy(filtered)
    philosophy_rds = compare_rds_by_philosophy(filtered, reference_period, current_period)
    philosophy_comment = generate_philosophy_commentary(filtered, calibration_alerts, stability_global)
    pd_components = summarize_pd_components(filtered)
    pd_floor_impact = analyze_pd_floor_impact(filtered, thresholds)
    moc_impact = analyze_moc_impact(filtered, thresholds)
    pd_layer_comparison = compare_raw_calibrated_regulatory_pd(filtered)
    ldp_segments = identify_low_default_segments(filtered, thresholds)
    ldp_multiyear = calculate_multi_year_default_summary(filtered)
    ldp_recommendation = recommend_ldp_treatment(filtered)
    population_diagnostic = {}
    new_exited_summary = summarize_new_and_exited_customers(filtered)
    if reference_period is not None and current_period is not None and reference_period != current_period:
        rds_summary = summarize_rds_stability(filtered, reference_period, current_period, thresholds)
        rds_drivers = diagnose_rds_change(filtered, reference_period, current_period)
        population_diagnostic = diagnose_population_change(filtered, reference_period, current_period, thresholds)
    summary = build_model_summary(filtered, calibration_alerts, discrimination_portfolio, stability_portfolio, current_period)
    global_status = combine_statuses(summary["statut_global"].tolist()) if not summary.empty else "grey"
    counts = status_counts(calibration_alerts, discrimination_portfolio, stability_portfolio, monotonicity_alerts)
    status_dist = status_distribution(calibration_alerts, discrimination_portfolio, stability_portfolio, monotonicity_alerts)
    executive_theme_status = build_executive_theme_status(
        dq_results,
        calibration_alerts,
        discrimination_portfolio,
        stability_portfolio,
        monotonicity_alerts,
        thresholds,
    )
    executive_portfolio_view = build_portfolio_executive_view(filtered)
    test_mapping = build_test_mapping()
    findings = generate_validation_findings(
        calibration_alerts.assign(test_family="Calibration"),
        discrimination_portfolio.assign(test_family="Discrimination"),
        stability_portfolio.assign(test_family="Stabilité RDS") if not stability_portfolio.empty else pd.DataFrame(),
        monotonicity_alerts,
    )
    if not exclusion_findings.empty:
        findings = pd.concat([findings, exclusion_findings], ignore_index=True)

    tabs = st.tabs([
        "Accueil",
        "Données & qualité",
        "Stabilité RDS",
        "Calibration PD",
        "Discrimination",
        "Alertes",
        "Rapport & export",
        "Méthodologie",
    ])

    with tabs[0]:
        render_section_header(
            "Synthèse exécutive",
            "Vue consolidée du périmètre filtré : niveau de risque, robustesse statistique, alertes et statut global du système de notation.",
            "Dashboard",
        )
        st.info(f"Scénario : {scenario_description}\n\nÀ observer : {expected_observation}")

        headline_cols = st.columns([1.05, 1, 1])
        with headline_cols[0]:
            render_kpi_card("Statut global", global_status, "Lecture consolidée du périmètre filtré")
        with headline_cols[1]:
            render_kpi_card(
                "Population analysée",
                f"{global_metrics['observations']:,}".replace(",", " "),
                f"{global_metrics['observed_defaults']:,}".replace(",", " ") + " défauts observés",
            )
        with headline_cols[2]:
            render_kpi_card(
                "Niveau de risque",
                f"PD {format_percent(global_metrics['pd_mean'])}",
                f"ODR {format_percent(global_metrics['odr'])} | profil {threshold_profile}",
            )

        st.markdown("### Lecture rapide des statuts")
        render_status_strip(executive_theme_status)

        visual_cols = st.columns([0.9, 1.25])
        with visual_cols[0]:
            alert_chart = status_dist[status_dist["count"] > 0].copy()
            if alert_chart.empty:
                alert_chart = pd.DataFrame([{"status": "green", "count": 1}])
            fig_alerts = px.pie(
                alert_chart,
                names="status",
                values="count",
                hole=0.62,
                color="status",
                color_discrete_map=STATUS_COLORS,
                title="Répartition des traffic lights",
            )
            fig_alerts.update_traces(textposition="inside", textinfo="label+value")
            fig_alerts.update_layout(showlegend=False)
            st.plotly_chart(fig_alerts, use_container_width=True)

        with visual_cols[1]:
            theme_chart = executive_theme_status.copy()
            theme_chart["score"] = 1
            fig_theme = px.bar(
                theme_chart,
                x="score",
                y="theme",
                color="status",
                orientation="h",
                color_discrete_map=STATUS_COLORS,
                labels={"score": "", "theme": "Thème", "status": "Statut"},
                title="Statut par domaine d'analyse",
            )
            fig_theme.update_layout(showlegend=False, xaxis_visible=False, height=360)
            st.plotly_chart(fig_theme, use_container_width=True)

        st.markdown("### Risque estimé vs risque observé")
        if not executive_portfolio_view.empty:
            risk_chart = executive_portfolio_view.melt(
                id_vars=["portfolio"],
                value_vars=["PD moyenne", "ODR"],
                var_name="indicateur",
                value_name="taux",
            )
            st.plotly_chart(
                px.bar(
                    risk_chart,
                    x="portfolio",
                    y="taux",
                    color="indicateur",
                    barmode="group",
                    text_auto=".2%",
                    labels={"portfolio": "Portefeuille", "taux": "Taux", "indicateur": "Indicateur"},
                    title="PD moyenne vs taux de défaut observé par portefeuille",
                ),
                use_container_width=True,
            )

            default_chart = executive_portfolio_view.melt(
                id_vars=["portfolio"],
                value_vars=["Défauts attendus", "Défauts observés"],
                var_name="indicateur",
                value_name="nombre",
            )
            st.plotly_chart(
                px.bar(
                    default_chart,
                    x="portfolio",
                    y="nombre",
                    color="indicateur",
                    barmode="group",
                    text_auto=".1f",
                    labels={"portfolio": "Portefeuille", "nombre": "Nombre de défauts", "indicateur": "Indicateur"},
                    title="Défauts attendus vs défauts observés par portefeuille",
                ),
                use_container_width=True,
            )

        st.markdown("### Points d'attention prioritaires")
        if not findings.empty and "niveau_de_severite" in findings.columns:
            priority_findings = findings[
                findings["niveau_de_severite"].isin(["Haute", "Moyenne", "À qualifier", "Ã€ qualifier"])
            ]
        else:
            priority_findings = pd.DataFrame()
        if priority_findings.empty:
            st.success("Aucun finding prioritaire n'est remonté sur le périmètre filtré.")
        else:
            display_columns = [
                column for column in ["finding_id", "theme", "perimetre", "constat", "niveau_de_severite", "recommandation"]
                if column in priority_findings.columns
            ]
            st.dataframe(priority_findings[display_columns].head(6), use_container_width=True, hide_index=True)

        with st.expander("Voir la table de synthèse modèle détaillée"):
            st.dataframe(display_rates(summary), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Donnees & qualite")
        if st.button("Dictionnaire de données", type="secondary"):
            data_dictionary_dialog(filtered)
        st.markdown(
            "Cette section vérifie que le dataset filtré est exploitable avant toute conclusion de backtesting : "
            "complétude, unicité, cohérence métier, distributions des variables et dépendances statistiques simples."
        )

        st.markdown("### Data quality")
        dq_threshold = thresholds["data_quality"]["max_error_rate"]
        missing = (
            filtered.isna()
            .mean()
            .rename("missing_rate")
            .reset_index()
            .rename(columns={"index": "column"})
            .sort_values("missing_rate", ascending=False)
        )
        st.markdown("#### Données manquantes")
        st.caption("Ce graphique montre le pourcentage de valeurs absentes par champ. Il permet d'identifier les variables qui peuvent fragiliser les calculs et les segmentations.")
        st.plotly_chart(
            px.bar(
                missing.head(18),
                x="column",
                y="missing_rate",
                text_auto=".1%",
                labels={"column": "Champ", "missing_rate": "% manquant"},
                color_discrete_sequence=[AURIA_COLORS["navy"]],
            ),
            use_container_width=True,
        )
        duplicate_rate = filtered["observation_id"].duplicated(keep=False).mean()
        pd_valid = filtered["pd_estimate"].notna() & filtered["pd_estimate"].between(0, 1, inclusive="right")
        defaults = filtered[filtered["default_flag_12m"] == 1]
        missing_default_date = int(defaults["default_date"].isna().sum()) if "default_date" in defaults.columns else 0
        dq_summary = pd.DataFrame(
            [
                {
                    "controle": "Données manquantes",
                    "indicateur": "Taux maximum par champ",
                    "valeur": format_percent(float(missing["missing_rate"].max()) if not missing.empty else 0.0),
                    "seuil": format_percent(dq_threshold),
                    "traffic_light": traffic_from_rate(float(missing["missing_rate"].max()) if not missing.empty else 0.0, dq_threshold),
                },
                {
                    "controle": "Doublons observation_id",
                    "indicateur": "% lignes avec ID dupliqué",
                    "valeur": format_percent(duplicate_rate),
                    "seuil": format_percent(dq_threshold),
                    "traffic_light": traffic_from_rate(duplicate_rate, dq_threshold),
                },
                {
                    "controle": "PD hors bornes",
                    "indicateur": "% non conforme",
                    "valeur": format_percent((~pd_valid).mean()),
                    "seuil": format_percent(dq_threshold),
                    "traffic_light": traffic_from_rate((~pd_valid).mean(), dq_threshold),
                },
                {
                    "controle": "Date défaut manquante",
                    "indicateur": "% défauts sans date",
                    "valeur": format_percent(missing_default_date / len(defaults) if len(defaults) else pd.NA),
                    "seuil": format_percent(dq_threshold),
                    "traffic_light": traffic_from_rate(missing_default_date / len(defaults) if len(defaults) else pd.NA, dq_threshold),
                },
            ]
        )
        st.dataframe(dq_summary, use_container_width=True, hide_index=True)
        st.markdown("#### Cohérence métier")
        st.caption("Ces graphiques résument deux contrôles métier clés : validità des probabilités de défaut et cohérence entre défaut observé et date de défaut.")
        dq_cols = st.columns(2)
        dq_cols[0].plotly_chart(
            compliance_pie(
                ["Conforme", "Non conforme"],
                [int(pd_valid.sum()), int((~pd_valid).sum())],
                "Conformité des PD",
            ),
            use_container_width=True,
        )
        dq_cols[1].plotly_chart(
            compliance_pie(
                ["Conforme", "Non conforme"],
                [int(len(defaults) - missing_default_date), missing_default_date],
                "Date de défaut renseignée",
            ),
            use_container_width=True,
        )
        st.dataframe(display_rates(dq_results), use_container_width=True, hide_index=True)

        st.markdown("### Population réglementaire éligible")
        st.markdown(
            "Cette sous-partie distingue la population initiale de la population réellement éligible au backtesting PD 12 mois : "
            "hors scope, défaut à l'observation, horizon incomplet, maturité avant horizon et données essentielles manquantes."
        )
        st.dataframe(display_rates(eligibility_summary), use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(
                eligibility_waterfall,
                x="step",
                y="observations",
                title="Waterfall population initiale vers population backtestée",
                color_discrete_sequence=[AURIA_COLORS["navy"]],
            ),
            use_container_width=True,
        )
        st.dataframe(display_rates(eligibility_exclusions), use_container_width=True, hide_index=True)

        st.markdown("### Analyse fine des exclusions")
        st.markdown(
            "Objectif : qualifier les exclusions par catégorie, règle, matérialité et EAD afin de vérifier qu'elles sont traçables et non biaisantes."
        )
        exc_cols = st.columns(2)
        exc_cols[0].dataframe(display_rates(exclusion_category), use_container_width=True, hide_index=True)
        exc_cols[1].dataframe(display_rates(exclusion_materiality), use_container_width=True, hide_index=True)
        st.dataframe(display_rates(exclusion_ead_impact), use_container_width=True, hide_index=True)
        st.dataframe(display_rates(material_exclusions), use_container_width=True, hide_index=True)

        st.markdown("### Statistique descriptive univariée")
        st.markdown(
            "Sélectionnez une variable pour visualiser sa distribution sur le périmètre filtré. "
            "Les variables continues sont affichées en histogramme ; les variables discrètes en bar chart."
        )
        excluded_for_univariate = {"default_date"}
        univariate_columns = [
            column
            for column in filtered.columns
            if column not in excluded_for_univariate and not column.startswith("_")
        ]
        selected_variable = st.selectbox("Variable à analyser", univariate_columns)
        kind = variable_kind(filtered, selected_variable)
        if kind == "continuous":
            st.plotly_chart(
                px.histogram(
                    filtered,
                    x=selected_variable,
                    nbins=40,
                    color="portfolio" if "portfolio" in filtered.columns else None,
                    labels={selected_variable: selected_variable},
                    color_discrete_sequence=[AURIA_COLORS["navy"], AURIA_COLORS["peach"]],
                ),
                use_container_width=True,
            )
        else:
            counts = (
                filtered[selected_variable]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(30)
                .reset_index()
            )
            counts.columns = [selected_variable, "count"]
            st.plotly_chart(
                px.bar(
                    counts,
                    x=selected_variable,
                    y="count",
                    color_discrete_sequence=[AURIA_COLORS["navy"]],
                ),
                use_container_width=True,
            )

        st.markdown("#### Monitoring temporel de la variable")
        if len(periods) < 2:
            st.warning("Le monitoring temporel n'est pas disponible : une seule période est sélectionnée dans le filtre Période.")
        else:
            if kind == "continuous":
                monitor = (
                    filtered.groupby("observation_year", dropna=False)[selected_variable]
                    .mean()
                    .reset_index(name="mean_value")
                )
                st.plotly_chart(
                    px.line(
                        monitor,
                        x="observation_year",
                        y="mean_value",
                        markers=True,
                        labels={"observation_year": "Période", "mean_value": f"Moyenne {selected_variable}"},
                        color_discrete_sequence=[AURIA_COLORS["navy"]],
                    ),
                    use_container_width=True,
                )
            else:
                monitor = (
                    filtered.assign(_value=filtered[selected_variable].fillna("Missing").astype(str))
                    .groupby(["observation_year", "_value"], dropna=False)
                    .size()
                    .reset_index(name="count")
                )
                total = monitor.groupby("observation_year")["count"].transform("sum")
                monitor["share"] = monitor["count"] / total
                top_values = (
                    monitor.groupby("_value")["count"].sum().sort_values(ascending=False).head(10).index
                )
                st.plotly_chart(
                    px.area(
                        monitor[monitor["_value"].isin(top_values)],
                        x="observation_year",
                        y="share",
                        color="_value",
                        labels={"observation_year": "Période", "share": "Part", "_value": selected_variable},
                    ),
                    use_container_width=True,
                )

        st.markdown("### Statistique descriptive multivariée")
        st.markdown("Cette partie donne une première lecture des dépendances entre variables numériques et catégorielles sur la population filtrée.")
        numeric_columns = [
            column
            for column in ["score", "pd_estimate", "ead_at_observation", "default_flag_12m", "performance_window_months", "observation_year"]
            if column in filtered.columns
        ]
        corr_data = filtered[numeric_columns].apply(pd.to_numeric, errors="coerce").dropna(how="all")
        if len(numeric_columns) >= 2 and not corr_data.empty:
            corr = corr_data.corr(numeric_only=True)
            st.plotly_chart(
                px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale=AURIA_CONTINUOUS_SCALE,
                    zmin=-1,
                    zmax=1,
                    title="Matrice de corrélation des variables numériques",
                ),
                use_container_width=True,
            )
        else:
            st.warning("La matrice de corrélation n'est pas disponible : nombre insuffisant de variables numériques.")

        chi2_results = chi2_categorical_tests(filtered)
        st.caption("Les tests du Khi 2 ci-dessous évaluent l'association entre chaque variable catégorielle et le flag de défaut 12 mois.")
        st.dataframe(display_rates(chi2_results), use_container_width=True, hide_index=True)

        st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)

    with tabs[2]:
        render_stability_rds_section(filtered, thresholds, periods)
        st.markdown("### RDS robuste N vs N-1")
        st.markdown(
            "Cette analyse compare les distributions de ratings, master scale, scores et buckets de PD entre la période de référence et la période courante."
        )
        if rds_summary.empty:
            st.warning("RDS robuste non calculable : sélectionnez au moins deux périodes.")
        else:
            st.dataframe(display_rates(rds_summary), use_container_width=True, hide_index=True)
            st.dataframe(display_rates(rds_drivers.head(20)), use_container_width=True, hide_index=True)

        st.markdown("### Matrice de migration")
        st.markdown("Objectif : mesurer les passages de grades entre N-1 et N, puis qualifier stabilité, upgrades, downgrades et migrations de plusieurs crans.")
        if migration_matrices["percentage_matrix"].empty:
            st.warning("Matrice de migration non disponible : historique de rating insuffisant.")
        else:
            st.plotly_chart(
                px.imshow(
                    migration_matrices["percentage_matrix"],
                    text_auto=".1%",
                    color_continuous_scale=AURIA_CONTINUOUS_SCALE,
                    title="Matrice de migration rating - pourcentage ligne",
                ),
                use_container_width=True,
            )
        st.dataframe(display_rates(migration_summary), use_container_width=True, hide_index=True)
        st.dataframe(display_rates(notch_distribution), use_container_width=True, hide_index=True)
        st.dataframe(display_rates(migration_patterns), use_container_width=True, hide_index=True)

        st.markdown("### Diagnostic des changements de population")
        st.markdown("Objectif : expliquer si les résultats viennent du modèle, du mix de population, des migrations, des nouveaux entrants ou des sorties.")
        st.dataframe(display_rates(new_exited_summary), use_container_width=True, hide_index=True)
        if population_diagnostic:
            st.info(population_diagnostic["comment"])
            st.metric("Statut population shift", population_diagnostic["status"])
            st.dataframe(display_rates(population_diagnostic["drivers"]), use_container_width=True, hide_index=True)
        else:
            st.warning("Diagnostic population shift non calculable avec une seule période.")

    with tabs[3]:
        st.subheader("Calibration PD")
        st.markdown(
            "Cette section compare le risque observe au risque estime par le modele, puis qualifie "
            "la significativite statistique des ecarts sur le perimetre filtre."
        )
        level_frame = aggregations.get(aggregation_level, aggregations["portfolio"])

        st.markdown("### Mesure du niveau de risque observe et estime")
        st.markdown(
            "Objectif : poser les indicateurs de niveau de risque avant toute conclusion statistique. "
            "Les chiffres ci-dessous sont recalcules apres application des filtres utilisateur."
        )
        risk_cols = st.columns(3)
        risk_cols[0].metric("ODR", format_percent(global_metrics["odr"]))
        risk_cols[1].metric("PD moyenne estimee", format_percent(global_metrics["pd_mean"]))
        risk_cols[2].metric("PD moyenne ponderee EAD", format_percent(global_metrics["ead_weighted_pd"]))
        st.caption("Detail selon le niveau d'agregation selectionne dans les filtres.")
        level_columns = [
            column
            for column in [
                "perimeter",
                "aggregation_level",
                "observations",
                "observed_defaults",
                "odr",
                "pd_mean",
                "ead_weighted_pd",
                "ead_total",
                "represented_grades",
            ]
            if column in level_frame.columns
        ]
        st.dataframe(display_rates(level_frame[level_columns]), use_container_width=True, hide_index=True)

        st.markdown("### Comparaison defauts attendus vs defauts observes")
        st.markdown(
            "Objectif : mesurer l'ecart absolu et relatif entre les defauts observes et les defauts attendus "
            "sous l'hypothese des PD individuelles."
        )
        compare_cols = st.columns(4)
        compare_cols[0].metric("Defauts attendus", format_number(global_metrics["expected_defaults"]))
        compare_cols[1].metric("Defauts observes", f"{global_metrics['observed_defaults']:,}".replace(",", " "))
        compare_cols[2].metric("Calibration gap", format_percent(global_metrics["calibration_gap"]))
        compare_cols[3].metric("Calibration ratio", format_number(global_metrics["calibration_ratio"]))

        observed_expected = pd.DataFrame(
            [
                {"indicateur": "Defauts observes", "valeur": global_metrics["observed_defaults"]},
                {"indicateur": "Defauts attendus", "valeur": global_metrics["expected_defaults"]},
            ]
        )
        st.plotly_chart(
            px.bar(
                observed_expected,
                x="indicateur",
                y="valeur",
                text_auto=".2f",
                color="indicateur",
                color_discrete_sequence=[AURIA_COLORS["navy"], AURIA_COLORS["peach"]],
                title="Defauts observes vs attendus - perimetre filtre",
            ),
            use_container_width=True,
        )

        rating_summary = aggregations["rating_grade"].sort_values("rating_grade")
        if not rating_summary.empty:
            rating_chart = rating_summary.melt(
                id_vars="rating_grade",
                value_vars=["pd_mean", "odr"],
                var_name="indicateur",
                value_name="taux",
            )
            st.plotly_chart(
                px.bar(
                    rating_chart,
                    x="rating_grade",
                    y="taux",
                    color="indicateur",
                    barmode="group",
                    text_auto=".2%",
                    title="ODR vs PD moyenne par rating grade",
                ),
                use_container_width=True,
            )
            defaults_chart = rating_summary.melt(
                id_vars="rating_grade",
                value_vars=["observed_defaults", "expected_defaults"],
                var_name="indicateur",
                value_name="nombre_defauts",
            )
            st.plotly_chart(
                px.bar(
                    defaults_chart,
                    x="rating_grade",
                    y="nombre_defauts",
                    color="indicateur",
                    barmode="group",
                    title="Defauts observes vs attendus par rating grade",
                ),
                use_container_width=True,
            )

        st.markdown("### Significativite statistique des ecarts")
        st.markdown(
            "Objectif : verifier si l'ecart entre defauts observes et attendus peut etre attribue au hasard "
            "ou s'il constitue un signal de calibration a investiguer."
        )
        volume_cfg = thresholds["minimum_volume"]
        binomial_cfg = thresholds["calibration_tests"]["binomial"]
        confidence_level = thresholds.get("pd_backtesting", {}).get("confidence_level", 0.95)
        two_sided_test = binomial_calibration_test(
            global_metrics["observations"],
            global_metrics["observed_defaults"],
            global_metrics["pd_mean"],
            confidence_level=confidence_level,
            method=binomial_cfg.get("confidence_interval_method", "wilson"),
            test_type="two_sided",
            min_observations=volume_cfg["min_observations"],
            min_defaults=volume_cfg["min_defaults"],
        )
        high_test = binomial_calibration_test(
            global_metrics["observations"],
            global_metrics["observed_defaults"],
            global_metrics["pd_mean"],
            confidence_level=confidence_level,
            method=binomial_cfg.get("confidence_interval_method", "wilson"),
            test_type="one_sided_high",
            min_observations=volume_cfg["min_observations"],
            min_defaults=volume_cfg["min_defaults"],
        )
        hl_test = run_hosmer_lemeshow_test(
            filtered,
            n_buckets=10,
            min_observations=volume_cfg["min_observations"],
            min_defaults=volume_cfg["min_defaults"],
        )
        significance = pd.DataFrame(
            [
                {
                    "test": "Binomial bilateral",
                    "objectif": "Tester tout ecart significatif entre defauts observes et PD moyenne.",
                    "p_value": two_sided_test["p_value"],
                    "ci_lower": two_sided_test["ci_lower"],
                    "ci_upper": two_sided_test["ci_upper"],
                    "status": assign_status(two_sided_test["p_value"], two_sided_test["test_interpretable"], thresholds),
                    "interpretabilite": "Interpretable" if two_sided_test["test_interpretable"] else "Non interpretable",
                },
                {
                    "test": "Binomial unilateral haut",
                    "objectif": "Detecter prioritairement une sous-estimation du risque.",
                    "p_value": high_test["p_value"],
                    "ci_lower": high_test["ci_lower"],
                    "ci_upper": high_test["ci_upper"],
                    "status": assign_status(high_test["p_value"], high_test["test_interpretable"], thresholds),
                    "interpretabilite": "Interpretable" if high_test["test_interpretable"] else "Non interpretable",
                },
                {
                    "test": "Hosmer-Lemeshow par buckets",
                    "objectif": "Tester la calibration par groupes ordonnes de risque.",
                    "p_value": hl_test["p_value"],
                    "ci_lower": pd.NA,
                    "ci_upper": pd.NA,
                    "status": assign_status(hl_test["p_value"], hl_test["test_interpretable"], thresholds),
                    "interpretabilite": "Interpretable" if hl_test["test_interpretable"] else "Non interpretable",
                },
            ]
        )
        st.dataframe(display_rates(significance), use_container_width=True, hide_index=True)
        if two_sided_test["test_interpretable"]:
            st.info(
                "Intervalle de confiance du taux de defaut observe "
                f"({binomial_cfg.get('confidence_interval_method', 'wilson')}, {format_percent(confidence_level)}) : "
                f"[{format_percent(two_sided_test['ci_lower'])} ; {format_percent(two_sided_test['ci_upper'])}]"
            )
        else:
            st.warning("Les tests statistiques sont non interpretables sur ce perimetre : volume ou defauts insuffisants.")
        if hl_test["test_interpretable"]:
            st.dataframe(display_rates(hl_test["hl_buckets"]), use_container_width=True, hide_index=True)
            hl_chart = hl_test["hl_buckets"].melt(
                id_vars="bucket",
                value_vars=["observed_defaults", "expected_defaults"],
                var_name="indicateur",
                value_name="nombre_defauts",
            )
            st.plotly_chart(
                px.bar(
                    hl_chart,
                    x="bucket",
                    y="nombre_defauts",
                    color="indicateur",
                    barmode="group",
                    title="Hosmer-Lemeshow - defauts observes vs attendus par bucket de PD",
                ),
                use_container_width=True,
            )

        st.markdown("### Calibration temporelle et vintage")
        st.markdown(
            "Objectif : verifier si la calibration reste stable selon la periode d'observation, le millesime "
            "d'entree en portefeuille et les faibles volumes annualises, notamment Corporate."
        )
        temporal_tests = pd.DataFrame(
            [
                ("Backtesting par annee d'observation", "Verifier la calibration selon la periode d'observation", "Annee, portefeuille, segment", "ODR vs PD par annee", "Recommande MVP+"),
                ("Backtesting par vintage", "Verifier la calibration selon l'annee d'octroi ou d'entree en portefeuille", "Millesime, segment", "ODR vs PD par vintage", "Recommande MVP+"),
                ("Comparaison N vs N-1", "Identifier une degradation recente de calibration", "Annee, segment, modele", "Evolution du calibration gap / ratio", "Optionnel MVP+"),
                ("Analyse multi-periodes corporate", "Pallier les faibles volumes annuels", "Corporate, segment", "Resultats agreges sur plusieurs annees", "Recommande MVP+"),
            ],
            columns=["test_indicateur", "objectif", "niveau_analyse", "sortie_attendue", "statut_mvp"],
        )
        st.dataframe(temporal_tests, use_container_width=True, hide_index=True)
        if len(periods) < 2:
            st.warning("Le suivi temporel n'est pas pertinent sur une seule periode selectionnee.")
        else:
            yearly = aggregate_pd_metrics(filtered, ["observation_year", "portfolio"], "observation_year_portfolio")
            yearly_chart = yearly.melt(
                id_vars=["observation_year", "portfolio"],
                value_vars=["odr", "pd_mean"],
                var_name="indicateur",
                value_name="taux",
            )
            st.plotly_chart(
                px.line(
                    yearly_chart,
                    x="observation_year",
                    y="taux",
                    color="portfolio",
                    line_dash="indicateur",
                    markers=True,
                    title="Backtesting par annee d'observation - ODR vs PD",
                ),
                use_container_width=True,
            )
            st.dataframe(display_rates(yearly), use_container_width=True, hide_index=True)

            vintage_frame = filtered.copy()
            if "origination_date" in vintage_frame.columns:
                vintage_frame["origination_year"] = pd.to_datetime(vintage_frame["origination_date"], errors="coerce").dt.year.astype("Int64")
                vintage = aggregate_pd_metrics(vintage_frame.dropna(subset=["origination_year"]), ["origination_year", "portfolio"], "origination_year_portfolio")
                if not vintage.empty:
                    vintage_chart = vintage.melt(
                        id_vars=["origination_year", "portfolio"],
                        value_vars=["odr", "pd_mean"],
                        var_name="indicateur",
                        value_name="taux",
                    )
                    st.plotly_chart(
                        px.line(
                            vintage_chart,
                            x="origination_year",
                            y="taux",
                            color="portfolio",
                            line_dash="indicateur",
                            markers=True,
                            title="Backtesting par vintage - ODR vs PD",
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("La colonne origination_date est presente mais non renseignee : le backtesting par vintage n'est pas disponible sur ce fichier.")
            else:
                st.info("La colonne origination_date est absente : le backtesting par vintage n'est pas disponible sur ce fichier.")

            n_minus_one = yearly.sort_values(["portfolio", "observation_year"]).copy()
            n_minus_one["calibration_gap_n_1"] = n_minus_one.groupby("portfolio")["calibration_gap"].shift(1)
            n_minus_one["variation_gap_vs_n_1"] = n_minus_one["calibration_gap"] - n_minus_one["calibration_gap_n_1"]
            st.markdown("#### Comparaison N vs N-1")
            st.dataframe(display_rates(n_minus_one), use_container_width=True, hide_index=True)

        corporate = filtered[filtered["portfolio"] == "Corporate"] if "portfolio" in filtered.columns else pd.DataFrame()
        if not corporate.empty:
            st.markdown("#### Analyse multi-periodes Corporate")
            st.caption("Aggregation multi-annees pour limiter la surinterpretation des faibles volumes annuels.")
            st.dataframe(display_rates(aggregate_pd_metrics(corporate, ["segment"], "corporate_segment_multi_period")), use_container_width=True, hide_index=True)

        st.markdown("### Philosophie de rating")
        st.markdown(
            "Comment interpreter les ecarts de calibration au regard de la philosophie du modele et du contexte economique  "
            "Cette lecture est volontairement qualitative dans le MVP."
        )
        philosophy_tests = pd.DataFrame(
            [
                ("Analyse PIT / TTC simplifiee", "Interpreter les ecarts selon la philosophie de rating", "Modele, portefeuille", "Commentaire methodologique", "Optionnel MVP+"),
                ("Commentaire sur cyclicite", "Relier les ecarts a un contexte economique favorable ou defavorable", "Annee, portfolio", "Commentaire automatique", "Optionnel MVP+"),
                ("Analyse des marges de prudence", "Identifier une sur-calibration potentiellement volontaire", "Modele, segment", "Commentaire qualitatif", "Optionnel MVP+"),
                ("Lecture low-default portfolio", "Eviter la surinterpretation des ecarts corporate", "Corporate, segment", "Statut gris / prudence", "Inclus via faibles volumes"),
            ],
            columns=["test_indicateur", "objectif", "niveau_analyse", "sortie_attendue", "statut_mvp"],
        )
        st.dataframe(philosophy_tests, use_container_width=True, hide_index=True)
        if global_metrics["calibration_gap"] > 0:
            st.info("Lecture validation : l'ODR depasse la PD moyenne sur le perimetre filtre. Le signal peut traduire une sous-estimation du risque, a confirmer par segment, grade et periode.")
        elif global_metrics["calibration_gap"] < 0:
            st.info("Lecture validation : la PD moyenne depasse l'ODR sur le perimetre filtre. Le signal peut traduire une marge de prudence, une phase economique favorable ou une philosophie plus TTC.")
        else:
            st.info("Lecture validation : aucun ecart moyen n'est visible au niveau global du perimetre filtre.")

        st.markdown("### Lecture PIT / TTC / Hybrid")
        st.markdown(
            "Cette sous-partie explicite la philosophie du système de notation sur des PD 12 mois : PIT, TTC ou Hybrid. "
            "Elle sert à contextualiser les écarts de calibration, sans créer de module IFRS 9 ni de PD lifetime."
        )
        st.info(philosophy_comment)
        if philosophy_pd_summary.empty:
            st.warning("La philosophie du modèle n'est pas disponible sur le périmètre filtré.")
        else:
            st.dataframe(display_rates(philosophy_pd_summary), use_container_width=True, hide_index=True)
        if not philosophy_behaviour.empty:
            st.plotly_chart(
                px.line(
                    philosophy_behaviour,
                    x="observation_year",
                    y="pd_mean",
                    color="model_philosophy",
                    markers=True,
                    labels={"observation_year": "Période", "pd_mean": "PD moyenne", "model_philosophy": "Philosophie"},
                    title="Évolution de la PD moyenne par philosophie",
                ),
                use_container_width=True,
            )
            st.dataframe(display_rates(philosophy_volatility), use_container_width=True, hide_index=True)
        if not philosophy_rds.empty:
            st.markdown("#### PSI/RDS par philosophie")
            st.dataframe(display_rates(philosophy_rds), use_container_width=True, hide_index=True)

        st.markdown("### Floors et MoC")
        st.markdown("Cette sous-partie présente les couches de PD : brute, calibrée, floored et réglementaire.")
        st.markdown("#### Composantes PD")
        st.dataframe(display_rates(pd_components), use_container_width=True, hide_index=True)
        st.dataframe(display_rates(pd_layer_comparison), use_container_width=True, hide_index=True)
        floor_cols = st.columns(2)
        floor_cols[0].dataframe(display_rates(pd_floor_impact), use_container_width=True, hide_index=True)
        floor_cols[1].dataframe(display_rates(moc_impact), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("Discrimination")
        if not ldp_segments.empty and ldp_segments["status"].isin(["orange", "grey", "red"]).any():
            st.warning(
                "Lecture prudente : au moins un segment est identifié comme low-default ou fragile. "
                "Les indicateurs AUC, Gini et KS doivent être interprétés avec les limites de volume."
            )
        if population_diagnostic and population_diagnostic.get("status") in {"orange", "red"}:
            st.warning("Une dérive de population est détectée ; la discrimination peut refléter un changement de mix autant qu'un changement de performance modèle.")
        st.markdown("### Mesure globale du pouvoir discriminant")
        st.markdown(
            "Objectif : mesurer la capacité du score ou de la PD à séparer les défauts des non-défauts, "
            "puis vérifier si ce pouvoir discriminant reste robuste par sous-population."
        )
        disc_status = build_discrimination_alerts(filtered, thresholds).iloc[0]
        cols = st.columns(4)
        cols[0].metric("AUC", format_percent(disc_status["auc"]))
        cols[1].metric("Gini", format_percent(disc_status["gini"]))
        cols[2].metric("KS", format_percent(disc_status["ks"]))
        cols[3].metric("Statut", disc_status["status"])
        st.info(disc_status["comment"])

        selected_disc_view = st.selectbox(
            "Indicateur à visualiser",
            ["Courbe ROC", "AUC", "Gini"],
            key="disc_global_view",
        )
        if discrimination_global["is_interpretable"]:
            if selected_disc_view == "Courbe ROC":
                roc = discrimination_global["roc_curve"]
                fig_roc = px.line(
                    roc,
                    x="fpr",
                    y="tpr",
                    title="Courbe ROC dynamique",
                    labels={"fpr": "False positive rate", "tpr": "True positive rate"},
                )
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Reference"))
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.plotly_chart(
                    px.bar(
                        pd.DataFrame(
                            [{"indicateur": selected_disc_view, "valeur": disc_status[selected_disc_view.lower()]}]
                        ),
                        x="indicateur",
                        y="valeur",
                        text_auto=".2%",
                        color_discrete_sequence=[AURIA_COLORS["navy"]],
                        range_y=[0, 1],
                    ),
                    use_container_width=True,
                )
        else:
            st.warning("Courbe ROC non disponible : défauts, non-défauts ou volume insuffisants.")

        st.markdown("#### AUC / Gini par sous-catégorie")
        auc_gini_sub = pd.concat(
            [
                discrimination_portfolio.assign(niveau="portfolio"),
                discrimination_segment.assign(niveau="segment"),
            ],
            ignore_index=True,
        )
        st.dataframe(
            display_rates(auc_gini_sub[["niveau", "perimeter", "observations", "defaults", "auc", "gini", "status", "interpretabilite"]]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### KS statistic")
        st.markdown(
            "Le KS mesure l'écart maximal entre la distribution cumulée des défauts et celle des non-défauts. "
            "Plus il est élevé, plus la séparation entre bons et mauvais risques est nette."
        )
        st.dataframe(
            display_rates(auc_gini_sub[["niveau", "perimeter", "observations", "defaults", "ks", "status", "interpretabilite"]]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Visualisation de la concentration du risque")
        st.markdown(
            "Objectif : vérifier si les défauts sont concentrés dans les classes les plus risquées et si le bad rate progresse par bucket."
        )
        if discrimination_global["is_interpretable"]:
            cap = discrimination_global["cap_curve"]
            fig_cap = px.line(
                cap,
                x="population_share",
                y="default_share",
                title="Courbe CAP dynamique",
                labels={"population_share": "Part cumulée de population", "default_share": "Part cumulée des défauts"},
            )
            fig_cap.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Reference"))
            st.plotly_chart(fig_cap, use_container_width=True)

        decile_variable = st.selectbox(
            "Variable de scoring pour les déciles",
            [column for column in ["pd_estimate", "score"] if column in filtered.columns],
            key="disc_decile_variable",
        )
        deciles = bad_rate_by_decile(filtered, decile_variable)
        st.markdown("#### Bad rate par décile de score")
        st.plotly_chart(
            px.bar(
                deciles,
                x="decile",
                y="bad_rate",
                text_auto=".2%",
                labels={"decile": "Décile de risque", "bad_rate": "Bad rate"},
                color_discrete_sequence=[AURIA_COLORS["navy"]],
            ),
            use_container_width=True,
        )
        st.dataframe(display_rates(deciles), use_container_width=True, hide_index=True)

        st.markdown("### Ordonnancement du risque")
        st.markdown(
            "Objectif : contrôler que les grades ou buckets de score ordonnent correctement le risque, sans inversion majeure entre classes adjacentes."
        )
        st.markdown("#### Ordonnancement des grades")
        st.dataframe(display_rates(monotonicity_results), use_container_width=True, hide_index=True)
        st.info(monotonicity_alert["comment"])
        violation_count = int(monotonicity_results["violation_monotonicity"].sum())
        st.metric("Nombre de violations de monotonie", violation_count)

        st.markdown("#### Bad rate par décile de score ou PD")
        st.plotly_chart(
            px.line(
                deciles,
                x="decile",
                y="bad_rate",
                markers=True,
                labels={"decile": "Décile de risque", "bad_rate": "Bad rate"},
                color_discrete_sequence=[AURIA_COLORS["peach"]],
            ),
            use_container_width=True,
        )

        st.markdown("### Suivi temporel de la discrimination")
        st.markdown(
            "Objectif : suivre l'évolution de l'AUC, du Gini et du KS dans le temps afin d'identifier une dégradation progressive du pouvoir discriminant."
        )
        if len(periods) < 2:
            st.warning(
                "Le suivi temporel de la discrimination n'est pas pertinent sur une seule période. "
                "Sélectionnez au moins deux années dans le filtre Période pour afficher l'évolution AUC, Gini et KS."
            )
        else:
            temporal_disc = discrimination_over_time(filtered, thresholds)
            if temporal_disc["is_interpretable"].any():
                temporal_chart = temporal_disc.melt(
                    id_vars=["observation_year"],
                    value_vars=["auc", "gini", "ks"],
                    var_name="indicateur",
                    value_name="valeur",
                )
                st.plotly_chart(
                    px.line(
                        temporal_chart,
                        x="observation_year",
                        y="valeur",
                        color="indicateur",
                        markers=True,
                        labels={"observation_year": "Période", "valeur": "Valeur"},
                    ),
                    use_container_width=True,
                )
            else:
                st.warning("Suivi temporel non interprétable : volumes, défauts ou non-défauts insuffisants par période.")
            st.dataframe(display_rates(temporal_disc), use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader("Alertes")
        st.markdown(
            "Cette page consolide les traffic lights selon la même logique que les onglets d'analyse. "
            "Elle sert de vue de pilotage : un test, un objectif, un indicateur et un statut."
        )
        alert_frames = [
            calibration_alerts.assign(test_family="Calibration"),
            discrimination_portfolio.assign(test_family="Discrimination"),
            stability_portfolio.assign(test_family="Stabilité RDS") if not stability_portfolio.empty else pd.DataFrame(),
            monotonicity_alerts,
        ]
        all_alerts = pd.concat([frame for frame in alert_frames if not frame.empty], ignore_index=True)
        alert_synthesis = build_alert_synthesis(
            filtered,
            thresholds,
            dq_summary,
            calibration_alerts,
            discrimination_portfolio,
            stability_global,
            stability_segment,
            monotonicity_alert,
            periods,
        )
        executive_traffic = alert_synthesis.groupby(["theme", "status"], as_index=False).size()
        st.plotly_chart(
            px.bar(
                executive_traffic,
                x="theme",
                y="size",
                color="status",
                color_discrete_map=STATUS_COLORS,
                labels={"theme": "Thème", "size": "Nombre de tests", "status": "Statut"},
                title="Synthèse traffic light par thème",
            ),
            use_container_width=True,
        )
        for theme in [
            "Data quality",
            "Stabilité Rating / Score Distribution (RDS)",
            "Calibration PD",
            "Discrimination",
        ]:
            render_alert_synthesis_section(alert_synthesis, theme)

        st.subheader("Findings de validation")
        st.markdown(
            "Les findings transforment les alertes orange, rouges ou grises en constats de validation priorisables."
        )
        st.dataframe(findings, use_container_width=True, hide_index=True)

    with tabs[6]:
        st.subheader("Rapport & export")
        narrative = generate_demo_narrative(scenario_description, expected_observation, calibration_alerts, discrimination_portfolio, stability_portfolio)
        st.text_area("Narrative demo", narrative, height=260)
        export_summary = display_rates(summary)
        st.download_button("Exporter synthese CSV", export_summary.to_csv(index=False).encode("utf-8"), "pd_backtesting_summary.csv", "text/csv")
        st.download_button("Exporter alertes CSV", all_alerts.to_csv(index=False).encode("utf-8"), "pd_backtesting_alerts.csv", "text/csv")
        st.download_button("Exporter findings CSV", findings.to_csv(index=False).encode("utf-8"), "pd_backtesting_findings.csv", "text/csv")
        try:
            html = (
                "<h1>PD Backtesting Report</h1>"
                + "<h2>Seuils et limites</h2><p>" + thresholds_limitations_text(threshold_profile) + "</p>"
                + "<h2>Cartographie des tests</h2>" + test_mapping.to_html(index=False)
                + "<h2>Synthese</h2>" + export_summary.to_html(index=False)
                + "<h2>Analyse population eligible</h2>" + display_rates(eligibility_summary).to_html(index=False)
                + "<h2>RDS</h2>" + display_rates(rds_summary).to_html(index=False)
                + "<h2>Migration</h2>" + display_rates(migration_summary).to_html(index=False)
                + "<h2>Population shift</h2>" + (display_rates(population_diagnostic.get("drivers", pd.DataFrame())).to_html(index=False) if population_diagnostic else "<p>Non calculable</p>")
                + "<h2>PIT TTC Analysis</h2>"
                + "<p>PIT/TTC/Hybrid est une cle d'interpretation des PD 12 mois, pas un test autonome de conformite.</p>"
                + "<p>" + philosophy_comment + "</p>"
                + display_rates(philosophy_pd_summary).to_html(index=False)
                + "<h3>Volatilite PD par philosophie</h3>" + display_rates(philosophy_volatility).to_html(index=False)
                + "<h3>RDS par philosophie</h3>" + display_rates(philosophy_rds).to_html(index=False)
                + "<h2>Floors et MoC</h2>" + display_rates(pd_floor_impact).to_html(index=False) + display_rates(moc_impact).to_html(index=False)
                + "<h2>Low Default Portfolio</h2>" + display_rates(ldp_segments).to_html(index=False)
                + "<h2>Exclusions</h2>" + display_rates(material_exclusions).to_html(index=False)
                + "<h2>Monotonie</h2>" + display_rates(monotonicity_results).to_html(index=False)
                + "<h2>Findings</h2>" + findings.to_html(index=False)
            )
            st.download_button("Exporter rapport HTML", html.encode("utf-8"), "pd_backtesting_report.html", "text/html")
        except Exception as exc:
            st.error(f"Export HTML impossible : {exc}")
        try:
            from io import BytesIO

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                export_summary.to_excel(writer, sheet_name="summary", index=False)
                test_mapping.to_excel(writer, sheet_name="test_mapping", index=False)
                display_rates(eligibility_summary).to_excel(writer, sheet_name="Regulatory Eligibility", index=False)
                display_rates(rds_summary).to_excel(writer, sheet_name="RDS Stability", index=False)
                display_rates(migration_summary).to_excel(writer, sheet_name="Migration Matrix", index=False)
                display_rates(population_diagnostic.get("drivers", pd.DataFrame()) if population_diagnostic else pd.DataFrame()).to_excel(writer, sheet_name="Population Shift", index=False)
                display_rates(philosophy_pd_summary).to_excel(writer, sheet_name="PIT TTC Analysis", index=False)
                display_rates(philosophy_volatility).to_excel(writer, sheet_name="PIT TTC Volatility", index=False)
                display_rates(philosophy_rds).to_excel(writer, sheet_name="PIT TTC RDS", index=False)
                pd.DataFrame(
                    [
                        {
                            "commentaire": philosophy_comment,
                            "rappel": "PIT/TTC/Hybrid est une cle d'interpretation des PD 12 mois, pas un test autonome de conformite.",
                        }
                    ]
                ).to_excel(writer, sheet_name="PIT TTC Comments", index=False)
                display_rates(pd_layer_comparison).to_excel(writer, sheet_name="PD Floors and MoC", index=False)
                display_rates(ldp_segments).to_excel(writer, sheet_name="Low Default Portfolio", index=False)
                display_rates(material_exclusions).to_excel(writer, sheet_name="Exclusion Analysis", index=False)
                display_rates(monotonicity_results).to_excel(writer, sheet_name="monotonicity", index=False)
                findings.to_excel(writer, sheet_name="Updated Findings", index=False)
                pd.DataFrame([{"rappel": thresholds_limitations_text(threshold_profile)}]).to_excel(writer, sheet_name="thresholds_limits", index=False)
            st.download_button(
                "Exporter rapport Excel",
                buffer.getvalue(),
                "pd_backtesting_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as exc:
            st.error(f"Export Excel impossible : {exc}")

    with tabs[7]:
        st.subheader("Methodologie")
        st.subheader("Methodologie en une phrase")
        render_methodology_text()
        st.subheader("Seuils et limites")
        st.warning(thresholds_limitations_text(threshold_profile))
        st.markdown(
            """
            **ODR** : taux de defaut observe, calcule comme defauts observes / observations.

            **PD moyenne** : moyenne arithmetique des PD individuelles ; les defauts attendus correspondent a la somme des PD.

            **Calibration gap** : ODR - PD moyenne. **Calibration ratio** : ODR / PD moyenne.

            **Test binomial** : compare le nombre de defauts observes au nombre attendu sous l'hypothese PD moyenne. Le test unilateral haut detecte une sous-estimation du risque.

            **AUC, Gini, KS** : mesurent le pouvoir discriminant du score ou de la PD. Gini = 2 x AUC - 1.

            **PSI** : mesure la derive de distribution entre une periode de reference et une periode courante.

            **Traffic light** : rouge prioritaire, orange a analyser, vert sans signal majeur, gris non interpretable.

            **Faibles volumes** : les tests sont classes gris lorsque le nombre d'observations, de defauts ou de non-defauts est insuffisant.
            """
        )

    render_auria_contact_footer()


if __name__ == "__main__":
    main()

