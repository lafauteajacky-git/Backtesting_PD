from __future__ import annotations

import numpy as np
import pandas as pd


def _excluded(df: pd.DataFrame) -> pd.DataFrame:
    if "exclusion_flag" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df["exclusion_flag"].fillna(0).astype(int).eq(1)].copy()


def summarize_exclusions_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize exclusions by category."""
    excluded = _excluded(df)
    if excluded.empty:
        return pd.DataFrame(columns=["exclusion_category", "observations", "exclusion_rate"])
    summary = excluded.groupby("exclusion_category", dropna=False).size().reset_index(name="observations")
    summary["exclusion_rate"] = summary["observations"] / max(len(df), 1)
    return summary.sort_values("observations", ascending=False)


def summarize_exclusions_by_materiality(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize exclusions by materiality."""
    excluded = _excluded(df)
    if excluded.empty:
        return pd.DataFrame(columns=["exclusion_materiality", "observations"])
    return excluded.groupby("exclusion_materiality", dropna=False).size().reset_index(name="observations")


def calculate_exclusion_ead_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EAD impact of exclusions."""
    excluded = _excluded(df)
    ead = pd.to_numeric(df.get("ead_at_observation", pd.Series(dtype=float)), errors="coerce")
    excluded_ead = pd.to_numeric(excluded.get("ead_at_observation", pd.Series(dtype=float)), errors="coerce")
    total_ead = float(ead.sum())
    return pd.DataFrame([{"excluded_ead": float(excluded_ead.sum()), "total_ead": total_ead, "excluded_ead_rate": float(excluded_ead.sum() / total_ead) if total_ead else np.nan}])


def identify_material_exclusions(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Identify material exclusions by category and rule."""
    excluded = _excluded(df)
    if excluded.empty:
        return pd.DataFrame()
    grouped = (
        excluded.groupby(["exclusion_category", "exclusion_rule_id"], dropna=False)
        .agg(observations=("observation_id", "size"), ead_at_observation=("ead_at_observation", "sum"))
        .reset_index()
    )
    grouped["exclusion_rate"] = grouped["observations"] / max(len(df), 1)
    orange = (thresholds or {}).get("exclusions", {}).get("rate_orange", 0.05)
    red = (thresholds or {}).get("exclusions", {}).get("rate_red", 0.15)
    grouped["status"] = grouped["exclusion_rate"].map(lambda value: "red" if value >= red else "orange" if value >= orange else "green")
    return grouped.sort_values("observations", ascending=False)


def generate_exclusion_findings(df: pd.DataFrame) -> pd.DataFrame:
    """Generate validation findings for exclusions."""
    material = identify_material_exclusions(df)
    rows = []
    for idx, row in material[material["status"].isin(["orange", "red"])].iterrows():
        rows.append(
            {
                "finding_id": f"EXC-{idx + 1:03d}",
                "theme": "Exclusions",
                "perimetre": row["exclusion_category"],
                "constat": f"Exclusions {row['exclusion_category']} materielles ou significatives.",
                "niveau_de_severite": "Haute" if row["status"] == "red" else "Moyenne",
                "justification_statistique": f"taux={row['exclusion_rate']:.2%}; EAD={row['ead_at_observation']:.2f}",
                "recommandation": "Documenter les regles d'exclusion et tester la sensibilite des resultats.",
            }
        )
    return pd.DataFrame(rows)
