import argparse
from pathlib import Path

from src.data_generation.demo_scenarios import generate_demo_scenario, scenario_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PD backtesting demo scenario.")
    parser.add_argument("--scenario", choices=sorted(scenario_catalog()), default="retail_well_calibrated")
    parser.add_argument("--output", default="data/generated/demo_scenario.csv")
    parser.add_argument("--retail", type=int, default=30000)
    parser.add_argument("--corporate", type=int, default=5000)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--data-quality-level", choices=["none", "low", "medium", "high"], default="low")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_demo_scenario(
        scenario=args.scenario,
        retail_observations=args.retail,
        corporate_observations=args.corporate,
        start_year=args.start_year,
        years=args.years,
        data_quality_level=args.data_quality_level,
        random_seed=args.random_seed,
    )
    frame.to_csv(output_path, index=False)
    print(f"Generated scenario '{args.scenario}' with {len(frame):,} rows at {output_path}")


if __name__ == "__main__":
    main()
