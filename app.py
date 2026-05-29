"""Streamlit Community Cloud entrypoint."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


STREAMLIT_APP_PATH = Path(__file__).parent / "app" / "streamlit_app.py"
__path__ = [str(Path(__file__).parent / "app")]


def main() -> None:
    """Load and run the main Streamlit application."""
    spec = spec_from_file_location("pd_backtesting_streamlit_app", STREAMLIT_APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Streamlit app from {STREAMLIT_APP_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
