"""
Strategy Robustness Lab — main orchestrator.

Runs the full PBO/CSCV pipeline or launches the Streamlit dashboard.
No business logic lives here — this file only wires modules together.

Usage:
    python3 main.py                    # run pipeline with default config
    python3 main.py --config path.yaml # custom config
    python3 main.py --mode dashboard   # launch Streamlit app
    python3 main.py --mode synthetic   # run on synthetic data (testing)
"""

import argparse
import logging

from src.utils.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Entry point — parse args and dispatch."""
    parser = argparse.ArgumentParser(description="Strategy Robustness Lab")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--mode", default="pipeline",
                        choices=["pipeline", "dashboard", "synthetic"],
                        help="Run mode")
    parser.add_argument("--connector", default="tsmom",
                        choices=["tsmom", "factor", "csv"],
                        help="Connector to use for trial matrix generation")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "dashboard":
        import subprocess
        subprocess.run(["streamlit", "run", "app/app.py"], check=True)
    elif args.mode == "synthetic":
        from src.pipeline import generate_synthetic
        from src.cscv import run_cscv
        from src.pbo import compute_pbo

        trial_matrix = generate_synthetic(config)
        logger.info("Synthetic matrix: %d days × %d trials", *trial_matrix.shape)
        cscv_results = run_cscv(trial_matrix, config)
        pbo_result = compute_pbo(cscv_results)
        logger.info("Synthetic PBO: %.3f", pbo_result["pbo"])
    else:
        from src.pipeline import run_pipeline
        run_pipeline(config, mode=args.connector)


if __name__ == "__main__":
    main()
