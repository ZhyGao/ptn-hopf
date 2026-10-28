"""Command-line entry point; all input/output paths can be overridden."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="One PTN: fit Hopf, stimulate 45 bilateral targets, plot FO changes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration; default: bundled configs/full.json",
    )
    parser.add_argument(
        "--data", type=Path, help="Processed input folder; default: bundled data/"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Result directory; default: outputs/<config name> in current directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel processes (default: up to 4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume checkpoints only when model, inputs, config and runtime match",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from completed CSV results; no simulation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the input bundle and configuration, then exit",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    # These limit numerical-library threads, not the user-selected process count.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    from .pipeline import ROOT, load_config, load_inputs, run

    asset_root = (
        ROOT
        if (ROOT / "data/ptn_demo.npz").exists()
        else Path(sys.prefix) / "share/ptn-hopf-demo"
    )
    config_path = args.config or asset_root / "configs/full.json"
    data_dir = args.data or asset_root / "data"
    try:
        if args.plot_only:
            if args.output is None:
                parser.error("--plot-only requires --output")
            from .plotting import plot_results

            plot_results(args.output)
            print(f"Plots saved in {args.output}")
            return
        config = load_config(config_path)
        if args.validate_only:
            _, targets, metadata = load_inputs(data_dir)
            print(
                f"Validated: {metadata['sample_id']}, 116 regions, {len(targets)} targets, HLC n={metadata['hlc_fo_participants']}, config={config['name']}"
            )
            return
        run(
            config,
            data_dir,
            args.output or Path.cwd() / "outputs" / config["name"],
            args.workers,
            args.resume,
        )
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
