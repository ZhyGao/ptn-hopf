"""Input validation, reproducible parameter sweeps, checkpoints and provenance."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

from . import __version__
from .model import repeat_summary

ROOT = Path(__file__).resolve().parents[1]
_WORKER = {}


def grid(start, end, step):
    if step <= 0 or end < start:
        raise ValueError("Grid step must be positive and end >= start.")
    count = (end - start) / step
    if not np.isclose(count, round(count), atol=1e-8, rtol=0):
        raise ValueError("Grid endpoints must be divisible by the step.")
    return np.round(np.linspace(start, end, int(round(count)) + 1), 8).tolist()


def load_config(path):
    config = json.loads(Path(path).read_text())
    for prefix in ("g", "a"):
        grid(config[f"{prefix}_start"], config[f"{prefix}_end"], config[f"{prefix}_step"])
    for key in ("fit_repeats", "stimulation_repeats"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["noise_strength"] < 0 or config["baseline_a"] != 0:
        raise ValueError("Noise must be nonnegative; this example uses baseline a=0.")
    if config["g_start"] < 0:
        raise ValueError("G must be nonnegative.")
    if not any(abs(a) < 1e-10 for a in grid(config["a_start"], config["a_end"], config["a_step"])):
        raise ValueError("The stimulation grid must include a=0 for the baseline.")
    if config["a_start"] == config["a_end"] == 0:
        raise ValueError("The stimulation grid must include a nonzero value.")
    return config


def load_inputs(data_dir):
    data_dir = Path(data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text())
    for name, expected in metadata["sha256"].items():
        if Path(name).name != name:
            raise ValueError("Input manifest names must be plain filenames.")
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input checksum failed: {name}")
    with np.load(data_dir / "ptn_demo.npz", allow_pickle=False) as bundle:
        data = {key: bundle[key] for key in bundle.files}
    shapes = {"scaled_sc": (116, 116), "omega": (116,), "state_centers": (2, 116),
              "empirical_fo": (2,), "empirical_tpm": (2, 2),
              "hlc_mean_fo": (2,), "hlc_mean_tpm": (2, 2),
              "n_timepoints": (), "seed_base": ()}
    for key, shape in shapes.items():
        if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key}: expected finite values and shape {shape}.")
    sc = data["scaled_sc"]
    if np.any(sc < 0) or not np.allclose(sc, sc.T) or not np.allclose(np.diag(sc), 0):
        raise ValueError("SC must be nonnegative, symmetric and have a zero diagonal.")
    if not np.isclose(sc.max(), 0.2):
        raise ValueError("Expected SC already scaled to maximum 0.2. Do not scale it twice.")
    for key in ("empirical_fo", "hlc_mean_fo", "empirical_tpm", "hlc_mean_tpm"):
        p = data[key]
        if np.any(p < 0) or not np.allclose(p.sum(axis=-1), 1):
            raise ValueError(f"Invalid probability distribution: {key}")
    if int(data["n_timepoints"]) < 116 or float(data["n_timepoints"]) != int(data["n_timepoints"]):
        raise ValueError("This example requires an integer n_timepoints >= 116.")
    if np.any(data["omega"] <= 0) or int(data["seed_base"]) < 0:
        raise ValueError("Frequencies must be positive and seed base nonnegative.")
    with (data_dir / "bilateral_targets.csv").open(newline="") as f:
        targets = list(csv.DictReader(f))
    if len(targets) != 45:
        raise ValueError("Expected 45 bilateral AAL90 targets.")
    with (data_dir / "aal116_regions.csv").open(newline="") as f:
        regions = list(csv.DictReader(f))
    if len(regions) != 116:
        raise ValueError("Expected 116 ordered AAL labels.")
    for i, target in enumerate(targets):
        if (int(target["target_id"]), int(target["left_1based"]), int(target["right_1based"])) != (i, 2*i+1, 2*i+2):
            raise ValueError("Unexpected bilateral target indices or order.")
        for index, side in ((2*i, "L"), (2*i+1, "R")):
            if regions[index]["region"] != f"{target['target']}_{side}":
                raise ValueError("Target names disagree with the AAL region table.")
    return data, targets, metadata


def _initialize(data, config):
    # Prevent each worker from launching its own large BLAS thread pool.
    _WORKER.update(data=data, config=config, limit=threadpool_limits(limits=1))


def _task(task):
    kind, g, target_id, target_name, left, right, a_value = task
    data, config = _WORKER["data"], _WORKER["config"]
    a = np.zeros(116)
    if kind == "stim":
        a[[left, right]] = a_value
    repeats = config["fit_repeats"] if kind == "fit" else config["stimulation_repeats"]
    summary = repeat_summary(data, g, a, repeats, config["noise_strength"])
    if kind == "stim":
        summary = {"target_id": target_id, "target": target_name, "a": a_value, **summary}
    return summary


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [{key: (value if key in {"target", "condition"} else float(value))
                 for key, value in row.items()} for row in csv.DictReader(f)]


def write_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sweep(pool, tasks, path, rows, key, label):
    completed = {key(row) for row in rows}
    if len(completed) != len(rows):
        raise ValueError(f"Duplicate checkpoint rows in {path}")
    remaining = [task for task in tasks if (task[1] if label == "fit" else (task[2], task[6])) not in completed]
    print(f"{label}: {len(rows)} cached, {len(remaining)} conditions to simulate", flush=True)
    started = time.monotonic()
    for index, result in enumerate(pool.map(_task, remaining, chunksize=1), 1):
        rows.append(result)
        # Atomic checkpoint after each condition; interruption can be resumed.
        write_csv(path, sorted(rows, key=key))
        if index == 1 or index % max(1, len(remaining)//20) == 0 or index == len(remaining):
            print(f"  {label}: {index}/{len(remaining)} new conditions; {time.monotonic()-started:.1f}s", flush=True)
    return sorted(rows, key=key)


def run(config, data_dir, output, workers, resume=False):
    import scipy
    import numba
    data, targets, metadata = load_inputs(data_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(b"".join((ROOT / "ptn_hopf" / name).read_bytes()
                                         for name in ("model.py", "pipeline.py"))).hexdigest()
    signature_parts = {"config": config, "input_hashes": metadata["sha256"], "model_code_sha256": source_hash,
                       "versions": {"python":platform.python_version(), "numpy":np.__version__,
                                    "scipy":scipy.__version__, "numba":numba.__version__}}
    signature = hashlib.sha256(json.dumps(signature_parts, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "run_metadata.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if not resume:
            raise ValueError(f"Output already contains a run. Use --resume or a new --output: {output}")
        if prior["signature"] != signature:
            raise ValueError("Resume rejected: inputs, configuration, model code or runtime versions changed.")
    elif any(output.iterdir()):
        raise ValueError("Output directory is nonempty without run metadata; choose another directory.")
    run_metadata = {"signature": signature, **signature_parts, "package_version":__version__,
                    "sample_id":metadata["sample_id"], "workers":workers,
                    "started_utc":datetime.now(timezone.utc).isoformat(), "complete":False,
                    "n_hlc":metadata["hlc_fo_participants"], "platform":platform.platform()}
    manifest_path.write_text(json.dumps(run_metadata, indent=2) + "\n")
    started = time.monotonic()
    gs = grid(config["g_start"], config["g_end"], config["g_step"])
    intensities = grid(config["a_start"], config["a_end"], config["a_step"])
    fit_tasks = [("fit",g,-1,"",0,1,0.0) for g in gs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize, initargs=(data,config)) as pool:
        fit = _sweep(pool, fit_tasks, output/"fit.csv", read_csv(output/"fit.csv"),
                     lambda r: r["g"], "fit")
        if {r["g"] for r in fit} != set(gs):
            raise ValueError("Fit checkpoint does not match the requested G grid.")
        best_fit = min(fit, key=lambda r:(r["fit_mean_repeat_kl"],r["g"]))
        g = best_fit["g"]
        print(f"Selected G={g:g}; mean repeat KL={best_fit['fit_mean_repeat_kl']:.6g}", flush=True)
        stim_tasks = []
        # a=0 is target-independent. Run once, then reproduce the baseline row.
        for target in targets:
            tid = int(target["target_id"])
            for a in intensities:
                if abs(a) < 1e-10 and tid != 0:
                    continue
                stim_tasks.append(("stim",g,tid,target["target"],int(target["left_1based"])-1,
                                   int(target["right_1based"])-1,a))
        stim = _sweep(pool, stim_tasks, output/"stimulation_checkpoint.csv",
                      read_csv(output/"stimulation_checkpoint.csv"),
                      lambda r:(int(r["target_id"]),r["a"]), "stimulation")
    expected = {(task[2],task[6]) for task in stim_tasks}
    if {(int(r["target_id"]),r["a"]) for r in stim} != expected:
        raise ValueError("Stimulation checkpoint does not match the requested grid.")
    if any(not np.isclose(r["g"],g) for r in stim):
        raise ValueError("Stimulation checkpoint G does not match the fitted G.")
    baseline = next(r for r in stim if r["target_id"]==0 and abs(r["a"])<1e-10)
    for target in targets[1:]:
        stim.append({**baseline,"target_id":int(target["target_id"]),"target":target["target"]})
    stim.sort(key=lambda r:(int(r["target_id"]),r["a"]))
    write_csv(output/"stimulation.csv",stim)
    # Descriptive optimum: lowest KL to HLC, excluding the zero-stimulation condition.
    best = min((r for r in stim if abs(r["a"])>1e-10),
               key=lambda r:(r["hlc_kl_of_mean_fo"],int(r["target_id"]),r["a"]))
    comparison=[]
    for name, fo, sd in (
        ("Empirical PTN",data["empirical_fo"],np.zeros(2)),
        ("Simulated baseline",[baseline["fo_state1"],baseline["fo_state2"]],
         [baseline["fo_sd_state1"],baseline["fo_sd_state2"]]),
        ("Selected stimulation",[best["fo_state1"],best["fo_state2"]],
         [best["fo_sd_state1"],best["fo_sd_state2"]]),
        ("HLC group mean",data["hlc_mean_fo"],np.zeros(2))):
        comparison.append({"condition":name,"fo_state1":float(fo[0]),"fo_state2":float(fo[1]),
                           "fo_sd_state1":float(sd[0]),"fo_sd_state2":float(sd[1])})
    write_csv(output/"fo_comparison.csv",comparison)
    summary = {"sample_id":metadata["sample_id"], "configuration":config["name"],
               "best_g":g,"fit_mean_repeat_kl":best_fit["fit_mean_repeat_kl"],
               "fit_kl_of_mean_fo":best_fit["fit_kl_of_mean_fo"],
               "selected_target":best["target"],"selected_a":best["a"],
               "baseline_fo_state1":baseline["fo_state1"],"stimulated_fo_state1":best["fo_state1"],
               "hlc_fo_state1":float(data["hlc_mean_fo"][0]),
               "baseline_kl_to_hlc":baseline["hlc_kl_of_mean_fo"],
               "stimulated_kl_to_hlc":best["hlc_kl_of_mean_fo"],
               "fit_conditions":len(fit),"stimulation_cells":len(stim),
               "simulated_trajectories":len(gs)*config["fit_repeats"]+len(stim_tasks)*config["stimulation_repeats"],
               "selection_rule":"Minimum symmetric KL(mean simulated FO, HLC mean FO) over nonzero stimulation; descriptive, same repeats used for selection and display."}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    run_metadata.update(complete=True,elapsed_seconds=time.monotonic()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(run_metadata,indent=2)+"\n")
    from .plotting import plot_results
    plot_results(output)
    print(json.dumps(summary,indent=2),flush=True)
    return summary
