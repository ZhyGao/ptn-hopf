# PTN Hopf model: one-participant reproducible example

[中文说明](README_zh.md)

Fit a structurally constrained whole-brain Hopf model to **one PTN participant**,
apply bilateral regional perturbations, and visualize fractional occupancy (FO)
changes relative to the **empirical HLC group mean (46 participants)**.

The bundled input is real, processed research data (about 79 KiB), not synthetic
data. Raw MRI, BOLD timeseries, clinical records and other individual participants
are not included. LEiDA state discovery has already been performed: this example
uses the supplied empirical state centers and does not rerun clustering.

This repository includes example inputs, runnable code and installation instructions.
Simulation outputs are generated locally when you run the example.

## Install

Python 3.10 or later; Python 3.12 is the tested environment. No MATLAB, GPU,
external data download, original project checkout or absolute source path is needed.

From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. To use the exact versions
tested with Python 3.12, install `requirements-tested.txt` instead.

Optionally, `python -m pip install .` installs the `ptn-hopf-demo` command and
bundled inputs/configurations. The commands below use `python run_demo.py` so
that running directly from a downloaded repository remains straightforward.

## Run

Validate the bundled inputs and their checksums:

```bash
python run_demo.py --validate-only
```

Run the full example, using four parallel processes:

```bash
python run_demo.py --workers 4
```

The same full configuration can be specified explicitly:

```bash
python run_demo.py --config configs/full.json --workers 4 --output outputs/full
```

The initial run compiles the numerical kernel with Numba. CPU time depends on
worker count and hardware. A validation run took about nine minutes using
24 processes on the validation server; a laptop or fewer workers will take longer.
`--workers 1` runs on a single worker. Seeds do not depend on worker scheduling.

| Parameter | Full configuration (default) |
|---|---|
| Fitting G | 0–0.5, step 0.005 (101 values) |
| Fit repeats per G | 200 |
| Stimulation a | −0.4–0.4, step 0.01 (81 values) |
| Stimulation repeats per condition | 50 |
| Bilateral targets | All 45 AAL90 pairs |
| Simulated network | All 116 AAL regions |

The default does not reduce grids or repeat counts. It generates **3,645
stimulation cells** and **200,250 trajectories** including model fitting.
The zero-stimulation trajectory is independent of target; it is simulated once
per repeat and reused across all target rows without reducing the repeat count.

An interrupted run can be resumed:

```bash
python run_demo.py --config configs/full.json --workers 4 --output outputs/full --resume
```

Resume requires matching inputs, configuration, numerical code and runtime
versions. Changing worker count is allowed. Use a new output directory for a new
configuration. Output directories containing existing results are never silently
overwritten.

## Generated outputs

After running the example, results are saved in `outputs/full/`. To regenerate
figures from your completed simulation tables:

```bash
python run_demo.py --plot-only --output outputs/full
```

This repository does not bundle completed simulation outputs. The following
files are created by running the code:

| Output | Meaning |
|---|---|
| `overview.png` | A: State 1 FO heatmap; B: TPM distance heatmap; C: G selection; D: FO changes |
| `stimulation_fo_heatmap.png` | Mean State 1 FO across all targets and intensities |
| `stimulation_tpm_heatmap.png` | Frobenius distance from mean simulated TPM to the HLC group-mean TPM |
| `fit.png` | G search curve and selected G |
| `fo_change.png` | Empirical PTN, simulated baseline, selected stimulation and HLC reference, grouped by state |
| `fit.csv` | Every G and its fitting metrics |
| `stimulation.csv` | All target × intensity conditions, mean FO/SD, TPM and distance metrics |
| `fo_comparison.csv` | Values underlying the FO comparison |
| `stimulation_fo_heatmap_matrix.csv` | Unclipped FO heatmap matrix |
| `stimulation_tpm_heatmap_matrix.csv` | Unclipped TPM distance heatmap matrix |
| `summary.json` | Selected G, target, intensity and key results |
| `run_metadata.json` | Configuration, versions, hashes, completion flag and runtime |
| `heatmap_display.json` | Per-metric p90 color limits and four-panel mapping |

Both heatmaps use the manuscript's p90 display cap: values above p90 use the
normal colormap endpoint color. All cells are colored. The raw matrices retain
all values at full precision. The only plotted stimulation metrics are State 1
FO and TPM Frobenius distance; fitting and optimum selection continue to use
the symmetric KL criteria described below.

Figures also have editable SVG and PDF exports. Simulation error bars are **SD
across noise repeats**, not participant-level uncertainty or SEM.

## What the model does

1. Read the already scaled individual SC, intrinsic angular frequencies,
   empirical PTN FO/TPM and fixed two-state LEiDA centers.
2. Search G; select the minimum **mean of repeat-wise symmetric KL to PTN FO**.
3. Hold the selected G fixed. Set local a at each bilateral target, leaving
   other regions at a=0, and repeat the simulation using common noise seeds.
4. Filter simulated signals (0.04–0.07 Hz), extract leading phase-coherence
   eigenvectors, and assign to empirical centers using Manhattan distance.
5. Compare **the repeat-mean FO** with HLC group-mean FO. Mark the nonzero
   stimulation cell with the smallest symmetric KL and plot its FO change.

The fitting and stimulation criteria intentionally use different aggregation
orders. See [methods](docs/methods.md) and [data dictionary](data/README.md).
The exact 2×116 empirical centroids are provided both in the input bundle and
in the human-readable `data/state_centers.csv` file.

The selected participant was chosen as a complete worked example with empirical
State 1 FO below the HLC group mean. This is a single-person model demonstration.
Selecting and displaying the minimum on the same simulated repeats is descriptive;
it is not independent validation or evidence of a clinical treatment effect.
TPM distance is retained as an auxiliary output; the optimum is selected by FO-KL.

## Repository map

```text
run_demo.py             One entry point for validation, simulation and plotting
ptn_hopf/model.py       Hopf integrator, LEiDA projection, FO/TPM and symmetric KL
ptn_hopf/pipeline.py    Sweeps, checkpoints, selection and provenance
ptn_hopf/plotting.py    Figures generated from saved tables
configs/               Full research configuration (default)
data/                  One PTN, HLC references and 2×116 empirical state centers
docs/                  Methods, reproducibility checks and external-code notes
tests/                 Regression against direct mathematical implementations
.github/workflows/     Automated input checks, regression tests and a small run
```

## Verify

```bash
python -m pip install pytest
python -m pytest -q
```

Tests compare the accelerated integrator to the direct source equations and the
rank-two LEiDA projection to a full phase-coherence eigendecomposition.

## External methods and reuse

LEiDA state discovery is external to this repository. The study authors will
provide the exact external repository/version used; no external LEiDA source code
is redistributed here. See [external methods](docs/external_methods.md).

This example does not assign a new software or research-data reuse license.
Repository owners should state their chosen terms when publishing. It also does
not assert a manuscript DOI, repository URL or author list that has not been set.
