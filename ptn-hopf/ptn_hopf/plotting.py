"""Manuscript-style figures rebuilt from full simulated result tables."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pipeline import read_csv, write_csv

STATE_COLORS = ["#367A9D", "#D58A55"]
CONDITION_COLORS = ["#8B7AAE", "#3B82F6", "#10B981", "#EF4444"]
CONDITION_LABELS = [
    "Empirical PTN",
    "Simulated PTN",
    "Stimulated PTN",
    "HLC mean target",
]
METRICS = {
    "hlc_tpm_frobenius": ("TPM Frobenius to HLC", "Frobenius distance", "viridis_r"),
    "fo_state1": ("State 1 fractional occupancy", "State 1 FO", "viridis"),
}


def _save(fig, stem):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            f"{stem}.{extension}", dpi=300, bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)


def _matrix(rows, metric):
    targets = sorted({int(r["target_id"]): r["target"] for r in rows}.items())
    intensities = sorted({r["a"] for r in rows})
    lookup = {(int(r["target_id"]), r["a"]): r[metric] for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(targets) * len(intensities):
        raise ValueError("Heatmap table has duplicate or missing conditions.")
    matrix = np.array([[lookup[(tid, a)] for a in intensities] for tid, _ in targets])
    if not np.isfinite(matrix).all():
        raise ValueError("Heatmap values must be finite.")
    return matrix, targets, intensities


def _fit(ax, rows, summary):
    ax.plot(
        [r["g"] for r in rows],
        [r["fit_mean_repeat_kl"] for r in rows],
        color=STATE_COLORS[0],
        lw=1.6,
    )
    ax.axvline(summary["best_g"], ls="--", lw=1, color="#666666")
    ax.scatter(
        [summary["best_g"]],
        [summary["fit_mean_repeat_kl"]],
        color="#222222",
        s=25,
        zorder=4,
    )
    ax.set(
        xlabel="Global coupling G",
        ylabel="Mean repeat symmetric KL to PTN",
        title=f"PTN model fitting | G = {summary['best_g']:g}",
    )
    ax.spines[["top", "right"]].set_visible(False)


def _fo_bars(ax, rows):
    """The paper's four condition colors, grouped by state; SD over noise seeds."""
    for condition, row in enumerate(rows):
        xs = np.arange(2) + (condition - 1.5) * 0.18
        ax.bar(
            xs,
            [row["fo_state1"], row["fo_state2"]],
            0.13,
            yerr=[row["fo_sd_state1"], row["fo_sd_state2"]],
            capsize=2.5,
            color=CONDITION_COLORS[condition],
            alpha=0.68,
            edgecolor="#252525",
            linewidth=0.65,
            label=CONDITION_LABELS[condition],
            error_kw={"elinewidth": 0.8},
            zorder=3,
        )
    for state in (1, 2):
        ax.plot(
            np.array([-0.27, -0.09, 0.09]) + (state - 1),
            [r[f"fo_state{state}"] for r in rows[:3]],
            color="#555555",
            alpha=0.4,
            lw=0.8,
            marker="o",
            markersize=3,
            zorder=4,
        )
    ax.set_xticks([0, 1], ["State 1", "State 2"])
    ax.set(
        ylim=(0, 1),
        xlim=(-0.55, 1.55),
        ylabel="Fractional occupancy",
        title="State occupancy before\nand after stimulation",
    )
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)


def _heatmap(ax, rows, metric, *, index_only=False):
    """Paper p90 display with all cells colored and no outline annotations."""
    matrix, targets, intensities = _matrix(rows, metric)
    title, color_label, cmap = METRICS[metric]
    vmin, vmax = float(matrix.min()), float(np.percentile(matrix, 90))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    image = ax.imshow(
        np.minimum(matrix, vmax),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    indices = np.unique(np.linspace(0, len(intensities) - 1, 9).round().astype(int))
    ax.set_xticks(
        indices,
        [f"{intensities[i]:.2f}" for i in indices],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    names = [f"{tid+1}_" if index_only else f"{tid+1}_{name}" for tid, name in targets]
    ax.set_yticks(
        range(len(targets)),
        names,
        fontsize=8.4,
        rotation=18,
        ha="right",
        va="center",
        rotation_mode="anchor",
    )
    ax.set_xlabel("Stimulation intensity a", fontsize=12)
    ax.set_ylabel("" if index_only else "Stimulation target", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    zero = intensities.index(0.0)
    ax.axvline(zero, color="white", linewidth=0.8, alpha=0.9)
    ax.axvline(zero, color="black", linewidth=0.35, alpha=0.55)
    colorbar = ax.figure.colorbar(
        image, ax=ax, fraction=0.04, pad=0.012, extend="max", shrink=0.62
    )
    colorbar.set_label(color_label + " (p90 display)", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    return {
        "metric": metric,
        "vmin": vmin,
        "vmax": vmax,
        "p90_display": True,
        "capped_cells": int(np.sum(matrix > vmax)),
        "raw_min": float(matrix.min()),
        "raw_max": float(matrix.max()),
        "colormap": cmap,
        "gray_cells": 0,
        "raw_values_modified": False,
    }


def plot_results(output):
    """Export only FO heatmap, TPM heatmap, G selection, FO change and overview."""
    output = Path(output)
    metadata = json.loads((output / "run_metadata.json").read_text())
    if not metadata.get("complete"):
        raise ValueError("This run is incomplete; resume it before plotting.")
    summary = json.loads((output / "summary.json").read_text())
    fit = read_csv(output / "fit.csv")
    stimulation = read_csv(output / "stimulation.csv")
    comparison = read_csv(output / "fo_comparison.csv")
    config = metadata["config"]
    label = f"{summary['sample_id']} | {config['name']} | fit: {config['fit_repeats']} repeats; stimulation: {config['stimulation_repeats']} repeats"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    _fit(ax, fit, summary)
    fig.suptitle(label, fontsize=10)
    _save(fig, output / "fit")
    display_records = {}
    for metric, stem in (
        ("fo_state1", "stimulation_fo_heatmap"),
        ("hlc_tpm_frobenius", "stimulation_tpm_heatmap"),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 11), layout="constrained")
        display_records[stem] = _heatmap(ax, stimulation, metric)
        fig.suptitle(label, fontsize=10)
        _save(fig, output / stem)
        matrix, targets, intensities = _matrix(stimulation, metric)
        write_csv(
            output / f"{stem}_matrix.csv",
            [
                {
                    "target": name,
                    **{
                        f"a={a:g}": float(matrix[i, j])
                        for j, a in enumerate(intensities)
                    },
                }
                for i, (_, name) in enumerate(targets)
            ],
        )
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    _fo_bars(ax, comparison)
    fig.suptitle(
        f"{label}\n{summary['selected_target']}, a={summary['selected_a']:g}",
        fontsize=10,
    )
    fig.supxlabel(
        f"Simulation bars: mean ± SD over {config['stimulation_repeats']} noise repeats; one PTN; HLC reference is the group mean.",
        fontsize=9,
    )
    _save(fig, output / "fo_change")
    fig = plt.figure(figsize=(24, 13))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.18, 1.18, 1.05),
        left=0.075,
        right=0.985,
        bottom=0.12,
        top=0.91,
        wspace=0.34,
    )
    right = gs[0, 2].subgridspec(2, 1, hspace=0.4)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(right[0, 0]),
        fig.add_subplot(right[1, 0]),
    ]
    _heatmap(axes[0], stimulation, "fo_state1")
    _heatmap(axes[1], stimulation, "hlc_tpm_frobenius", index_only=True)
    _fit(axes[2], fit, summary)
    _fo_bars(axes[3], comparison)
    for letter, ax in zip("ABCD", axes):
        ax.text(
            -0.1, 1.025, letter, transform=ax.transAxes, fontweight="bold", fontsize=21
        )
    fig.suptitle(f"Single-PTN stimulation example | {label}", fontsize=14)
    fig.supxlabel(
        "Heatmaps use the manuscript p90 color scale; cells above p90 retain the endpoint color.\nFO simulation bars show mean ± SD across noise repeats. One PTN example; HLC reference is the group mean.",
        fontsize=11,
    )
    _save(fig, output / "overview")
    (output / "heatmap_display.json").write_text(
        json.dumps(
            {
                "style": "Paper-style p90 color scales; no outlines or gray masks",
                "panels": {
                    "A": "State 1 FO",
                    "B": "TPM Frobenius to HLC",
                    "C": "G selection",
                    "D": "FO changes",
                },
                "gray_cells": 0,
                "raw_values_modified": False,
                "a_step": config["a_step"],
                "stimulation_repeats": config["stimulation_repeats"],
                "heatmaps": display_records,
            },
            indent=2,
        )
        + "\n"
    )
