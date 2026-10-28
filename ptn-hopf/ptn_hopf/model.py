"""Hopf dynamics and projection into fixed empirical LEiDA states.

The integrator preserves the source workflow's noise draws, Euler–Maruyama
step, warm-up and sampling. Two algebraic optimizations are regression-tested:
diffusive coupling as C @ x - row_sum(C) * x, and rank-two phase coherence.
Neither operation changes the model or the empirical state definitions.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.signal import butter, detrend, filtfilt, hilbert
from scipy.stats import zscore

TR = 3.0
DT = TR / 20.0
FILTER_LOW = 0.04
FILTER_HIGH = 0.07


@njit(cache=True)
def _integrate(sc, omega, a, g, x, y, noise_x, noise_y, warmup, n_timepoints):
    row_sum = np.sum(sc, axis=1)
    sample_steps = 20
    # The source discards one TR after warm-up, then samples every TR.
    effective = (n_timepoints + 1) * sample_steps
    saved = np.empty((n_timepoints, sc.shape[0]))
    sample = 0
    for step in range(warmup + effective):
        coupling_x = g * (sc @ x - row_sum * x)
        coupling_y = g * (sc @ y - row_sum * y)
        radius = x * x + y * y
        dx = (a - radius) * x - omega * y + coupling_x
        dy = (a - radius) * y + omega * x + coupling_y
        x = x + DT * dx + np.sqrt(DT) * noise_x[step]
        y = y + DT * dy + np.sqrt(DT) * noise_y[step]
        local = step - warmup
        if local >= sample_steps and local % sample_steps == 0:
            saved[sample] = x
            sample += 1
    return saved


def simulate(sc, omega, n_timepoints, a, g, noise_strength, seed):
    """Return BOLD-like x samples, shape (time, AAL region)."""
    sc = np.ascontiguousarray(sc, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    a = np.broadcast_to(np.asarray(a, dtype=float), omega.shape).copy()
    rng = np.random.default_rng(int(seed))
    effective = (n_timepoints + 1) * 20
    warmup = int(effective * (606.0 / 2000.0))
    total = warmup + effective
    x = 0.1 * rng.standard_normal(len(omega))
    y = 0.1 * rng.standard_normal(len(omega))
    # Draw the whole x noise array before y, as in the source implementation.
    noise_x = noise_strength * rng.standard_normal((total, len(omega)))
    noise_y = noise_strength * rng.standard_normal((total, len(omega)))
    bold = _integrate(sc, omega, a, float(g), x, y, noise_x, noise_y, warmup, int(n_timepoints))
    if not np.isfinite(bold).all():
        raise FloatingPointError("Non-finite Hopf trajectory. Check model inputs and parameters.")
    return bold


def filter_bold(bold):
    """Detrend, fourth-order 0.04–0.07 Hz band-pass, then z-score."""
    bold = np.asarray(bold, dtype=float)
    signal = detrend(bold - np.mean(bold, axis=0, keepdims=True), axis=0)
    b, a = butter(4, [FILTER_LOW / (0.5 / TR), FILTER_HIGH / (0.5 / TR)], btype="band")
    filtered = filtfilt(b, a, signal, axis=0)
    return np.nan_to_num(zscore(filtered, axis=0))


def leading_vectors(bold):
    """Leading eigenvectors of cos(phi_i-phi_j), with negative-sum sign.

    cos(phi_i-phi_j) = c_i*c_j + s_i*s_j. Its nonzero eigenspace is
    spanned by c and s, so solve a 2x2 Gram problem instead of 116x116.
    """
    phase = np.angle(hilbert(filter_bold(bold), axis=0))
    basis = np.stack([np.cos(phase), np.sin(phase)], axis=-1)
    gram = np.einsum("tni,tnj->tij", basis, basis)
    values, vectors = np.linalg.eigh(gram)
    lead = np.einsum("tni,ti->tn", basis, vectors[:, :, -1])
    lead /= np.sqrt(values[:, -1, None])
    lead[np.sum(lead, axis=1) > 0] *= -1
    return lead


def state_metrics(bold, centers):
    """Assign by Manhattan distance; return FO and row-stochastic TPM."""
    vectors = leading_vectors(bold)
    labels = np.argmin(np.abs(vectors[:, None, :] - centers[None, :, :]).sum(axis=2), axis=1)
    k = len(centers)
    fo = np.bincount(labels, minlength=k).astype(float) / len(labels)
    counts = np.zeros((k, k), dtype=float)
    np.add.at(counts, (labels[:-1], labels[1:]), 1)
    totals = counts.sum(axis=1, keepdims=True)
    tpm = np.divide(counts, totals, out=np.full_like(counts, 1.0 / k), where=totals > 0)
    tpm += 1e-12
    tpm /= tpm.sum(axis=1, keepdims=True)
    return fo, tpm


def symmetric_kl(p, q):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, None)
    p, q = p / p.sum(), q / q.sum()
    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))


def repeat_summary(data, g, a, repeats, noise_strength):
    """Common noise seeds across conditions at a given G, as in the source."""
    fos, tpms, fit_kls = [], [], []
    first_seed = int(data["seed_base"]) + int(round(g * 100000)) * 1000
    for repeat in range(repeats):
        bold = simulate(data["scaled_sc"], data["omega"], int(data["n_timepoints"]),
                        a, g, noise_strength, first_seed + repeat)
        fo, tpm = state_metrics(bold, data["state_centers"])
        fos.append(fo)
        tpms.append(tpm)
        fit_kls.append(symmetric_kl(data["empirical_fo"], fo))
    fos, tpms = np.asarray(fos), np.asarray(tpms)
    mean_fo, mean_tpm = fos.mean(axis=0), tpms.mean(axis=0)
    std_fo = fos.std(axis=0, ddof=1) if repeats > 1 else np.zeros(2)
    return {
        "g": float(g), "repeats": repeats,
        "fit_mean_repeat_kl": float(np.mean(fit_kls)),
        "fit_kl_of_mean_fo": symmetric_kl(data["empirical_fo"], mean_fo),
        "hlc_kl_of_mean_fo": symmetric_kl(data["hlc_mean_fo"], mean_fo),
        "hlc_tpm_frobenius": float(np.linalg.norm(mean_tpm - data["hlc_mean_tpm"])),
        "fo_state1": float(mean_fo[0]), "fo_state2": float(mean_fo[1]),
        "fo_sd_state1": float(std_fo[0]), "fo_sd_state2": float(std_fo[1]),
        **{f"tpm_{i+1}{j+1}": float(mean_tpm[i, j]) for i in range(2) for j in range(2)},
    }
