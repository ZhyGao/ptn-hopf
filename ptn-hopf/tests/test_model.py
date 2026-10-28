"""Scientific regression tests against direct, unoptimized formulations."""

import numpy as np
from scipy.signal import hilbert

from ptn_hopf.model import (
    DT,
    filter_bold,
    leading_vectors,
    simulate,
    state_metrics,
    symmetric_kl,
)
from ptn_hopf.pipeline import ROOT, load_inputs


def direct_simulation(sc, omega, n_timepoints, a, g, noise, seed):
    """Source Euler–Maruyama equations, with explicit pairwise coupling."""
    rng = np.random.default_rng(seed)
    effective = (n_timepoints + 1) * 20
    warmup = int(effective * (606 / 2000))
    total = warmup + effective
    x = 0.1 * rng.standard_normal(len(omega))
    y = 0.1 * rng.standard_normal(len(omega))
    noise_x = noise * rng.standard_normal((total, len(omega)))
    noise_y = noise * rng.standard_normal((total, len(omega)))
    saved = []
    for step in range(total):
        dx = (
            (a - x * x - y * y) * x
            - omega * y
            + g * np.sum(sc * (x[None, :] - x[:, None]), axis=1)
        )
        dy = (
            (a - x * x - y * y) * y
            + omega * x
            + g * np.sum(sc * (y[None, :] - y[:, None]), axis=1)
        )
        x, y = (
            x + DT * dx + np.sqrt(DT) * noise_x[step],
            y + DT * dy + np.sqrt(DT) * noise_y[step],
        )
        local = step - warmup
        if local >= 20 and local % 20 == 0:
            saved.append(x.copy())
    return np.asarray(saved)


def test_integrator_preserves_source_trajectory():
    rng = np.random.default_rng(5)
    sc = rng.uniform(0, 0.2, (8, 8))
    sc = (sc + sc.T) / 2
    np.fill_diagonal(sc, 0)
    omega = np.linspace(0.26, 0.42, 8)
    for a in (np.zeros(8), np.array([0.2, 0.2, 0, 0, 0, 0, 0, 0])):
        actual = simulate(sc, omega, 120, a, 0.19, 0.02, 12345)
        expected = direct_simulation(sc, omega, 120, a, 0.19, 0.02, 12345)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)


def test_rank_two_projection_matches_full_phase_coherence():
    data, _, _ = load_inputs(ROOT / "data")
    bold = simulate(
        data["scaled_sc"],
        data["omega"],
        198,
        np.zeros(116),
        0.205,
        0.02,
        531000666 + 20500 * 1000,
    )
    phase = np.angle(hilbert(filter_bold(bold), axis=0))
    direct = []
    for p in phase:
        _, vec = np.linalg.eigh(np.cos(p[:, None] - p[None, :]))
        v = vec[:, -1]
        direct.append(-v if v.sum() > 0 else v)
    np.testing.assert_allclose(leading_vectors(bold), direct, atol=1e-10, rtol=1e-9)
    direct = np.asarray(direct)
    labels = np.argmin(
        np.abs(direct[:, None, :] - data["state_centers"][None, :, :]).sum(axis=2),
        axis=1,
    )
    fo, tpm = state_metrics(bold, data["state_centers"])
    np.testing.assert_array_equal(fo, np.bincount(labels, minlength=2) / len(labels))
    np.testing.assert_allclose(tpm.sum(axis=1), 1)


def test_probability_metrics_and_input_bundle():
    data, targets, metadata = load_inputs(ROOT / "data")
    assert len(targets) == 45 and metadata["hlc_fo_participants"] == 46
    csv_centers = np.loadtxt(
        ROOT / "data/state_centers.csv", delimiter=",", skiprows=1, usecols=range(1, 117)
    )
    np.testing.assert_array_equal(csv_centers, data["state_centers"])
    assert symmetric_kl(data["hlc_mean_fo"], data["hlc_mean_fo"]) == 0
    assert np.isfinite(symmetric_kl([0, 1], [1, 0]))
    assert np.isclose(
        symmetric_kl([0.2, 0.8], [0.7, 0.3]), symmetric_kl([0.7, 0.3], [0.2, 0.8])
    )


def test_stimulation_zero_is_target_independent():
    data, _, _ = load_inputs(ROOT / "data")
    zero = np.zeros(116)
    zero[[0, 1]] = 0
    other = np.zeros(116)
    other[[78, 79]] = 0
    first = simulate(data["scaled_sc"], data["omega"], 198, zero, 0.205, 0.02, 123)
    second = simulate(data["scaled_sc"], data["omega"], 198, other, 0.205, 0.02, 123)
    np.testing.assert_array_equal(first, second)
