# Numerical methods

## Hopf model

For each region i, using already scaled SC weights C:

```text
dx_i = [(a_i - x_i² - y_i²)x_i - omega_i*y_i
        + G * sum_j C_ij*(x_j - x_i)] dt + beta*sqrt(dt)*xi_i
dy_i = [(a_i - x_i² - y_i²)y_i + omega_i*x_i
        + G * sum_j C_ij*(y_j - y_i)] dt + beta*sqrt(dt)*eta_i
```

Euler–Maruyama integration uses dt=0.15 s, TR=3 s, beta=0.02 and independent
standard normal x/y noise. Initial x/y values are 0.1 times independent standard
normal draws. The NumPy `default_rng` stream draws initial x, initial y, the
entire x-noise array, then the entire y-noise array, in that order.

For T=198 saved samples, the source workflow uses `(T+1)*20 = 3980` effective
steps and `int(3980*606/2000) = 1205` warm-up steps. After warm-up, discard the
first 20 effective steps, then save post-update x at local steps 20, 40, …, 3960.
This reproduces the original sampling convention exactly (including its
post-update sample timing); warm-up is not rounded to a different duration.

The accelerated coupling expression `C @ x - C.sum(axis=1)*x` is algebraically
equivalent to the explicit pairwise sum. Numba compiles the time loop without
fast-math. Tests compare trajectories against the direct source equations.

## LEiDA projection of simulated signals

Each regional simulated signal is demeaned, linearly detrended, filtered using
a fourth-order Butterworth 0.04–0.07 Hz band-pass with `scipy.signal.filtfilt`,
and z-scored. Hilbert phases produce the phase-coherence matrix
`M_ij(t) = cos(phi_i(t)-phi_j(t))`.

The matrix factors as `M = c*c.T + s*s.T`, with c=cos(phi), s=sin(phi). Its
nonzero eigenvectors can be recovered from the 2×2 Gram matrix of `[c,s]`.
The leading vector is normalized to unit length and flipped if its component
sum is positive. Tests compare this result against a full 116×116 symmetric
eigendecomposition. Each vector is assigned to the supplied empirical centroid
with minimum Manhattan distance.

All 198 simulated timepoints are classified, matching the source simulation
pipeline. Empirical state discovery used its own endpoint handling upstream;
this example uses its supplied FO and centroids rather than reprocessing data.

## FO, TPM and distance

FO is the fraction of assigned timepoints in each state. TPM counts transitions
between adjacent labels and normalizes each row. A state with no outgoing
observations receives a uniform row. Add 1e−12 to all entries and renormalize,
as in the source code.

Symmetric KL is `0.5 * (KL(P || Q) + KL(Q || P))`, using natural logarithms.
Clip probabilities below 1e−12 and renormalize before evaluating the logarithms.

Fitting minimizes **mean_r symmetric_KL(empirical_PTN_FO, simulated_FO_r)**.
Stimulation is evaluated by **symmetric_KL(HLC_mean_FO, mean_r simulated_FO_r)**.
These are different quantities; their column names preserve the distinction.
The auxiliary TPM metric is the Frobenius norm of mean simulated TPM minus mean
HLC TPM. No TPM threshold is applied when choosing the demonstrated optimum.

## Perturbations and seeds

Baseline a=0 in all 116 regions. Stimulation changes a only at the left/right
members of one of the 45 AAL90 target pairs. G, SC, frequencies, simulation
length and noise amplitude remain fixed at the selected participant's settings.

For repeat index r (starting at zero):

```text
seed = seed_base + round(G * 100000) * 1000 + r
```

Seeds are shared across stimulation targets and intensities at the selected G,
matching the source workflow. They are independent across repeats. Reassigning
the public sample identifier or changing process count does not change seeds.
At a=0 the target has no effect, so the same baseline is reused for all targets.

The descriptive optimum minimizes HLC FO-KL across **nonzero** stimulation
conditions. Exact ties are broken by target order, then increasing a. The distributed default uses the full G/intensity grids and repeat counts.
No independent selection/validation split is claimed.
