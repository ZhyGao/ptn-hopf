# Processed input dictionary

`ptn_demo.npz` contains one real PTN participant's processed model inputs and
group-level HLC targets. Public example identifier: `PTN_demo_01`.
It contains no pickle objects; load with `numpy.load(..., allow_pickle=False)`.

| Key | Shape | Definition |
|---|---|---|
| `scaled_sc` | 116 × 116 | Individual MRtrix structural connectome, symmetrized, nonnegative, zero diagonal, scaled once to maximum 0.2 |
| `omega` | 116 | Angular frequencies, radians/second, pre-estimated from the participant's processed BOLD spectra in 0.04–0.07 Hz |
| `empirical_fo` | 2 | Participant's empirical FO/PMS in the fixed state order |
| `empirical_tpm` | 2 × 2 | Participant's empirical transition matrix; rows=current state, columns=next state |
| `state_centers` | 2 × 116 | Empirical LEiDA centroids shared with the original analysis; do not reorder or renormalize |
| `hlc_mean_fo` | 2 | Arithmetic mean of individual FO vectors from 46 HLC participants, normalized to sum to one |
| `hlc_mean_tpm` | 2 × 2 | Arithmetic mean of 46 individual HLC TPMs, not a TPM reconstructed from mean FO |
| `n_timepoints` | scalar | 198 output samples per simulated trajectory |
| `seed_base` | scalar | Numeric seed prefix preserving the original simulation stream without dependence on the public example identifier |

FO is fractional occupancy; PMS (probability of metastable states) is equivalent
terminology in the source analysis. State 1 and State 2 retain their empirical
order throughout fitting, simulation and plotting.

`aal116_regions.csv` defines the ordered one-based AAL region indices used by SC,
frequencies and centroids. `bilateral_targets.csv` lists 45 left/right pairs in
the first 90 AAL regions. The cerebellar regions remain in the simulated network
but are not stimulation targets in this example.

`metadata.json` records dimensions, the target-group count, selection rationale,
units, seed rule and SHA256 checksums. The loader validates those checksums,
probability normalization, matrix dimensions and target/region agreement.

The SC is already scaled. **Do not scale it again.** No EC optimization is used.
Raw DICOM/NIfTI images, tractograms, regional BOLD timeseries, clinical fields and
individual HLC records are not included. Thus this repository reproduces the
processed-input model workflow, not MRI preprocessing or LEiDA state discovery.

The participant was previously used as a worked model example and has empirical
State 1 FO below the HLC group mean. This choice is illustrative, not a random or
representative sample. Group-derived centroids and HLC means are reference
summaries, not additional individual datasets.

`state_centers.csv` is a human-readable 2×116 export with explicit state and
AAL column labels. It preserves the full float values of `state_centers` in the
canonical NPZ bundle. Model runs use the NPZ array.
