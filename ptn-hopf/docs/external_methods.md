# External methods and study-specific code

The empirical LEiDA discovery workflow is external. The exact upstream repository
URL and version used by the study are to be supplied by the study authors. They
are not inferred from similarly named repositories. No third-party LEiDA toolbox
source is copied into this example, and no external checkout is required to run it.

The example contains a self-contained mathematical projection of simulated
signals into the **supplied** empirical states, plus the study-specific Hopf
simulation, fitting, perturbation and plotting workflow.

The provided empirical state space was derived with AAL116, TR=3 s,
0.04–0.07 Hz filtering, negative-sum leading eigenvectors and Manhattan-distance
state assignment. The two-state centers are fixed. Discovery parameters and
upstream code should be reported separately when releasing the full study.

Raw-image processing, tractography, MRI quality control, clinical analyses and
follow-up analyses are outside this single-participant demonstration.
