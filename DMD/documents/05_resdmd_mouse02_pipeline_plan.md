# Focused ResDMD pipeline plan: `mouse02_sleep`

## Corrections to the previous pipeline

The previous smoke test did **not** use PyDMD or Residual DMD. It fitted a
custom ridge-regularized, no-intercept linear map to PCA scores,

\[
z_{t+\ell}=Bz_t+\varepsilon_t,
\qquad
\widehat B=YX^{\mathsf T}
(XX^{\mathsf T}+\alpha I)^{-1}.
\]

Its selected input was native `spike_deconv`, followed by development-only
neuronwise centering/RMS scaling and PCA. It did not first convert the signal to
a mean firing rate.

PyDMD 2025.8.1 does not implement Residual DMD. PyDMD's `RDMD` class means
**Randomized DMD**, not Residual DMD. PyDMD is now installed and performs every
DMD and Hankel-DMD fit. The published Residual-DMD matrices are calculated only
to diagnose the eigenfunction residuals of the PyDMD candidates; they do not
replace the PyDMD fit.

Official references:

- [PyDMD API index](https://pydmd.github.io/PyDMD/code.html)
- [PyDMD Hankel-DMD API](https://pydmd.github.io/PyDMD/hankeldmd.html)
- [PyDMD introductory tutorial](https://pydmd.github.io/PyDMD/tutorial1dmd.html)
- [PyDMD Randomized DMD (`RDMD`)](https://pydmd.github.io/PyDMD/rdmd.html)
- [Residual-DMD authors' reference implementation](https://github.com/MColbrook/Residual-Dynamic-Mode-Decomposition)
- [Residual DMD paper](https://doi.org/10.1017/jfm.2022.1052)

## Narrow scope

This iteration uses only:

- recording: `mouse02_sleep.mat`;
- state: one contiguous, acquisition-safe NREM block;
- signal: `spike_deconv` only;
- block length: 1,500 native frames;
- estimator: PyDMD `HankelDMD`, followed by linear-dictionary Residual-DMD
  diagnostics;
- no awake comparison, second mouse, window sweep, bootstrap, null analysis,
  Grassmann tracking, or biological state claim.

The existing multi-recording smoke test remains a historical artifact and is
not extended by this tutorial.

## One explicit preprocessing choice

The default tutorial converts each neuron's deconvolved event mass into a
non-overlapping four-frame rate proxy:

\[
r_i[b] = \frac{\sum_{j=0}^{3}s_i[4b+j]}{4/7.65\ \mathrm{s}}.
\]

This preserves one row per neuron and produces 375 samples from the 1,500-frame
block. It is deliberately called an **event-mass-rate proxy**, not calibrated
spikes/s: OASIS amplitudes are not known spike counts. Non-overlapping bins avoid
the artificial serial dependence introduced by a moving average. The tutorial
plots before/after sparsity so this choice can be rejected visibly if it remains
too discontinuous.

Only the first 60% of rate bins fit neuron eligibility and centering/RMS
scaling. No arbitrary PCA rank is imposed before PyDMD. Instead,
`svd_rank=0` invokes PyDMD's documented automatic hard-threshold rank. The next
20% choose the delay order, and the final 20% remain untouched until one-step
forecast and Residual-DMD checks. Snapshot pairs never cross these boundaries.

## PyDMD and Hankel parameters

Delay candidates are (d\in\{1,2,4,8,16\}). Here (d) is the number of
consecutive binned neural snapshots in each PyDMD Hankel column; (d=1) means
no delay augmentation. At 0.523 s per bin, these candidates contain 0, 0.523,
1.569, 3.660, or 7.843 s of history before the newest observed snapshot.

All candidates use `tlsq_rank=0`, `exact=True`, `opt=True`,
`rescale_mode=None`, `forward_backward=False`, no Tikhonov regularization, and
`reconstruction_method="mean"`. Delay is selected by calibration one-step
(R^2) relative to the training-mean predictor. Persistence skill is reported
but is not the selection metric because a near-zero model can beat noisy
persistence without explaining neural variance.

## Residual-DMD calculation

For dictionary values \(\Psi_X,\Psi_Y\) and uniform quadrature weights,

\[
G=\Psi_X^*W\Psi_X,\qquad
A=\Psi_X^*W\Psi_Y,\qquad
L=\Psi_Y^*W\Psi_Y.
\]

Candidate eigenfunctions solve \(Ag=\lambda Gg\). Their independent calibration
residual is

\[
\operatorname{res}(\lambda,g)^2=
\frac{g^*\left(L-\lambda A^*-\bar\lambda A+|\lambda|^2G\right)g}
{g^*Gg}.
\]

This is a Koopman eigenfunction residual, not the ordinary reconstruction
residual used in the previous report. For stochastic neural observations it
also contains stochastic variance, so a large value does not by itself prove
that the underlying brain lacks dynamics.

## Interactive script order

1. Correct the method names and freeze this narrow scope.
2. Run an exact synthetic check of the Residual-DMD equations.
3. Inspect the full state timeline and verify the single NREM block.
4. Load only the selected `spike_deconv` slice and visualize all neurons.
5. Construct and visualize the four-frame event-rate proxy.
6. Make the chronological 60/20/20 split and fit reversible scaling on training
   data only.
7. Fit PyDMD Hankel candidates and select (d) using calibration (R^2).
8. Quantify retained Hankel energy, training reconstruction, and untouched-test
   one-step capture in transformed and physical neuron units.
9. Verify PyDMD eigenvalues and evaluate training, calibration, and test
   Residual-DMD eigenfunction residuals.
10. Count real modes and conjugate pairs; convert eigenvalues to frequencies and
    decay times.
11. Lift candidate modes to all neuron coordinates and report participation and
    nearest-neighbor coherence.
12. State what technically succeeded, what scientifically failed, and stop.

## Review checkpoint

This script fixes and audits only the analysis plumbing. The current result
selects (d=1), explaining that Hankel augmentation did not improve calibration
capture. The selected 25-mode fit explains approximately 0.2% of untouched
transformed-neuron variance and has test residuals of approximately 0.80--1.04.
It therefore does not support a successful biological DMD claim. After
inspecting its figures, the next decision is limited to one of three choices:

1. retain the four-frame rate proxy;
2. revise only the rate representation or block duration; or
3. abandon linear-dictionary ResDMD for an observation-aware latent model.
