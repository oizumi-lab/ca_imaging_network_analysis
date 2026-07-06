# Shuffle-null control — are the state differences genuine, and what confounds clustering?

**Question.** When we compare the awake and unconscious functional networks
(modularity Q, clustering C, path length L / small-world propensity), are the
awake→unconscious differences driven by genuine changes in **cross-neuron
coupling**, or by a per-neuron property of the traces? The test: a **circular-shift
shuffle null** — roll each neuron's smoothed trace by an independent random lag.
This destroys all cross-neuron timing (coupling) while preserving each neuron's
own signal (its autocorrelation *and* its full marginal distribution). Anything a
measure keeps under this shuffle is *not* coupling.

All analyses use the same active-neuron subsample, window, and density (K = 5%)
across states; 10 recordings (6 sleep, 4 anaesthesia). Code:
`verification/shuffle_investigation.py` (shared per-recording cache),
`verification/shuffle_null_control.py` (raw values + state differences),
`verification/why_QL_robust_C_confounded.py` (OQ1),
`verification/state_difference_cause.py` (OQ2).

## Headline: how much of each state difference survives the shuffle?

Confound fraction = mean(ΔM_shuffle) / mean(ΔM_real) over 10 recordings:

| Measure | ΔM_real | ΔM_shuffle | **Confound** | Genuine excess |
|---|---|---|---|---|
| **L** (path length) | +0.072 | +0.003 | **~4%** | p = 0.005 |
| **Q** (modularity) | +0.056 | +0.010 | **~18%** | p = 0.015 |
| **C** (clustering) | +0.078 | +0.044 | **~56%** | p = 0.0015 |

So the awake→unconscious increases in **path length and modularity are genuine**
(coupling), while **most of the clustering increase is reproduced by
time-shuffled data** — a confound. SWP inherits it (it is built from C). The
lecture's main result (unconscious cortex is more modular, ΔQ ≈ +0.04 sleep /
+0.06 anaesthesia) stands.

Robustness: the shuffle over-reproduces C in **10/10** recordings; the raw
shuffle-floor shift is ordered L < Q < C in 10/10. The confound *fraction*
ordering is cleanest under anaesthesia (C mean 56% is pulled up by the deepest
recording; median-per-recording ~42%, leave-out-mouse03_ane ~46%). L's ~4% is a
real, high-effect-size result (ΔL_real ≈ 40× the shuffle sampling noise).

## OQ1 — why does the shuffle confound C but not Q or L?

`why_QL_robust_C_confounded.py` → `results/figures/oq1_*.png`.

A tempting guess — "chance coincidence-cliques are random, so they inflate local
triangles (C) but form no communities (Q) or shortcuts (L)" — is **false**: in an
independent-signal model, making the signals burstier inflates the shuffle value
of **all three** measures above the Erdős–Rényi baseline (Q is *not* flat).

The confound is instead set by how **local** each measure is:

- **C** is a local triangle count — exactly what chance coincidence-cliques
  create — so it inherits most of the marginal-driven difference (most confounded).
- **L** is a global integration measure. Chance-cliques are local and add no
  genuine shortcuts: in a real deep-anaesthesia graph the shuffle's path length
  sits within ~2–3% of a random null, while the real graph is 6–14% above it. So
  the awake→unconscious rise in L is almost entirely genuine (least confounded).
- **Q** is intermediate: chance-cliques form small scattered modules, but the real
  state change is a large-scale reorganisation into a **few large coherent
  modules** (mouse05_ane: 6 modules, largest 37% of nodes — vs the shuffle's 13
  small modules, largest 14%). See the adjacency-block figure
  `oq1_mechanism_modules.png`.

Adversarial cross-check: PARTIAL–CONFIRMED. The numbers reproduce exactly and a
steelman "it's just a big clustering floor" counter-explanation fails; "locality"
is an interpretation the data support (not prove), and the clean L<Q<C ordering is
clearest under anaesthesia.

## OQ2 — what drives the clustering confound, under identical smoothing?

`state_difference_cause.py` → `results/figures/oq2_state_difference_cause.png`.

The 15-frame Gaussian smoothing is identical across states, so the confound's
state difference must come from a marginal the shuffle preserves.

1. **Unconscious states are sparser and burstier.** Event rate and active-frame
   fraction fall ~40%; per-neuron kurtosis rises ~70%; concentration rises ~9%.
   Crucially the trace **autocorrelation is unchanged** (0.9755 → 0.9733) — the
   smoothing really is identical.
2. **Kurtosis is the unique predictor.** Across recordings, Δ(shuffle-C) is
   predicted by Δ(kurtosis) at Spearman ρ ≈ 0.99 (≈0.94 within sleep alone;
   survives dropping the two extreme-anaesthesia points and a partial correlation
   controlling for window length). Every other marginal has |ρ| < 0.3.
3. **Burstiness is causally sufficient.** An independent-signal model with **zero
   coupling**, matched only to each recording's kurtosis, reproduces the observed
   Δ(shuffle-C) at r ≈ 0.96 — and *over-predicts* the magnitude ~1.5×. A
   coupling-free null producing *more* difference than the data is strong evidence
   that coupling is not needed to explain it.

Nuance: kurtosis is the operative predictor on the **fixed-smoothing manifold** —
it works precisely because the data hold autocorrelation constant across states.
Clustering depends on kurtosis *and* smoothing width; kurtosis is not a universal
sufficient statistic off that manifold. Amplitude/variance is irrelevant
throughout (Pearson correlation is scale-invariant) — which is why an earlier
"variance/degree-heterogeneity" explanation was wrong and has been retracted.

Adversarial cross-check: CONFIRMED (all four attacks — circularity, leverage,
alternative predictors, causal direction — survive).

**Mechanism in one line.** Burstier traces let a few extreme frames dominate each
pair's correlation, so chance coincidences at those frames create triangle cliques
→ inflated *local* clustering, with no genuine modules or global shortcuts.

## Practical takeaway

- Trust **L** (and small-world path-length terms) most, and **Q** for coupling
  claims. Report **C / SWP** as *excess over the shuffle null*, not raw.
- When comparing clustering across conditions that differ in firing sparsity
  (states, drugs, cell types), the marginal burstiness alone can move C — control
  for it.
