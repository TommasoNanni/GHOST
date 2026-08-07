# Why a parameter-free median beats a learned fusion module
### Slide draft — RICH test, 52 scenes, 2026-08-03

Notation used throughout: `b` = error component identical in every camera,
`n_k` = view-specific error, `K` = number of cameras (3.8 average on RICH).

---

## 1 — The question

We built a 1.1M-parameter transformer to fuse per-camera SMPL-X body poses.
A parameter-free geodesic median beats it.

**Why?** And is that a property of our module, or of the problem?

> Takeaway: this is not a negative result about one architecture. It is a
> measurement of how much fusion can be worth at all.

---

## 2 — Setup: what a fusion rule actually does

K cameras, each running the same monocular estimator, produce K rotations for the
same joint. Write each one's error in the tangent space at ground truth:

$$\xi_k = \log(R_{gt}^\top R_k) = \underbrace{b}_{\text{same in every view}} + \underbrace{n_k}_{\text{view-specific}}$$

**b exists because it is one network, one prior, K images of one pose.** When the
estimator misjudges a wrist, it misjudges it from every angle.

Any fusion rule averages over the camera index:

$$\text{fused error} = \sqrt{\|b\|^2 + \sigma^2/K} \;\xrightarrow[K\to\infty]{}\; \|b\|$$

> Takeaway: fusion shrinks σ. It cannot touch b. So the ceiling is set by how much
> of the error is shared.

---

## 3 — Measurement: 55% of the error is shared

One-way random-effects decomposition, 52 scenes, GT root + GT betas so only body
pose varies. Reported in millimetres of joint position (what RR-MPJPE scores).

| | mm |
|---|---|
| single view | 55.87 |
| **fused, K = 3.8** | **46.44** |
| ‖B‖ floor, K → ∞ | 42.48 |
| σ, view-specific | 35.72 |
| **shared fraction** | **55.2%** |

Consistency check E‖ξ‖² = ‖b‖² + σ² reproduces exactly (0.00% error).

> Takeaway: more than half of every view's error is identical across cameras and
> survives any amount of averaging.

---

## 4 — The budget: fusion matters, the fusion rule does not

| change | gain |
|---|---|
| 1 view → 4 views | **−9.4 mm (17%)** |
| naive rule → best rule | **−0.5 mm (1.3%)** |

**This is the headline.** Fusing views is worth 17%. Choosing *how* to fuse them
is worth 1%.

It explains everything observed: three learned modules all within ~1 mm of a plain
mean, five deterministic estimators within 0.8 mm of each other.

> Takeaway: we spent the effort on the 1% and left the 17% assumed.

---

## 5 — Why the median, then? Because the noise is pathological

| statistic | measured | Gaussian |
|---|---|---|
| excess kurtosis | **36.3** | 0 |
| p99 / median of ‖n‖ | 6.07 | ~2.4 |
| fraction beyond 3σ | **3.70%** | 0.3% |

Proof requiring no experiment: under Gaussian noise the median is only 2/π ≈ 0.64
as efficient as the mean, so it would *cost* ~0.3 mm. It *gains* 0.8 mm.
Therefore the tails are heavier than Gaussian.

Non-parametric bootstrap over real residuals, K = 4:

| estimator | vs mean |
|---|---|
| best of tuned family (Huber c=0.5) | −26.9% |
| **geometric median** | **−26.0%** |
| trimmed (drop worst) | −18.0% |
| Huber c=1.345 *(textbook default)* | −16.7% |

> Takeaway: the median is within **1.2%** of the best estimator in its family.
> Not "the best we tried" — effectively optimal.

---

## 6 — Optional: the gain depends on how many cameras you have

| K | gain from robust rule |
|---|---|
| 3 | 2.08 mm |
| **4** | **1.60 mm** |
| 8 | 0.82 mm |

> Takeaway: robustness and extra cameras are **substitutes**, not complements.
> The sparser the rig, the more the rule matters.

---

## 7 — Why per-view weighting can't help: the oracle is a mirage

An oracle that picks the best camera per joint gains **4.9 mm**. That looks like
headroom. It isn't.

Perfect averaging with *unlimited* cameras only reaches 42.5 mm. The oracle
reaches below that — so it is **not estimating b**. It is selecting views whose
noise happens to point *against* b (σ = 35.7 mm is nearly as large as ‖b‖ = 42.5,
so among 4 views one often partly cancels the bias).

**Why no signal can find it:**
1. b is common-mode — it **cancels in every inter-view comparison**
   ($e_k - e_l = n_k - n_l$), so disagreement-based signals are structurally blind
   to it
2. side channels measure ‖n_k‖ (*magnitude*); the cancellation depends on
   *direction* relative to an unobservable b
3. worse, they are near-anti-correlated: the lucky view is often a **high-noise**
   view whose noise pointed the right way

Measured, three independent signals:

| signal | picks best camera | chance |
|---|---|---|
| learned per-joint confidence | 29.2% | ~17–25% |
| 2D reprojection residual | 16.1% | 16.2% |
| mesh self-occlusion | 20.4% | 16.2% |

And the ceiling for *any* magnitude-based weighting, using the true per-camera σ:
**−0.59 mm vs the median's −0.52.**

> Takeaway: the three probes did not fail for being weak. A *perfect* magnitude
> signal is worth 0.07 mm. ~3.9 mm of the oracle is directional and unobservable.

---

## 8 — Why the learned module couldn't win either

v3 uses a **zero-initialised residual head**: at step 0 it is bit-exact the
chordal mean (verified 0.000e+00). The mean is not merely in its hypothesis
space — it is the **starting point**.

| | WA-100 | W-100 | PA |
|---|---|---|---|
| chordal mean (= v3 at init) | 47.6 | 67.7 | 26.5 |
| v3 after 112 epochs | 48.0 | 68.0 | **27.4** |

Gradient descent moved it **away from its own initialisation** on test.

That rules out capacity, architecture and initialisation — v3 was designed to
rule out exactly those. What remains is the training signal, and slide 9 shows
what it learned.

---

## 9 — The bias is real, large, and does not transfer

Fit one constant offset per joint (162 numbers), apply as
$R_{corr} = R_{fused}\exp(-b_j)$ — inference-time legal, no GT needed.

| | mm vs uniform |
|---|---|
| geodesic median | −0.54 |
| **calibration fitted on the eval data (oracle)** | **−2.63** |
| calibration fitted on train scenes | **+2.32** |

The oracle removes **5× what the median wins** — so the model is right. But
fitted elsewhere it actively *harms*.

Aggregates look fine (correlation 0.976, cos 0.897). The per-joint breakdown
shows why they lie:

| joint | ‖b‖ train | ‖b‖ test | outcome |
|---|---|---|---|
| l_knee | 15.1° | 17.4° | 17.4 → 4.6 ✅ |
| **l_shoulder** | **7.2°** | **1.9°** | 1.9 → **5.7** ❌ |

Legs transfer (everyone stands alike). Shoulders don't — shoulder bias depends on
what the arms are doing, and train activities differ from test.

> Takeaway: **b is pose-distribution-dependent.** This is exactly what v3 learned,
> and why it transferred negatively. Same sign, same magnitude.

---

## 10 — Conclusion

**Fusing views is worth 17%. Every remaining lever needs information only the
ground truth provides.**

- the noise *direction* relative to b — b cancels in every observable comparison
- b itself — knowable only from the GT of the target pose distribution

One mechanism, five measurements, explaining three signals at chance, five
estimators within 0.8 mm, and three learned modules at or behind a plain mean.

**Ship:** the geodesic median. Free, zero-shot, wins the pose metric on RICH
(PA 25.4 vs 26.5), EgoExo4D (52.8 vs 53.3) and EgoHumans (30.4 vs 42).

**What would actually move the needle:**
1. **more cameras** — 4 → 8 is 4.6%, ~3× any fusion rule
2. **break the shared bias** — b is shared *because* it is one network; different
   models per view would decorrelate part of it into averageable noise (untested,
   and the only idea aimed at the 42.5 mm)
3. **per-deployment calibration** — with GT for a few scenes of your own pose
   distribution, 162 numbers are worth 2.6 mm

---

## Backup — scope

Safe unconditionally: b cancels in every inter-view comparison (structural); the
median beats the mean (all three datasets). Everything else is regime-specific to
SAM3D / RICH / K≈4 / inter-camera spread ≈7°. Inter-camera spread p99 is 34.9°,
so "small-angle ⇒ chordal ≈ intrinsic" holds for the bulk, not the tail — the
claim survives because chordal vs Karcher was measured directly at 0.0 mm.

## Backup — caveats on the decomposition

GT error is view-independent, so it is indistinguishable from b ⇒ ‖b‖ is an
**upper bound** on the estimator's own shared error. Correlated n_k is likewise
indistinguishable from larger b ⇒ the true floor can only be **higher**. Both
caveats push the same way.

## Backup — reproducibility

All scripts in `debug/`, all reproduce `uniform = 38.9` as a harness check:
`bias_variance_decomp_rich.py`, `bias_variance_mm_rich.py`,
`noise_tail_shape_rich.py`, `inverse_variance_ceiling_rich.py`,
`bias_calibration_rich.py`. Results in `eval_explainability/*.json`.
