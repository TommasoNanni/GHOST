# ghost: what changed after v2 — a briefing

Written for someone who knows the pipeline **up to the v2 trained fusion module**
and nothing after. Everything below is measured, with the script and split noted
so you can re-derive it. Numbers are millimetres unless stated.

---

## 0. Where you left off

You know: SAM3D gives a per-camera SMPL-X body estimate; `PoseFusionModule` (v2,
~1.1M params, spatio-temporal attention over cameras and time) fuses the
per-view rotations; `BodyPlacer` then triangulates a root translation and global
orientation via Procrustes-DLT and places the body in the world.

Two things you should update.

1. **v2 is gone.** It is beaten on every metric on all three datasets by a
   parameter-free rule with no checkpoint.
2. **The pose problem and the placement problem are completely separate.**
   Fusion only ever touched pose. Almost all of the remaining error is
   placement, and placement is limited by camera geometry and metric scale.

---

## 1. The headline: geodesic median replaces v2

Per (frame, person, joint), fuse the per-camera rotations by their **geodesic
median** on SO(3) instead of learning a fusion:

    R̄ = argmin_R  Σ_k  d(R, R_k),     d(R₁,R₂) = arccos((tr(R₁ᵀR₂) − 1)/2)

Solved by Weiszfeld/IRLS: seed with the chordal mean, then 5 times recompute
θ_k = d(R_k, R̄) and re-take the weighted chordal mean with w_k = m_k/(θ_k+ε),
ε = 1e-3 rad. The inner step is a closed-form SVD projection onto SO(3)
(`U diag(1,1,det(UVᵀ)) Vᵀ`). No training, no checkpoint, no parameters.

Implementation: `median_fuse()` in `evaluation/evaluate_{rich,egohumans,egoexo}_median.py`
(identical copies). Visibility is per (camera, person), **not** per joint — the
mask is expanded across joints.

Note the inner solve is the *chordal* mean, not the Riemannian log/exp update.
That is consistent for the geodesic objective because ‖R−R_k‖_F = 2√2·sin(θ/2),
so with w_k = 1/θ_k the step minimises Σ 8sin²(θ_k/2)/θ_k ≈ Σ 2θ_k, exact as
θ→0. Inter-view spread here is ~7° (p99 34.9°), so the approximation is tight.

### Results

**RICH test, 52 scenes, M10 (neutral pred / gendered GT), scale=baseline, smooth=median**

| pipeline | WA-100 | W-100 | PA | RTE |
|---|---|---|---|---|
| **geodesic median** | **46.8** | **66.9** | **25.4** | **0.98** |
| chordal mean | 47.6 | 67.7 | 26.5 | 0.98 |
| v3 residual @ep112 | 48.0 | 68.0 | 27.4 | 0.98 |
| R2 @ep217 | 48.3 | 68.2 | 27.7 | 1.00 |
| **v2 (what you know)** | 50.4 | 70.4 | 30.4 | 1.00 |
| CHROMM | 53.1 | 79.0 | ~60 | 1.4 |

**EgoHumans, 132 scenes, SMPL-24, scale=baseline**

| | W† | GA | PA |
|---|---|---|---|
| **geodesic median** | **593.6** | **113.7** | **30.4** |
| v2 | 594.8 | 119.8 | 41.7 |
| CHROMM | 510 | 150 | 50 |

**EgoExo4D val, 179 takes, scale=baseline**

| | W† | PA |
|---|---|---|
| **geodesic median (17 COCO joints)** | **286.2** | **50.6** |
| geodesic median (12 joints, superseded) | 286.4 | 52.8 |
| chordal mean (12 joints) | 286.5 | 53.3 |
| v2 (12 joints) | 290.1 | 60.8–65.2 |
| CHROMM | 260 | ~60 |

⚠ **The EgoExo joint set changed on 2026-08-04** (see §5). Only the median row
was re-run at 17 joints; the mean and v2 rows are still 12-joint numbers and are
**not** directly comparable to it. Re-run them before putting them in one table.

The estimator was selected on RICH's 12-scene *train* pool and applied unchanged
— EgoExo4D and EgoHumans are zero-shot for it.

**The pattern:** PA improves everywhere (RICH −1.1, EgoExo −0.5, EgoHumans
−11.6) while W†/WA barely move. That is the whole story of §2 and §4.

---

## 2. Why the median wins, and why nothing beats it

Model each per-view estimate as

    ξ_k = b + n_k

where **b is identical in every view** (one network, one prior, K images of the
same pose) and n_k is view-specific noise. Averaging over K views gives

    E[ξ̄²] = ‖b‖² + σ²/K   →   ‖b‖²  as K → ∞

b adds *coherently* over views and survives; n adds *incoherently* and shrinks.
That asymmetry is the entire argument, and it bounds what any fusion rule can do.

Measured on RICH test 52 (position space, RMS): single view **55.87**, fused at
K≈3.8 **46.44**, floor at K→∞ **42.48**, σ = 35.72. So **55.2% of the per-view
error is shared** and unreachable by fusing. In rotation space ‖b‖ = 17.74°,
σ = 7.87°, 83.6% shared.

### The budget, on 38.9 mm RR-MPJPE

| lever | worth | reachable? |
|---|---|---|
| 1 → 4 views | **17%** | already done |
| robust rule instead of the mean | 0.53 (1.4%) | **banked — this is the median** |
| a better rule than the median | 0.04 | no point |
| perfect magnitude weighting | +0.07 over median | no point |
| 4 → 8 cameras | 4.6% | hardware; substitutes for robustness, not additive |
| oracle view selection | ~4.9 | **unreachable in principle** |
| oracle bias calibration | **2.63** | needs GT of the target pose distribution |

### The five steps, each a closed question

1–2. **What can fusion touch?** 55.2% shared; views worth 17%, the rule worth
   1.4%. → `fusion_bias_variance_decomposition`
3. **Is the median the right rule?** Noise kurtosis is 36 (≈12× Gaussian
   outlier rate). The median is within **1.2%** of the best M-estimator.
   → `noise_shape_median_optimal`
4. **Can weighting beat it?** The inverse-variance ceiling is −0.60 vs the
   median's −0.53 — a **0.07 mm** gap. → `inverse_variance_ceiling`
5. **Can the bias be calibrated away?** An oracle per-joint offset removes
   **−2.63**, but the same offset fitted on train **transfers +2.32** (i.e.
   makes test *worse*). → `bias_calibration_pose_dependent`

### Every negative is a mechanism, not a null

- **Three per-view quality signals all sit at chance**: confidence 29.2% (vs
  17–25% baseline), 2D reprojection 16.1% (vs 16.2%), occlusion 20.4% (vs
  16.2%). Reason: an observable tells you how *noisy* a view is, but the oracle
  wants the view whose noise happens to point *against* b. Those are nearly
  anti-correlated — the lucky view is often a high-noise one.
  → `per_joint_view_signals_exhausted`
- **b is structurally unobservable.** It cancels in every inter-view comparison
  (e_k − e_l = n_k − n_l), so any disagreement-based signal is blind to it.
- **v3 failed for a diagnosed reason.** The v3 residual architecture starts
  bit-exact at the chordal mean (zero-init residual head, verified 0.000e+00)
  and trains away from it. It isn't capacity or architecture: it learned the
  *train* pose distribution's bias, which transfers negatively — exactly the
  +2.32 measured directly in step 5. Legs transfer (l_knee cos 0.971);
  shoulders do not (train 7.19° vs test 1.89°). → `fusion_v3_residual_architecture`

**Conclusion: fusion is saturated.** Do not spend more time on fusion rules,
architectures, or per-view weighting. The only untested idea aimed at the 42.5 mm
floor is *breaking the shared bias* — b is shared because one network resolves
the same ambiguity identically in every view, so genuinely different per-view
models would decorrelate part of b into averageable noise.

---

## 3. Placement is a separate problem, and it is camera-limited

W-MPJPE† and WA/W are **not** pose metrics. Decomposition on EgoHumans:
W 594 ≈ pelvis-only 596; root-relative pose 51; residual orientation 5.5°.
**W is ~100% root placement.** On EgoExo, per-scene corr(W, PA) = 0.05 — the two
metrics are statistically independent.

That is why the median improves PA everywhere and leaves W untouched.

### Oracle ladder — where the placement error actually is

| dataset | prod | GT scale | GT cameras | GT pose | attribution |
|---|---|---|---|---|---|
| EgoExo4D (179) | 286 | **105** | 97 | — | scale **63%**, cameras 3% |
| EgoHumans (132) | 595 | 410 | **171** | 165 | scale 31%, **cameras 40%**, pose 5.5 mm |
| RICH (52), W-100 | 70.4 | 69.6 | **46.3** | 38.8 | scale 1%, cameras 33%, pose 7.5 |

**Do not say "W is mostly scale" — it is only true for EgoExo.** The defensible
claim across all three: *placement is limited by camera geometry and its metric
scale; pose contributes essentially nothing.*

### The scale mechanism (2026-08-04)

MapAnything's metric scale is wrong on the ego datasets. Measured per scene as
`s_ma / s_opt − 1`, where `s_opt` is the Umeyama scale taking the unscaled VGGT
camera centres onto the GT camera centres:

| dataset | \|scale error\| | rig distortion (Sim3 resid / extent) |
|---|---|---|
| RICH | **0.8%** | — |
| EgoHumans (132) | **8.9%** median | 5.2% (badminton 8.8%) |
| EgoExo4D (182) | **18.8%** median | 1.6% |

W† applies **one SE(3) fitted from camera centres, with no scale freedom**, so a
global scale error lands directly on the bodies. Kabsch on a point set that is a
(1+ε) dilation of the truth matches centroids exactly and recovers the rotation
exactly, but cannot shrink — leaving residual ε·r for anything at distance r from
the camera centroid. Fitted over 132 EgoHumans scenes:

    W = 0.91·(|ε| · lever) + 0.31·rig_resid + 32 mm      R² = 0.957
        scale term alone R² = 0.913     rig alone R² = 0.021

The 0.91 ≈ 1 coefficient is the physical prediction. `lever` = mean distance from
the bodies to the camera centroid.

Root cause: MapAnything infers focal length from scene *content*. On RICH's
indoor/parking content the guess is right; on large outdoor courts it thinks the
lens is a telephoto (badminton self-focal 468 vs true 296), and the scale follows.

**Two distinct failure modes, don't merge them:** tennis (+21.3%), volleyball
(−17.3%) and fencing (−11.8%) are scale-limited with clean rigs; **badminton is
rig-limited** (8.8% distortion, the worst of any activity).

EgoExo's scale error is mostly a **per-venue constant** (between-venue RMS 17.2%
vs within-venue 10.5%; uniandes +23.7% across all 73 takes), so averaging more
frames cannot fix it.

Scripts: `debug/w_error_decomposition_egohumans.py`, `debug/scale_error_egoexo.py`.

### There is no scale estimator that wins everywhere

Tried: MapAnything centered, MapAnything baseline (current default), MA with
forced VGGT focal, bone-length triangulation, and the human-reference estimator
("V2 scale" — unrelated to the v2 fusion module; `scale = L_metric/L_vggt` from
SAM3D 3D keypoints over DLT-triangulated 2D of the same landmarks).

Human-reference wins EgoHumans' compact venues (1–2% vs MA's 3–20%, tennis 2% vs
20%) but fails on wide venues (13–30%, thin covisibility) and **regresses RICH
badly** (W-100 77.9 → 93.6). It was deleted from the codebase on 2026-07-16;
recover from `git show 80618f7:fusion/placer.py`.

A covisibility gate does **not** rescue it: RICH's failure is a systematic +5%
SAM3D limb-length bias, not covisibility, so the gate would route RICH to the
estimator that breaks it. **This is a genuinely open problem, not an oversight.**

### Badminton's camera error is ours, not the dataset's

Every activity has one dominant bad camera (badminton has two: cam05 at 1090 mm,
cam07 at 991 mm mean residual). The offsets are **static** per camera
(sd/|mean| 0.05–0.36) and almost **purely horizontal** (|z|/|xy| 0.02–0.13),
which suggested stale GT colmap extrinsics.

**Tested and disproven (2026-08-04):** rendering GT-SMPL through the COLMAP
fisheye calibration onto raw exo frames of 039_badminton puts all four people
pixel-perfect in **both** cam05 (worst) and cam02 (best). The GT extrinsics are
correct; our VGGT reconstruction is the wrong one. No dataset-defect claim is
available. Tool: `utilities/render_egohumans_gt.py`.

Also note: only **4 of the 15** GT exo cameras are in the released `exo/` data
(cam01/02/05/07), and badminton scenes 050–061 lose the good cam01 — leaving
three cameras of which two are the bad ones. That is why that block reads 11.4%
rig distortion and W† ≈ 900 against 001–049's 8.3% and ≈ 530.

---

## 4. Reading the numbers without tripping

- **Two poolings, never compare across them.** RR-MPJPE is the *mean of
  per-joint distances* (38.9). The bias/variance decomposition must use **RMS**
  (46.44), because the correction ‖b̂‖² − s²/K is a second-moment identity.
  Ratio 1.19 — and that 19% gap is itself a symptom of the kurtosis-36 tail.
  **Fractions transfer, absolutes do not.**
- **The 48.5/47.8 family is the 12-scene TRAIN pool**, not test. That is the
  estimator-selection set, deliberately kept off test.
- **Check the scale-smoothing flag before comparing RICH rows.** `eval_*_mean.sh`
  defaults `SCALE_SMOOTH=none`; the published tables are `median`. On RICH the
  difference is W-100 83.5 vs 66.9 — with *identical* PA (25.4 both), the
  signature of a scale-only effect since Procrustes is scale-invariant.
- **Both decomposition caveats are conservative.** GT error is view-independent
  so it is indistinguishable from b (‖b‖ is an *upper* bound on the estimator's
  own shared error — never run this on EgoExo's 2-view pseudo-GT). Correlated
  n_k is likewise indistinguishable from larger b, so the true floor is only
  higher.
- **Scope.** Safe unconditionally: b cancels in every inter-view comparison
  (structural), and the median beats the mean (holds on all three datasets).
  Everything else is regime-specific to SAM3D / RICH / K≈4 / inter-camera spread
  ~7°. State the regime.

---

## 5. Metrics are protocol-verified (2026-08-04)

Audited against **source**, not paper summaries: WHAM's actual `eval_utils.py`,
plus CHROMM (arXiv 2603.12789), HSfM (2412.17806), DuoMo (2603.03265).

Verified identical: WA-MPJPE-100 (Sim3 with scale over the whole segment),
W-MPJPE-100 (Sim3 from the **first two frames**), RTE (SE(3) no scale via
`align_pcl(fixed_scale=True)`, normalised by **total path length**), W-MPJPE†
(SE(3) from camera **positions**, no scale — HSfM's wording), GA (single Sim3
per frame over all persons), PA (per-person Sim3), RICH joints (24 SMPL via
`smplx2smpl` → `J_regressor`), EgoHumans joints (24).

**One deviation, now fixed:** EgoExo4D was scoring 12 limb joints; the benchmark
is defined on all 17 COCO keypoints, and the GT annotates all 17 (nose in 173 of
182 takes, ears 141–168, vs ankles 150–152) — the restriction was a convention
choice, not a data limit. Fixed in all four EgoExo eval scripts. Matching is
provably unaffected: it gates on `GT_TO_MHR70` (12 limbs), so the joint set
cannot flip a person match, confirmed by identical scene count and excluded-take
list across the two runs.

**Unresolvable from public material** (disclose, don't chase): how competitors
handle joints with no triangulated GT; CHROMM's EgoExo joint set (their "24
joints" cannot apply — EgoExo4D has no SMPL GT); their scene counts.

---

## 6. Open items

1. **Volleyball GA = 323 against PA = 30.** GA is Sim(3)-aligned, therefore
   scale-free and frame-free, so *nothing in §3 explains it*. People are posed
   correctly but arranged wrongly relative to each other. ReID was manually
   verified correct, so this is probably a real bug. Highest value per hour of
   anything left.
2. **Metric scale**, if someone wants the hard open problem (§3).
3. Cosmetic: `evaluate_egohumans_median.py::scene_metrics` docstring says "12
   limb joints" while the code correctly uses `range(24)`.

**Do not restart:** fusion rules/architectures/weighting (§2), bundle adjustment
(three attempts, rotation got *worse*, 1.70° → 4.29°), generic "better cameras"
(EgoExo rig distortion is already 1.6%).
