# Solving the geodesic median with IRLS

Precise derivation of what `median_fuse()` computes and why. Implementation:
`evaluation/evaluate_{rich,egohumans,egoexo}_median.py:191`.

---

## 1. The problem

Fusion runs **independently for every (frame t, person p, joint j)**. Fix one
such element and drop the indices. Let $\mathcal{K}$ be the set of cameras in
which the person is detected ($m_k = 1$), and $R_k \in SO(3)$ the per-camera
SAM3D rotation, recovered from its 6D form by Gram–Schmidt.

Geodesic (angular) distance on $SO(3)$:

$$d(R_1,R_2) \;=\; \arccos\!\left(\frac{\operatorname{tr}(R_1^{\top}R_2)-1}{2}\right) \;\in\; [0,\pi]$$

We want the **$L_1$ Fréchet median**:

$$\bar R \;=\; \arg\min_{R \in SO(3)} \; f(R), \qquad f(R) \;=\; \sum_{k\in\mathcal{K}} \theta_k(R), \qquad \theta_k(R) := d(R,R_k)$$

Contrast with the chordal mean, which minimises $\sum_k \lVert R-R_k\rVert_F^2$.
The $L_2$ objective has a closed form; the $L_1$ objective does not. That is the
entire reason we need an iterative scheme.

Why $L_1$: its influence function is bounded, so one badly-wrong camera shifts
the estimate by a bounded amount instead of dragging it proportionally.

---

## 2. Why IRLS, and where the weights come from

The trick is to write the $L_1$ term as an $L_2$ term divided by itself:

$$\theta_k \;=\; \frac{\theta_k^{2}}{\theta_k}$$

then **freeze the denominator** at the current iterate $R^{(i)}$. Define the
surrogate

$$g\!\left(R \mid R^{(i)}\right) \;=\; \sum_k w_k^{(i)}\,\theta_k(R)^{2}, \qquad w_k^{(i)} \;=\; \frac{1}{\theta_k\!\left(R^{(i)}\right)}$$

This is a *weighted least-squares* problem — hence "iteratively reweighted least
squares".

**This is a genuine majorise–minimise scheme, so it descends.** By concavity of
$\sqrt{\cdot}$, for any $t,t_i \ge 0$: $\sqrt{t} \le \frac{t+t_i}{2\sqrt{t_i}}$.
Applying it with $t=\theta_k(R)^2$, $t_i=\theta_k(R^{(i)})^2$:

$$f(R) \;\le\; \tfrac12\sum_k\!\left[\frac{\theta_k(R)^{2}}{\theta_k(R^{(i)})} + \theta_k(R^{(i)})\right]$$

with **equality at $R=R^{(i)}$**. The right-hand side differs from
$g(R\mid R^{(i)})$ only by a factor $\tfrac12$ and an $R$-independent constant,
so minimising $g$ cannot increase $f$.

Intuition: a view currently far from the estimate ($\theta_k$ large) gets a small
weight. Outliers demote themselves.

---

## 3. The inner solve

Minimising $\sum_k w_k\,\theta_k(R)^2$ exactly is the *weighted Karcher mean*,
which has no closed form either. We substitute the **chordal** weighted mean,
which does:

$$M \;=\; \frac{\sum_k w_k R_k}{\sum_k w_k}, \qquad M \overset{\text{SVD}}{=} U\Sigma V^{\top}, \qquad \mathcal{P}(M) \;=\; U\,\operatorname{diag}\!\left(1,1,\det(UV^{\top})\right)V^{\top}$$

$\mathcal{P}(M) = \arg\min_{R\in SO(3)} \sum_k w_k\lVert R-R_k\rVert_F^2$ — the
orthogonal Procrustes solution. The $\det(UV^{\top})$ term forces $\det = +1$,
projecting onto $SO(3)$ rather than $O(3)$ so reflections are excluded.

### Why substituting chordal for geodesic is legitimate

For rotations the two metrics are linked exactly:

$$\lVert R-R_k\rVert_F^{2} \;=\; 6-2\operatorname{tr}(R^{\top}R_k) \;=\; 4-4\cos\theta_k \;=\; 8\sin^{2}(\theta_k/2)$$

so $\lVert R-R_k\rVert_F = 2\sqrt2\,\sin(\theta_k/2)$. As $\theta\to0$,
$8\sin^2(\theta/2) \to 2\theta^2$: the chordal inner objective is twice the
geodesic one, and a constant factor does not move the argmin. The two inner
solves therefore agree to second order.

Combining with the IRLS weights, the fixed point of our iteration minimises

$$\sum_k \frac{8\sin^{2}(\theta_k/2)}{\theta_k} \;\approx\; \sum_k 2\,\theta_k \;=\; 2f(R)$$

**exact in the small-angle limit.** Our inter-camera spread is ~7° (p99 34.9°),
where $8\sin^2(\theta/2)/(2\theta^2) = 0.999$ at 7° and $0.973$ at 35° — so the
approximation is tight across the entire operating range.

State this as a footnote in the paper rather than claiming the exact Riemannian
median.

---

## 4. The singularity, and what $\varepsilon$ actually does

If the iterate lands exactly on a data point, $\theta_k = 0$ and $w_k = \infty$.
This is the classical Weiszfeld singularity. We use

$$w_k \;=\; \frac{m_k}{\theta_k+\varepsilon}, \qquad \varepsilon = 10^{-3}\ \text{rad} \approx 0.057^{\circ}$$

This is not merely a numerical guard — it changes the objective in a
characterisable way. IRLS with weights $w=\rho'(\theta)/\theta$ minimises
$\sum_k\rho(\theta_k)$. Solving $\rho'(\theta)/\theta = 1/(\theta+\varepsilon)$
with $\rho(0)=0$:

$$\rho(\theta) \;=\; \theta - \varepsilon\ln\!\left(1+\frac{\theta}{\varepsilon}\right)$$

$$\rho(\theta) \approx \frac{\theta^{2}}{2\varepsilon} \quad (\theta \ll \varepsilon), \qquad\qquad \rho(\theta) \approx \theta - \varepsilon\ln(\theta/\varepsilon) \approx \theta \quad (\theta \gg \varepsilon)$$

So we are minimising a **Huber-like loss**: quadratic within $\varepsilon$ of the
estimate, $L_1$ everywhere beyond. Since $\varepsilon = 0.057^{\circ}$ is far
below the ~7° spread, effectively every camera sits in the $L_1$ regime and the
regularisation only activates in the degenerate case it exists to handle.

---

## 5. The algorithm as implemented

**Seed** — unweighted chordal mean over the visible cameras ($w_k = m_k$). The
$L_2$ solution is a cheap, sensible starting point; Weiszfeld on a manifold
converges *locally*, so the seed matters, and the chordal mean is inside the
basin for any realistic spread.

$$\bar R^{(0)} = \mathcal{P}\!\left(\frac{\sum_k m_k R_k}{\sum_k m_k}\right)$$

**Iterate**, $i = 0,\dots,N-1$ with $N=5$:

$$\theta_k^{(i)} = \arccos\!\left(\frac{\operatorname{tr}\!\left(R_k\,\bar R^{(i)\top}\right)-1}{2}\right), \qquad \bar R^{(i+1)} = \mathcal{P}\!\left(\frac{\sum_k \frac{m_k}{\theta_k^{(i)}+\varepsilon}R_k}{\sum_k \frac{m_k}{\theta_k^{(i)}+\varepsilon}}\right)$$

**Output** — rows 1–2 of $\bar R^{(N)}$, already orthonormal, back in 6D.

```
R̄ ← P( Σ mₖRₖ / Σ mₖ )                        # chordal-mean seed
repeat N = 5 times:
    θₖ ← arccos( (tr(Rₖ R̄ᵀ) − 1) / 2 )        # residual of each camera
    wₖ ← mₖ / (θₖ + ε)                         # demote the far ones
    R̄ ← P( Σ wₖRₖ / Σ wₖ )                     # weighted chordal mean, SVD
return rows 1–2 of R̄
```

### Implementation notes

- `arccos` is clamped to $[-1+10^{-7},\,1-10^{-7}]$ before evaluation; the
  derivative of $\arccos$ blows up at $\pm1$ and rounding does put it there.
- Every $(t,p,j)$ is solved **in parallel**, batched — the SVD is applied to a
  $(B,T,P,J,3,3)$ tensor. Cost is 6 batched $3\times3$ SVDs per element
  (1 seed + 5 iterations), negligible against the rest of the pipeline.
- Elements where **no** camera has a detection would give an all-zero $M$ whose
  SVD is arbitrary; those are seeded to $I$ purely to keep the decomposition
  well-conditioned. They carry no placed prediction downstream.
- Visibility $m_k$ is per **(camera, person)**, not per joint — the mask is
  expanded across $J$. We do not have, and do not use, per-joint visibility.

### Choice of $N$

Fixed at 5, no convergence test. 3 iterations are already converged at this
spread; 5 was used for every reported number. A fixed count keeps the whole
estimator branch-free and batched.

---

## 6. What this buys, and its ceiling

Against the chordal mean the median is worth **0.53 mm** of RR-MPJPE on RICH
(1.4% of 38.9). That is small but free — no training, no parameters, no
checkpoint — and it holds on all three datasets.

It is also near-optimal for this noise: the per-view error distribution has
kurtosis 36 (≈12× the Gaussian outlier rate), and the median lands within
**1.2%** of the best M-estimator we could fit. The best possible *magnitude*
weighting beats it by 0.07 mm. There is nothing left in the rule.

See `BRIEFING_post_v2.md` §2 for why the remaining error is not reachable by any
fusion rule.
