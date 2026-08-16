"""Re-score a finished synchronisation run's AUC at a different assumed fps.

``alignment_experiments_multi.py`` and ``alignment_experiments_multi_egohumans.py``
hard-code the AUC thresholds at 15 fps and only convert MAE/MedAE/within-N to
milliseconds afterwards, so a table quoted at the dataset's true capture rate
cannot be read off the log.  AUC is the one metric that moves with fps (MAE,
MedAE and the within-N columns are frame-space quantities), and re-running the
experiment just to change a threshold costs hours of GPU time.

This script recovers the pooled error set from the log instead.  Both drivers
print one ``error=<frames>`` line per camera per solved trial, and their TOTAL
row reports how many errors were pooled; the two counts agree exactly, so the
parsed set is the same E the log's own table was computed from.  ``--check``
asserts that, and re-deriving the 15 fps AUC reproduces the logged value.

One caveat: the logs print errors rounded to one decimal, so a residual of 1.02
frames reads as "1.0".  That is invisible in AUC (integrating min(E, tau) over
tau >= 2 frames averages the rounding away — the re-derived 15 fps AUC matches
the logged one to within 0.1 pt on all eight paper logs) but it does bite the
within-1-frame column, which tests the boundary the rounding sits on: EgoHumans
re-scores to 98.9% where the log says 97.1%.  Trust this script for AUC and the
log for within-N.

The error is the min-anchored per-camera residual ``|(est - est.min()) - (true
- true.min())|`` — the same quantity, at the same granularity and gauge, that
the VisualSync-side ``faithful/score_dumps.py`` pools, so numbers from the two
scorers are directly comparable as long as both are given the same ``--fps``.

Capture rates (verified from the dataset papers, quote them in table captions):
    RICH       30 fps  — "The 142 multi-view videos in RICH are recorded at a
                          rate of 30 frames per second." (CVPR'22 supp., App. B)
    EgoHumans  20 fps  — "We divide each sequence into shorter clips of 30
                          seconds on average at 20 FPS." (ICCV'23)

Usage:
    pixi run python -m evaluation.rescore_sync_auc --fps 30 \
        paper_results/sync_fixed_shift*.err
    pixi run python -m evaluation.rescore_sync_auc --fps 20 --check \
        paper_results/alignment_experiment_multi_egohumans_shift*.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# "INFO       cam_03: true=+86  estimated=+86.0  error=0.0"
ERROR_RE = re.compile(r"error=(-?\d+\.?\d*)")
# "INFO    Coverage: 92.9%  (1114 pooled per-camera errors)"
POOLED_RE = re.compile(r"\((\d+) pooled per-camera errors\)")


def pooled_stats(E: np.ndarray, fps: float) -> dict:
    """Table metrics as statistics of the pooled per-camera error set.

    Copied from the drivers' ``pooled_stats`` rather than imported, so this
    script keeps scoring old logs unchanged if the drivers are edited.  AUC@tau
    closed form: mean over thresholds t in [0, tau] of P(E <= t)
    == 1 - mean(min(E, tau)) / tau (exact integral, no discretisation).
    """
    auc = lambda tau: 1.0 - float(np.minimum(E, tau).mean()) / tau  # noqa: E731
    return {"n": len(E), "mae": float(E.mean()), "median_ae": float(np.median(E)),
            "within_half": float((E <= 0.5).mean()),
            "within_1": float((E <= 1).mean()),
            "within_2": float((E <= 2).mean()),
            "auc_100ms": auc(100 / 1000 * fps),
            "auc_500ms": auc(500 / 1000 * fps)}


def parse_log(path: Path, check: bool) -> np.ndarray:
    text = path.read_text(errors="replace")
    E = np.array([float(v) for v in ERROR_RE.findall(text)], dtype=np.float64)
    if len(E) == 0:
        raise SystemExit(f"{path}: no 'error=' lines — not an alignment-experiment log")
    claimed = POOLED_RE.findall(text)
    if claimed:
        want = int(claimed[-1])
        if len(E) != want:
            msg = (f"{path}: parsed {len(E)} errors but the log's TOTAL row pooled "
                   f"{want} — the log is truncated or interleaved")
            if check:
                raise SystemExit(msg)
            print(f"  WARNING: {msg}")
    elif check:
        raise SystemExit(f"{path}: no TOTAL row to check the parsed error count against")
    return E


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path,
                    help="alignment-experiment .log/.err files (shell globs fine)")
    # Comma-separated, not nargs="+": a variadic option would swallow the
    # positional log paths when --fps is given before them.
    ap.add_argument("--fps", type=lambda s: [float(v) for v in s.split(",")],
                    default=[15.0, 30.0],
                    help="comma-separated capture rate(s) to score at; "
                         "RICH=30, EgoHumans=20 (default: 15,30)")
    ap.add_argument("--check", action="store_true",
                    help="fail unless the parsed error count matches the log's TOTAL row")
    args = ap.parse_args()

    name_w = max(len(p.name) for p in args.logs)
    head = (f"{'log':<{name_w}}  {'fps':>5}  {'N':>5}  {'MAE':>6}  {'MedAE':>6}  "
            f"{'W-0.5':>6}  {'W-1':>6}  {'W-2':>6}  {'AUC@100':>8}  {'AUC@500':>8}")
    print(head)
    print("-" * len(head))
    for path in args.logs:
        E = parse_log(path, args.check)
        for fps in args.fps:
            s = pooled_stats(E, fps)
            print(f"{path.name:<{name_w}}  {fps:>5.0f}  {s['n']:>5}  {s['mae']:>6.2f}  "
                  f"{s['median_ae']:>6.2f}  {s['within_half']*100:>5.1f}%  "
                  f"{s['within_1']*100:>5.1f}%  {s['within_2']*100:>5.1f}%  "
                  f"{s['auc_100ms']*100:>7.1f}%  {s['auc_500ms']*100:>7.1f}%")
    print("\n  MAE/MedAE in frames; within-N and MAE do not depend on fps — only AUC does.")


if __name__ == "__main__":
    main()
