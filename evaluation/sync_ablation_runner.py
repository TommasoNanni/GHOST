"""
Runner for the temporal-synchronization ablation study.

Evaluates the four variants of evaluation.sync_ablation (full, no_confidence,
root_relative, mean_diagonal) on the RICH and EgoHumans scene lists of the
existing sync experiments, at max_shift ∈ {30, 60, 75}, 10 trials per scene.

Seeding replicates the existing experiments exactly: per (dataset, max_shift)
one np.random.default_rng(42) advances over the scenes in sorted(SCENE_PIDS)
order with the same draw order and sizes as alignment_experiments_multi[_
egohumans].run_scene — so the injected shifts and end cuts are bit-identical
to the published GHOST runs at the same max_shift, and identical across all
four variants (draws are made once per trial, every variant sees the same
slices).

Coverage counts scenes whose every trial was protocol-rejected in the
denominator (the RICH shift-75 reporting mismatch of the original experiment
is NOT reproduced here).

Metrics are pooled per-camera errors (pooled_stats, imported read-only from
the experiment module), grouped by dataset and max_shift.

Usage (GPU node):
    pixi run python -m evaluation.sync_ablation_runner
    pixi run python -m evaluation.sync_ablation_runner --dataset rich --max-shift 30
    pixi run python -m evaluation.sync_ablation_runner --tabulate   # re-print from saved JSONs

NOTE: --scene / --trials change the rng stream relative to the full run —
smoke tests only, not comparable to full-run numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

import evaluation.alignment_experiments_multi as rich_exp
import evaluation.alignment_experiments_multi_egohumans as ego_exp
from evaluation.sync_ablation import (
    DEVICE,
    VARIANTS,
    AblationSynchronizer,
    load_scene_bundle,
    run_trial_all_variants,
)

logger = logging.getLogger(__name__)

DATASETS = {"rich": rich_exp, "egohumans": ego_exp}
MAX_SHIFTS = (30, 60, 75)
N_TRIALS = 10
SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "paper_results" / "sync_ablation"

# pooled_stats is identical in both experiment modules; import one read-only.
pooled_stats = rich_exp.pooled_stats


def run_dataset_shift(
    dataset: str,
    max_shift: int,
    variants: tuple[str, ...],
    n_trials: int,
    scene_filter: list[str] | None,
) -> dict:
    """All scenes of one dataset at one max_shift, all variants per trial."""
    mod = DATASETS[dataset]
    engine = AblationSynchronizer(device=DEVICE, min_overlap=100, max_shift=max_shift)
    need_pos = "root_relative" in variants

    rng = np.random.default_rng(SEED)
    scene_names = sorted(mod.SCENE_PIDS)
    if scene_filter:
        unknown = [s for s in scene_filter if s not in mod.SCENE_PIDS]
        if unknown:
            raise KeyError(f"{dataset}: no SCENE_PIDS entry for {unknown}")
        logger.warning("--scene filter active: rng stream differs from the full run")
        scene_names = [s for s in scene_names if s in scene_filter]

    scenes_out = []
    for key in scene_names:
        scene_dir = mod.SCENES_ROOT / key
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"scene directory missing: {scene_dir}")
        logger.info(f"\n{'=' * 60}\n[{dataset} | shift {max_shift}] Scene: {key}")
        bundle = load_scene_bundle(mod, scene_dir, key, need_pos=need_pos)
        cam_ids = bundle.cam_ids
        logger.info(f"  Cameras: {cam_ids}   Persons: {bundle.pids}")

        trials_out = []
        for trial in range(n_trials):
            # Draw order and sizes replicate run_scene exactly; the draws are
            # consumed even when the trial is later protocol-rejected.
            raw_shifts  = [0] + rng.integers(-max_shift, max_shift + 1, size=len(cam_ids) - 1).tolist()
            true_shifts = {c: int(s) for c, s in zip(cam_ids, raw_shifts)}
            end_cuts    = {c: int(e) for c, e in zip(cam_ids, rng.integers(0, max_shift + 1, size=len(cam_ids)).tolist())}

            logger.info(f"  ── Trial {trial + 1}/{n_trials}  true shifts: {true_shifts}  end_cuts: {end_cuts}")
            res = run_trial_all_variants(engine, mod, bundle, true_shifts, end_cuts, variants)
            if res is None:
                logger.warning(f"  Trial {trial + 1} skipped (insufficient frames after shift)")
                trials_out.append({"true_shifts": true_shifts, "end_cuts": end_cuts,
                                   "rejected": True, "variants": {}})
                continue
            for v in variants:
                if res[v] is None:
                    logger.info(f"     [{v:>13}] solve produced NaN — trial dropped for this variant")
                else:
                    logger.info(f"     [{v:>13}] MAE={res[v]['mae']:.2f}")
            trials_out.append({
                "true_shifts": true_shifts, "end_cuts": end_cuts, "rejected": False,
                "variants": {v: (None if res[v] is None
                                 else {kk: res[v][kk] for kk in ("errors", "estimated", "true_times")})
                             for v in variants},
            })

        scenes_out.append({"scene": key, "cam_ids": cam_ids, "pids": bundle.pids,
                           "trials": trials_out})

    return {"dataset": dataset, "max_shift": max_shift, "seed": SEED,
            "n_trials": n_trials, "variants": list(variants), "scenes": scenes_out}


# ──────────────────────────────────────────────────────────────────────────
# Tabulation
# ──────────────────────────────────────────────────────────────────────────

def summarize(result: dict) -> dict[str, dict]:
    """Per-variant pooled stats + coverage for one (dataset, max_shift) result.

    Coverage denominator = scenes × trials, including scenes where every trial
    was protocol-rejected.
    """
    out = {}
    for v in result["variants"]:
        errors, solved, total = [], 0, 0
        for scene in result["scenes"]:
            for t in scene["trials"]:
                total += 1
                r = t["variants"].get(v)
                if t["rejected"] or r is None:
                    continue
                solved += 1
                errors.append(np.asarray(r["errors"]))
        E = np.concatenate(errors) if errors else np.array([])
        stats = pooled_stats(E) or {}
        out[v] = {**stats, "solved": solved, "total": total, "n_errors": len(E)}
    return out


def print_table(results: list[dict]) -> None:
    """Grouped tables: dataset × max_shift, one row per variant."""
    for res in results:
        summ = summarize(res)
        n_scenes = len(res["scenes"])
        print(f"\n── {res['dataset'].upper()}  δmax={res['max_shift']}  "
              f"({n_scenes} scenes × {res['n_trials']} trials, seed {res['seed']}) ──")
        hdr = f"  {'variant':<15}  {'MAE':>6}  {'MedAE':>6}  {'≤1fr':>7}  {'≤2fr':>7}  {'coverage':>15}"
        print(hdr)
        print(f"  {'-'*15}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*15}")
        for v in res["variants"]:
            s = summ[v]
            if not s.get("n_errors"):
                cov0 = f"0/{s['total']}"
                print(f"  {v:<15}  {'-':>6}  {'-':>6}  {'-':>7}  {'-':>7}  {cov0:>15}")
                continue
            cov = f"{s['solved'] / s['total'] * 100:.1f}% ({s['solved']}/{s['total']})"
            print(f"  {v:<15}  {s['mae']:>6.2f}  {s['median_ae']:>6.2f}  "
                  f"{s['within_1'] * 100:>6.1f}%  {s['within_2'] * 100:>6.1f}%  {cov:>15}")


def result_path(dataset: str, max_shift: int) -> Path:
    return OUT_DIR / f"{dataset}_shift{max_shift}.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync ablation runner")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--max-shift", type=int, nargs="+", default=list(MAX_SHIFTS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help="smoke tests only — changing this desyncs the rng stream")
    parser.add_argument("--scene", nargs="+", default=None,
                        help="smoke tests only — filtering desyncs the rng stream")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip (dataset, shift) pairs whose JSON already exists")
    parser.add_argument("--tabulate", action="store_true",
                        help="only re-print tables from the saved JSONs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    variants = tuple(v for v in VARIANTS if v in args.variants)  # canonical order

    if args.tabulate:
        results = []
        for ds in datasets:
            for ms in args.max_shift:
                p = result_path(ds, ms)
                if p.exists():
                    results.append(json.loads(p.read_text()))
                else:
                    logger.warning(f"missing: {p}")
        print_table(results)
        sys.exit(0)

    logger.info(f"Device: {DEVICE}  |  variants: {list(variants)}  |  "
                f"datasets: {datasets}  |  max_shifts: {args.max_shift}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for ds in datasets:
        for ms in args.max_shift:
            p = result_path(ds, ms)
            if args.skip_existing and p.exists():
                logger.info(f"skipping {ds} shift {ms} — {p} exists")
                results.append(json.loads(p.read_text()))
                continue
            res = run_dataset_shift(ds, ms, variants, args.trials, args.scene)
            results.append(res)
            if args.scene is None and args.trials == N_TRIALS:
                p.write_text(json.dumps(res))
                logger.info(f"saved {p}")
            else:
                logger.warning("smoke-test flags active — result NOT saved")
            print_table([res])

    print("\n" + "=" * 60 + "\nFINAL TABLES\n" + "=" * 60)
    print_table(results)
