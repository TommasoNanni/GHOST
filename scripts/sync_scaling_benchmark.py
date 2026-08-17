"""Sync scaling benchmark: pairwise offset-estimation runtime vs window length T.

Takes one long EgoHumans scene (default 07_tennis/007_tennis: 1551 frames,
4 exo cameras, 2 people), injects random per-camera offsets, and runs the
Synchronizer's pairwise DTW offset estimation on the first T frames for
T = t_min, t_min+step, ..., t_max. Each T is repeated with several random
offset draws; the report is the mean wall-clock time to estimate ONE pairwise
offset (total offset-matrix time / number of camera pairs).

Pose sequences come from the curated body_data_clean/ tracks (same recipe as
scripts/runtime_benchmark.py stage 3): per-frame 51x3 axis-angle body+hand
pose as the DTW "joints", SAM3D joint confidences as weights. The Synchronizer
search range (max_shift) is held constant across T so runtime scales with T
only.

The recovered global start times are checked against the injected shifts at
every run (mean absolute error, saved to the JSON — sanity, not plotted).

Outputs (in --out-dir):
    sync_scaling.json         per-T timings + errors (written incrementally)
    sync_scaling.png / .pdf   the graph: x = T, y = seconds per pairwise offset

Usage:
    pixi run python scripts/sync_scaling_benchmark.py
    pixi run python scripts/sync_scaling_benchmark.py --plot-only   # re-plot JSON
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from synchronize_videos.synchronizer import Synchronizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── data loading ─────────────────────────────────────────────────────────────

def load_scene_sequences(
    clean_scene: Path, device: torch.device
) -> tuple[list[str], list[int], list[list[torch.Tensor]], list[list[torch.Tensor]], int]:
    """Full-timeline pose/conf tensors per (camera, person) from body_data_clean/.

    Returns (cams, pids, joints[K][P] (L,51,3), confs[K][P] (L,51), L).
    Frames a track misses stay zero-pose / zero-confidence, exactly like the
    runtime benchmark's sync stage.
    """
    cams = sorted(
        d.name for d in clean_scene.iterdir()
        if d.is_dir() and (d / "body_data_clean").is_dir()
    )
    if len(cams) < 2:
        raise FileNotFoundError(f"need >=2 cameras with body_data_clean under {clean_scene}")

    raw: dict[str, dict[int, dict]] = {}
    pid_set: set[int] = set()
    stem_min, stem_max = None, None
    for cam in cams:
        raw[cam] = {}
        for f in sorted((clean_scene / cam / "body_data_clean").glob("person_*.npz")):
            pid = int(f.stem.split("_")[1])
            d = dict(np.load(str(f), allow_pickle=False))
            fi = d["frame_indices"].astype(int)
            raw[cam][pid] = d
            pid_set.add(pid)
            lo, hi = int(fi.min()), int(fi.max())
            stem_min = lo if stem_min is None else min(stem_min, lo)
            stem_max = hi if stem_max is None else max(stem_max, hi)
    pids = sorted(pid_set)
    L = stem_max - stem_min + 1

    joints: list[list[torch.Tensor]] = []
    confs: list[list[torch.Tensor]] = []
    for cam in cams:
        per_p_seq, per_p_conf = [], []
        for pid in pids:
            seq = np.zeros((L, 51, 3), dtype=np.float32)
            conf = np.zeros((L, 51), dtype=np.float32)
            d = raw[cam].get(pid)
            if d is not None:
                pose153 = np.concatenate([
                    d["smplx_body_pose"], d["smplx_left_hand_pose"],
                    d["smplx_right_hand_pose"],
                ], axis=1).astype(np.float32)               # (T_stored, 153)
                c51 = d["pred_joint_confidence"][:, 1:52].astype(np.float32)
                rows = d["frame_indices"].astype(int) - stem_min
                seq[rows] = pose153.reshape(-1, 51, 3)
                conf[rows] = c51
            per_p_seq.append(torch.from_numpy(seq).to(device))
            per_p_conf.append(torch.from_numpy(conf).to(device))
        joints.append(per_p_seq)
        confs.append(per_p_conf)
    return cams, pids, joints, confs, L


# ── one timed run ────────────────────────────────────────────────────────────

def run_once(
    joints: list[list[torch.Tensor]],
    confs: list[list[torch.Tensor]],
    cams: list[str],
    T: int,
    max_shift: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[float, float, dict[str, int]]:
    """Inject random per-camera shifts, time the pairwise offset matrix.

    Returns (seconds_per_pair, mean_abs_start_time_error_frames, shifts).
    """
    K = len(cams)
    shifts = {cams[0]: 0}
    for cam in cams[1:]:
        shifts[cam] = int(rng.integers(-max_shift, max_shift + 1))
    base_pos = max_shift

    win_j, win_c = [], []
    for k, cam in enumerate(cams):
        pos = base_pos + shifts[cam]
        win_j.append([s[pos:pos + T] for s in joints[k]])
        win_c.append([c[pos:pos + T] for c in confs[k]])

    sync = Synchronizer(
        use_acceleration_weights=False, device=str(device),
        min_overlap=max(20, T - 3 * max_shift),
        max_shift=2 * max_shift, verbose=False,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    offset_mat = sync.estimate_offset_matrix(win_j, win_c)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_pairs = K * (K - 1) // 2
    weights = sync.cycle_consistency_weights(offset_mat)
    est = sync.estimate_initial_times(offset_mat, weights).cpu().numpy()
    est = est - est.min()
    true = np.array([shifts[c] for c in cams], dtype=np.float32)
    true = true - true.min()
    err = float(np.abs(est - true).mean())
    return elapsed / n_pairs, err, shifts


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [r["T"] for r in results["sweep"]]
    means = np.array([r["mean_s_per_pair"] for r in results["sweep"]])
    stds = np.array([r["std_s_per_pair"] for r in results["sweep"]])

    line = "#3d6de0"      # single series: one mid-blue, band = same hue faded
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    ax.plot(ts, means, color=line, linewidth=2, marker="o", markersize=4,
            markerfacecolor="white", markeredgewidth=1.4, zorder=3)
    ax.fill_between(ts, means - stds, means + stds, color=line, alpha=0.15,
                    linewidth=0, zorder=2)

    ax.set_xlabel("Input frames per camera $T$")
    ax.set_ylabel("Time per pairwise offset (s)")
    pad = 0.02 * (max(ts) - min(ts))
    ax.set_xlim(min(ts) - pad, max(ts) + pad)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="0.88", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sync_scaling.{ext}", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved → {out_dir / 'sync_scaling.png'} (+ .pdf)")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenes-root", type=Path,
                        default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new"))
    parser.add_argument("--scene", type=str, default="07_tennis/007_tennis")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/sync_scaling_benchmark"))
    parser.add_argument("--t-min", type=int, default=150)
    parser.add_argument("--t-max", type=int, default=1500)
    parser.add_argument("--t-step", type=int, default=50)
    parser.add_argument("--max-shift", type=int, default=25,
                        help="Max |injected offset| in frames; needs 2*max_shift + t_max <= scene length")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Random offset draws per T (timings averaged)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only re-render the figure from an existing sync_scaling.json")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "sync_scaling.json"

    if args.plot_only:
        with open(json_path) as f:
            plot_results(json.load(f), args.out_dir)
        return

    device = torch.device(args.device)
    cams, pids, joints, confs, L = load_scene_sequences(
        args.scenes_root / args.scene, device
    )
    K = len(cams)
    n_pairs = K * (K - 1) // 2
    logger.info(f"Scene {args.scene}: {K} cams {cams}, persons {pids}, "
                f"{L} frames, {n_pairs} camera pairs, device={args.device}")

    if 2 * args.max_shift + args.t_max > L:
        raise ValueError(
            f"2*max_shift + t_max = {2 * args.max_shift + args.t_max} exceeds the "
            f"scene's {L} frames — lower --t-max or --max-shift"
        )

    # warm-up (CUDA init / kernel compilation), untimed
    rng = np.random.default_rng(args.seed)
    run_once(joints, confs, cams, args.t_min, args.max_shift, rng, device)

    results = {
        "scene": args.scene, "cams": cams, "pids": pids, "n_pairs": n_pairs,
        "max_shift": args.max_shift, "repeats": args.repeats, "seed": args.seed,
        "device": args.device, "sweep": [],
    }
    ts = list(range(args.t_min, args.t_max + 1, args.t_step))
    for T in ts:
        times, errs = [], []
        for r in range(args.repeats):
            rng = np.random.default_rng(args.seed + 1000 * T + r)
            s_per_pair, err, _ = run_once(
                joints, confs, cams, T, args.max_shift, rng, device
            )
            times.append(s_per_pair)
            errs.append(err)
        entry = {
            "T": T,
            "mean_s_per_pair": float(np.mean(times)),
            "std_s_per_pair": float(np.std(times)),
            "times_s_per_pair": [float(t) for t in times],
            "mean_abs_error_frames": float(np.mean(errs)),
            "errors_frames": [float(e) for e in errs],
        }
        results["sweep"].append(entry)
        with open(json_path, "w") as f:      # incremental: partial sweeps survive
            json.dump(results, f, indent=2)
        logger.info(f"T={T:5d}  {entry['mean_s_per_pair']:.3f} ± "
                    f"{entry['std_s_per_pair']:.3f} s/pair   "
                    f"sync err {entry['mean_abs_error_frames']:.2f} fr")

    plot_results(results, args.out_dir)

    print("\n" + "=" * 58)
    print(f"SYNC SCALING — {args.scene}  ({K} cams, {n_pairs} pairs, "
          f"±{args.max_shift} fr shifts, {args.repeats} repeats)")
    print("=" * 58)
    print(f"  {'T':>6} {'s/pair':>10} {'±':>8} {'err (fr)':>10}")
    for e in results["sweep"]:
        print(f"  {e['T']:>6} {e['mean_s_per_pair']:>10.3f} "
              f"{e['std_s_per_pair']:>8.3f} {e['mean_abs_error_frames']:>10.2f}")
    print("=" * 58)
    logger.info(f"Results saved → {json_path}")


if __name__ == "__main__":
    main()
