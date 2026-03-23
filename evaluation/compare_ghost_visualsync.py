"""
compare_ghost_visualsync.py

Same random-shift alignment experiment as alignment_experiments_multi.py,
but compares Ghost (pose-based DTW) against VisualSync (visual, cross-view
object motion) on the same scenes and the same random shifts.

Ghost  — runs inline using Synchronizer on SMPL-X body pose sequences.
VisualSync — called as subprocess steps (must run from the visualsync conda env):
  Step 1: run_dino_sam2.py    (dynamic object segmentation, once per scene)
  Step 2: DEVA tracking        (once per scene, requires submodule init)
  Step 3: vggt_to_colmap.py   (camera poses, once per scene)
  Step 4: TODO sync.py         (synchronization — script not in repo yet)

Prerequisites for VisualSync steps to actually run:
  1. conda env 'visualsync' installed  (bash visualsync/scripts/install.sh)
  2. git submodule update --init --recursive  (inside visualsync/)
  3. bash visualsync/scripts/download_weights.sh
  4. sync.py added to visualsync/ (core sync algorithm, not in repo yet)

Usage (GPU node):
    pixi run python -m evaluation.compare_ghost_visualsync
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from synchronize_videos.synchronizer import Synchronizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GHOST_ROOT      = Path("/cluster/project/cvg/students/tnanni/ghost")
VISUALSYNC_ROOT = Path("/cluster/project/cvg/students/tnanni/visualsync")
VISUALSYNC_DATA = VISUALSYNC_ROOT / "data" / "rich"        # flat: <token>_cam_*/rgb/
VISUALSYNC_SCENE_DATA = VISUALSYNC_ROOT / "data" / "rich_by_scene"  # grouped: <token>/<token>_cam_*/

# Ghost uses rich10_segmentation_test (rich9 doesn't exist)
SCENES_ROOT = GHOST_ROOT / "test_outputs" / "rich10_segmentation_test"

SKIP_SCENES: list[str] = [
    "BBQ_001_juggle",
    "Pavallion_003_018_tossball",
]

N_TRIALS  = 10
MAX_SHIFT = 100
SEED      = 42
VERBOSE   = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VISUALSYNC_PYTHON = str(
    Path.home() / ".." / ".." / "cluster/project/cvg/students/tnanni/conda_envs"
    / "visualsync" / "bin" / "python"
)
# Fall back to system python if env not ready yet
if not Path(VISUALSYNC_PYTHON).exists():
    VISUALSYNC_PYTHON = sys.executable

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: RICH ↔ VisualSync name mapping
# ---------------------------------------------------------------------------

def rich_to_token(rich_scene: str) -> str:
    """'BBQ_001_guitar' → 'BBQ001guitar'  (no underscores → single scene token)."""
    return rich_scene.replace("_", "")


def cam_dirs_for_scene(scene: str) -> list[Path]:
    """Sorted VisualSync per-camera directories under data/rich/ for a RICH scene."""
    token = rich_to_token(scene)
    return sorted(VISUALSYNC_DATA.glob(f"{token}_*/"))


# ---------------------------------------------------------------------------
# VisualSync: per-scene grouped workdir
# ---------------------------------------------------------------------------

def make_scene_workdir(scene: str) -> Path:
    """Create (once) data/rich_by_scene/<token>/ with symlinks to each camera dir.

    VisualSync preprocessing scripts expect:
      <workdir>/<view_dir>/rgb/*.jpg
    where all <view_dir>s in the same workdir belong to one scene.
    """
    token = rich_to_token(scene)
    scene_wd = VISUALSYNC_SCENE_DATA / token
    scene_wd.mkdir(parents=True, exist_ok=True)

    for cam_dir in cam_dirs_for_scene(scene):
        link = scene_wd / cam_dir.name
        if not link.exists():
            link.symlink_to(cam_dir.resolve())

    return scene_wd


# ---------------------------------------------------------------------------
# VisualSync: preprocessing (once per scene)
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path) -> None:
    logger.info(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def preprocess_visualsync_scene(scene: str, scene_wd: Path) -> bool:
    """Run VisualSync preprocessing for a scene; skip steps already cached.

    Returns False if a required binary/script is missing so the caller can
    skip VisualSync for this scene gracefully.
    """
    token = rich_to_token(scene)

    # Quick sanity: conda env must exist
    if not Path(VISUALSYNC_PYTHON).exists():
        logger.warning("  [VisualSync] conda env not ready — skipping preprocessing.")
        return False

    # ── Step 1 & 2: segmentation + DEVA tracking ───────────────────────────
    deva_marker = scene_wd / f"{token}_cam_00" / "deva"
    if not deva_marker.exists():
        dino_sam2 = VISUALSYNC_ROOT / "preprocess" / "run_dino_sam2.py"
        logger.info(f"  [VisualSync] DINO+SAM2 segmentation for {scene}…")
        _run([VISUALSYNC_PYTHON, dino_sam2, "--workdir", scene_wd], VISUALSYNC_ROOT)

        deva_eval = (
            VISUALSYNC_ROOT
            / "Tracking-Anything-with-DEVA"
            / "evaluation"
            / "eval_with_detections.py"
        )
        if not deva_eval.exists():
            logger.warning(
                "  [VisualSync] DEVA submodule not initialised "
                "(run: git submodule update --init --recursive) — skipping."
            )
            return False

        logger.info(f"  [VisualSync] DEVA tracking for {scene}…")
        _run(
            [VISUALSYNC_PYTHON, deva_eval,
             "--workdir", scene_wd,
             "--max_missed_detection_count", "9000",
             "--output-dir", "deva"],
            VISUALSYNC_ROOT / "Tracking-Anything-with-DEVA",
        )
    else:
        logger.info(f"  [VisualSync] Segmentation/tracking cached for {scene}.")

    # ── Step 3: VGGT camera poses ────────────────────────────────────────────
    vggt_marker = scene_wd / f"{token}_cam_00" / "vggt"
    if not vggt_marker.exists():
        vggt_script = VISUALSYNC_ROOT / "preprocess" / "vggt_to_colmap.py"
        vis_path    = VISUALSYNC_ROOT / "data" / "rich_vggt_vis" / token
        vis_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  [VisualSync] VGGT pose estimation for {scene}…")
        _run(
            [VISUALSYNC_PYTHON, vggt_script,
             "--workdir", scene_wd,
             "--vis_path", vis_path],
            VISUALSYNC_ROOT,
        )
    else:
        logger.info(f"  [VisualSync] VGGT cached for {scene}.")

    return True


# ---------------------------------------------------------------------------
# VisualSync: shifted workdir for one trial
# ---------------------------------------------------------------------------

def build_shifted_workdir(scene: str, shifts: dict[str, int], base_tmp: Path) -> Path:
    """Symlink the shifted frame window for each camera into a temp workdir.

    Ghost selects frames [max_s - s : T_base] per camera.
    We mirror that here: each camera's rgb/ dir contains only those frames.
    Preprocessed outputs (deva/, vggt/) are symlinked as full sequences —
    sync.py is expected to receive the per-camera start index via the shifts
    it estimates, or accept a --frame-range argument.
    """
    token    = rich_to_token(scene)
    cam_dirs = cam_dirs_for_scene(scene)
    cam_ids  = [d.name[len(token) + 1:] for d in cam_dirs]  # strip "<token>_"

    T_base = min(
        len(list((VISUALSYNC_DATA / f"{token}_{c}" / "rgb").glob("*.jpg")))
        for c in cam_ids
    )
    max_s = max(shifts.get(c, 0) for c in cam_ids)

    shifted_wd = base_tmp / token
    shifted_wd.mkdir(parents=True, exist_ok=True)

    for cam_id in cam_ids:
        src = VISUALSYNC_DATA / f"{token}_{cam_id}"
        dst = shifted_wd / f"{token}_{cam_id}"
        dst_rgb = dst / "rgb"
        dst_rgb.mkdir(parents=True, exist_ok=True)

        all_frames = sorted((src / "rgb").glob("*.jpg"))
        s = max_s - shifts.get(cam_id, 0)
        for frame in all_frames[s:T_base]:
            link = dst_rgb / frame.name
            if not link.exists():
                link.symlink_to(frame.resolve())

        # Symlink precomputed preprocessing outputs
        for sub in ("deva", "vggt", "cotracker"):
            src_sub = src / sub
            if src_sub.exists():
                dst_sub = dst / sub
                if not dst_sub.exists():
                    dst_sub.symlink_to(src_sub.resolve())

    return shifted_wd


# ---------------------------------------------------------------------------
# VisualSync: run sync for one trial
# ---------------------------------------------------------------------------

def run_visualsync_trial(
    scene: str,
    true_shifts: dict[str, int],
    base_tmp: Path,
) -> dict | None:
    sync_script = VISUALSYNC_ROOT / "sync.py"
    if not sync_script.exists():
        # TODO: add sync.py to visualsync/ when the core algorithm is released.
        # It should accept:
        #   --workdir <shifted_workdir>   (contains per-camera rgb/ + deva/ + vggt/)
        #   --output  <result_json>
        # and write JSON: {"offsets": {"cam_00": <float>, "cam_01": <float>, ...}}
        return None

    trial_key  = hash(frozenset(true_shifts.items())) & 0xFFFFFF
    trial_tmp  = base_tmp / f"trial_{trial_key:06x}"
    trial_tmp.mkdir(parents=True, exist_ok=True)

    shifted_wd  = build_shifted_workdir(scene, true_shifts, trial_tmp)
    result_path = trial_tmp / "sync_result.json"

    _run(
        [VISUALSYNC_PYTHON, sync_script,
         "--workdir", shifted_wd,
         "--output",  result_path],
        VISUALSYNC_ROOT,
    )

    if not result_path.exists():
        logger.warning("  [VisualSync] sync_result.json not produced.")
        return None

    with open(result_path) as f:
        raw = json.load(f)

    cam_ids   = list(true_shifts.keys())
    offsets   = raw["offsets"]
    estimated = torch.tensor([float(offsets.get(c, 0.0)) for c in cam_ids])
    true_t    = torch.tensor([float(true_shifts[c]) for c in cam_ids])
    true_t   -= true_t.min()
    estimated -= estimated.min()

    errors = (estimated - true_t).abs()
    return {
        "true_times": true_t.tolist(),
        "estimated":  estimated.tolist(),
        "errors":     errors.tolist(),
        "mae":        errors.mean().item(),
        "within_1":   (errors <= 1).float().mean().item(),
        "within_2":   (errors <= 2).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Ghost: load + run  (mirrors alignment_experiments_multi.py exactly)
# ---------------------------------------------------------------------------

def load_ghost_scene(
    scene_dir: Path,
) -> dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    for cam_dir in sorted(scene_dir.iterdir()):
        body_dir = cam_dir / "body_data"
        if not cam_dir.is_dir() or not body_dir.exists():
            continue
        persons: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for npz_path in sorted(body_dir.glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            with np.load(str(npz_path)) as d:
                required = {
                    "smplx_body_pose", "smplx_left_hand_pose",
                    "smplx_right_hand_pose", "pred_joint_confidence",
                }
                if not required.issubset(d.files):
                    continue
                pose = np.concatenate(
                    [d["smplx_body_pose"], d["smplx_left_hand_pose"],
                     d["smplx_right_hand_pose"]], axis=1,
                )
                rotations = torch.from_numpy(pose.astype(np.float32)).reshape(-1, 51, 3)
                conf = torch.from_numpy(
                    d["pred_joint_confidence"][:, 1:52].astype(np.float32)
                )
            persons[pid] = (rotations, conf)
        if persons:
            cam_data[cam_dir.name] = persons
    return cam_data


def common_persons(cam_data: dict) -> list[int]:
    sets = [set(p.keys()) for p in cam_data.values()]
    return sorted(set.intersection(*sets))


def apply_shifts_ghost(cam_data, shifts, pids):
    cam_ids = list(shifts.keys())
    T_base  = min(cam_data[c][p][0].shape[0] for c in cam_ids for p in pids)
    max_s   = max(shifts.values())
    joints_list, confs_list = [], []
    for cam_id in cam_ids:
        s = max_s - shifts[cam_id]
        per_j, per_c = [], []
        for pid in pids:
            rot, conf = cam_data[cam_id][pid]
            per_j.append(rot[s:T_base].to(DEVICE))
            per_c.append(conf[s:T_base].to(DEVICE))
        joints_list.append(per_j)
        confs_list.append(per_c)
    return joints_list, confs_list


def run_ghost_trial(cam_data, pids, true_shifts, sync: Synchronizer) -> dict:
    cam_ids    = list(true_shifts.keys())
    jl, cl     = apply_shifts_ghost(cam_data, true_shifts, pids)
    offset_mat = sync.estimate_offset_matrix(jl, cl)
    weights    = sync.cycle_consistency_weights(offset_mat)
    estimated  = sync.estimate_initial_times(offset_mat, weights)
    true_t     = torch.tensor([true_shifts[c] for c in cam_ids], dtype=torch.float32)
    true_t    -= true_t.min()
    errors     = (estimated.cpu() - true_t).abs()
    return {
        "true_times": true_t.tolist(),
        "estimated":  estimated.cpu().tolist(),
        "errors":     errors.tolist(),
        "mae":        errors.mean().item(),
        "within_1":   (errors <= 1).float().mean().item(),
        "within_2":   (errors <= 2).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Per-scene experiment
# ---------------------------------------------------------------------------

def _summarise(results: list[dict], method: str) -> dict | None:
    if not results:
        return None
    return {
        "method":     method,
        "mae_mean":   float(np.mean([r["mae"]      for r in results])),
        "mae_median": float(np.median([r["mae"]    for r in results])),
        "mae_max":    float(np.max([r["mae"]       for r in results])),
        "within_1":   float(np.mean([r["within_1"] for r in results])),
        "within_2":   float(np.mean([r["within_2"] for r in results])),
    }


def run_scene(
    scene_dir: Path,
    sync: Synchronizer,
    rng: np.random.Generator,
    tmp_root: Path,
) -> dict | None:
    scene = scene_dir.name
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Scene: {scene}")

    cam_data = load_ghost_scene(scene_dir)
    if len(cam_data) < 2:
        logger.warning(f"  Skipping: need ≥2 cameras, found {len(cam_data)}")
        return None
    pids = common_persons(cam_data)
    if not pids:
        logger.warning("  Skipping: no common person ID across cameras.")
        return None

    cam_ids = list(cam_data.keys())
    logger.info(f"  Cameras: {cam_ids}  |  common persons: {pids}")

    # VisualSync: create grouped workdir and preprocess (once)
    scene_wd             = make_scene_workdir(scene)
    visualsync_ready     = preprocess_visualsync_scene(scene, scene_wd)
    visualsync_has_sync  = (VISUALSYNC_ROOT / "sync.py").exists()
    if not visualsync_has_sync:
        logger.warning("  [VisualSync] sync.py not found — will record N/A for VisualSync.")

    ghost_results, vsync_results = [], []

    for trial in range(N_TRIALS):
        raw_shifts  = [0] + rng.integers(
            -MAX_SHIFT, MAX_SHIFT + 1, size=len(cam_ids) - 1
        ).tolist()
        true_shifts = {c: int(s) for c, s in zip(cam_ids, raw_shifts)}
        logger.info(f"\n  ── Trial {trial + 1}/{N_TRIALS}  shifts: {true_shifts}")

        g = run_ghost_trial(cam_data, pids, true_shifts, sync)
        ghost_results.append(g)
        logger.info(
            f"     [Ghost]      MAE={g['mae']:.2f}  "
            f"W-1={g['within_1']*100:.0f}%  W-2={g['within_2']*100:.0f}%"
        )

        if visualsync_ready and visualsync_has_sync:
            v = run_visualsync_trial(scene, true_shifts, tmp_root)
            if v is not None:
                vsync_results.append(v)
                logger.info(
                    f"     [VisualSync] MAE={v['mae']:.2f}  "
                    f"W-1={v['within_1']*100:.0f}%  W-2={v['within_2']*100:.0f}%"
                )

    g_sum = _summarise(ghost_results, "Ghost")
    v_sum = _summarise(vsync_results, "VisualSync")

    logger.info(f"\n  SUMMARY — {scene}  ({N_TRIALS} trials)")
    for s in [g_sum, v_sum]:
        if s:
            logger.info(
                f"    [{s['method']:<12}]  MAE mean={s['mae_mean']:.2f}  "
                f"W-1={s['within_1']*100:.1f}%  W-2={s['within_2']*100:.1f}%"
            )

    return {
        "scene":      scene,
        "n_cameras":  len(cam_ids),
        "n_persons":  len(pids),
        "ghost":      g_sum,
        "visualsync": v_sum,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Scenes root:     {SCENES_ROOT}")
    logger.info(f"VisualSync data: {VISUALSYNC_DATA}")
    logger.info(f"Device: {DEVICE}  |  trials: {N_TRIALS}  |  max_shift: {MAX_SHIFT}")
    if SKIP_SCENES:
        logger.info(f"Skipping: {SKIP_SCENES}")

    scene_dirs = sorted(
        d for d in SCENES_ROOT.iterdir()
        if d.is_dir() and d.name not in SKIP_SCENES
    )
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found under {SCENES_ROOT}")
    logger.info(f"Found {len(scene_dirs)} scenes: {[d.name for d in scene_dirs]}")

    sync = Synchronizer(device=DEVICE)
    rng  = np.random.default_rng(SEED)

    with tempfile.TemporaryDirectory(prefix="vsync_compare_") as tmp:
        tmp_root  = Path(tmp)
        summaries = []
        for scene_dir in scene_dirs:
            s = run_scene(scene_dir, sync, rng, tmp_root)
            if s is not None:
                summaries.append(s)

    if not summaries:
        logger.error("No usable scenes — nothing to summarise.")
        sys.exit(1)

    # Aggregate table
    logger.info("\n" + "=" * 80)
    logger.info(
        f"AGGREGATE  ({len(summaries)} scenes × {N_TRIALS} trials)"
    )
    hdr = f"  {'Scene':<32}  {'Cams':>4}  {'Method':<12}  {'MAE':>8}  {'W-1':>7}  {'W-2':>7}"
    sep = "  " + "-" * (len(hdr) - 2)
    logger.info(hdr)
    logger.info(sep)

    agg: dict[str, list] = {"Ghost": [], "VisualSync": []}

    for s in summaries:
        for key, method in [("ghost", "Ghost"), ("visualsync", "VisualSync")]:
            m = s[key]
            if m is None:
                continue
            logger.info(
                f"  {s['scene']:<32}  {s['n_cameras']:>4}  {m['method']:<12}  "
                f"{m['mae_mean']:>8.2f}  {m['within_1']*100:>6.1f}%  {m['within_2']*100:>6.1f}%"
            )
            agg[method].append(m)

    logger.info(sep)
    for method, rows in agg.items():
        if rows:
            logger.info(
                f"  {'MEAN':<32}  {'':>4}  {method:<12}  "
                f"{np.mean([r['mae_mean'] for r in rows]):>8.2f}  "
                f"{np.mean([r['within_1'] for r in rows])*100:>6.1f}%  "
                f"{np.mean([r['within_2'] for r in rows])*100:>6.1f}%"
            )
