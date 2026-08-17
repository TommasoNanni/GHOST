"""Runtime benchmark: per-stage wall-clock of the full ghost pipeline on one scene.

Measures, on a single GPU, the runtime of every pipeline stage on an EgoHumans
fencing scene (3 people, 4 exo cameras) with randomly desynchronised camera
streams:

    (untimed)  1. load the scene's frame lists
    (untimed)  2. apply random per-camera offsets (async 60-frame windows)
    [clock 1]  3. SAM3 segmentation                (own timed step, dummy output)
    [clock 2]  4. SAM3D body parameter estimation  (dummy output, then discarded)
    [clock 3]  5. synchronization (DTW offset matrix + global solve)
    [clock 4]  6. VGGT cameras/depth + MapAnything metric scale
    [clock 5]  7. fusion: geodesic median pose + mean-betas shape averaging
    [clock 6]  8. BodyPlacer (Procrustes DLT root translation + orientation)

Each stage's clock includes its model loading and teardown: a stage's clock ends
exactly when the next stage's clock starts. A global clock spans stages 3-8.

The SAM3/SAM3D outputs land in a throw-away directory; the downstream stages
(sync, fusion, placer) read the curated body_data_clean/ tracks of the scene,
sliced to the same asynchronous windows, so every stage runs on well-defined
inputs regardless of what the dummy segmentation produced. VGGT + MapAnything
run on the asynchronous images themselves.

Reconstruction happens on the temporal INTERSECTION of the aligned streams;
inside it, pose fusion uses every camera that sees a person (>=1) and placement
triangulates wherever >=2 cameras have 2D keypoints (NaN elsewhere).

Usage (single GPU):
    pixi run python scripts/runtime_benchmark.py
    sbatch bash_jobs/runtime_benchmark.sh
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.append(str(_REPO_ROOT / "MHR" / "tools" / "mhr_smpl_conversion"))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
import torch

from configuration import CONFIG
from data.video_dataset import Video, EgoHumansScene

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Deep inner path baked into every EgoHumans activity folder (as in
# scripts/egohumans_pipeline.py).
_INNER = Path("media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# 03_fencing exo cameras kept by the pipeline, and the body_data_clean person
# ids of 005_fencing (3 people) — same as alignment_experiments_multi_egohumans.
FENCING_CAMS = ["cam04", "cam05", "cam10", "cam13"]
PIDS = [1, 2, 3]

_NUM_JOINTS = 55   # SMPL-X joints packed before the root is dropped


# ── fusion helpers ───────────────────────────────────────────────────────────
# Copied verbatim from scripts/infer_scene.py (itself copied from
# evaluation/evaluate_rich_median.py) so this script stands alone.

def _sixd_to_matrix(sixd: torch.Tensor) -> torch.Tensor:
    """(..., 6) 6D rotation → (..., 3, 3) rotation matrix (Gram-Schmidt)."""
    r0, r1 = sixd[..., :3], sixd[..., 3:]
    b1 = r0 / (r0.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def median_fuse(pose_t: torch.Tensor, mask_t: torch.Tensor,
                iters: int = 5, eps: float = 1e-3) -> torch.Tensor:
    """Fuse per-camera poses by the geodesic median over the visible cameras.

    pose_t: (B, T, K, P, J, 6);  mask_t: (B, T, K, P).  Returns (B, T, P, J, 6).
    See evaluation/evaluate_rich_median.py for the full derivation.
    """
    R = _sixd_to_matrix(pose_t)                             # (B,T,K,P,J,3,3)
    J = R.shape[4]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    empty = (mask_t.sum(dim=2) == 0)[..., None]             # (B,T,P,1)

    def _chordal(w: torch.Tensor) -> torch.Tensor:
        ww = w[..., None, None]
        M = (R * ww).sum(dim=2) / ww.sum(dim=2).clamp_min(1e-8)
        if bool(empty.any()):
            M = torch.where(empty[..., None, None], eye.expand_as(M), M)
        U, _, Vh = torch.linalg.svd(M)
        d = torch.linalg.det(U @ Vh)
        D = eye.expand(*d.shape, 3, 3).clone()
        D[..., 2, 2] = d
        return U @ D @ Vh

    w0 = mask_t[..., None].expand(*mask_t.shape, J)
    R_bar = _chordal(w0)
    for _ in range(iters):
        rel = R @ R_bar[:, :, None].transpose(-1, -2)
        cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
        theta = torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        R_bar = _chordal(w0 / (theta + eps))

    return torch.cat([R_bar[..., 0, :], R_bar[..., 1, :]], dim=-1)


def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """(J,3) axis-angle → (J,6) = first two ROWS of R (training convention)."""
    from scipy.spatial.transform import Rotation as SciR
    Rm = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()
    return Rm[:, :2, :].reshape(-1, 6)


def _pack_pose_54(go: np.ndarray, bp: np.ndarray,
                  lh: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """SMPL-X aa params of one frame → (54, 6) root-excluded 6-D pose.
    Copied from scripts/infer_scene.py:_egohumans_pack_pose."""
    aa = np.concatenate([
        go.reshape(1, 3), bp.reshape(21, 3),
        lh.reshape(15, 3), rh.reshape(15, 3),
    ], 0).astype(np.float64)
    if aa.shape[0] < _NUM_JOINTS:
        aa = np.concatenate([aa, np.zeros((_NUM_JOINTS - aa.shape[0], 3))], 0)
    return _aa_to_6d(aa)[1:]


# ── timing ───────────────────────────────────────────────────────────────────

class GpuSampler(threading.Thread):
    """Background nvidia-smi poller: whole-GPU memory/utilisation samples.

    Process-agnostic — catches the SAM3D worker subprocesses that parent-side
    torch.cuda peak stats cannot see. Timestamps share time.perf_counter with
    StageClock so samples can be attributed to stages by boundary."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.idx = device_index
        self.dt = interval_s
        self.samples: list[tuple[float, float, float]] = []  # (t, mem MiB, util %)
        # NB: name must not be `_stop` — threading.Thread uses that internally.
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.idx}",
                     "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                mem, util = (float(x) for x in out.split(","))
                self.samples.append((time.perf_counter(), mem, util))
            except Exception:
                pass
            self._stop_evt.wait(self.dt)

    def stop(self) -> None:
        self._stop_evt.set()

    def stats_between(self, t0: float, t1: float) -> dict | None:
        s = [(m, u) for t, m, u in self.samples if t0 <= t <= t1]
        if not s:
            return None
        mems = [m for m, _ in s]
        utils = [u for _, u in s]
        return {
            "gpu_mem_peak_mib": max(mems),
            "gpu_mem_mean_mib": sum(mems) / len(mems),
            "gpu_util_peak_pct": max(utils),
            "gpu_util_mean_pct": sum(utils) / len(utils),
            "n_samples": len(s),
        }


class StageClock:
    """Boundary-based stage timer: a stage ends exactly when the next starts.

    Also snapshots per-stage torch.cuda peak memory (parent process only — the
    SAM3D workers are separate processes, covered by GpuSampler instead)."""

    def __init__(self) -> None:
        self.marks: list[tuple[str, float]] = []
        self.torch_stats: dict[str, dict[str, float]] = {}

    def start(self) -> None:
        self._sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.marks.append(("__start__", time.perf_counter()))

    def mark(self, name: str) -> None:
        """Close the stage named `name` (started at the previous mark)."""
        self._sync()
        if torch.cuda.is_available():
            self.torch_stats[name] = {
                "torch_max_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
                "torch_max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            }
            torch.cuda.reset_peak_memory_stats()
        self.marks.append((name, time.perf_counter()))
        extra = ""
        if name in self.torch_stats:
            extra = (f"  (torch peak alloc "
                     f"{self.torch_stats[name]['torch_max_alloc_mib']:.0f} MiB)")
        logger.info(f"  [clock] {name}: {self.durations()[name]:.1f} s{extra}")

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def durations(self) -> dict[str, float]:
        return {
            name: t - self.marks[i - 1][1]
            for i, (name, t) in enumerate(self.marks) if i > 0
        }

    @property
    def total(self) -> float:
        return self.marks[-1][1] - self.marks[0][1]


def _free_gpu(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── setup helpers (untimed) ──────────────────────────────────────────────────

def _stem_num(p: Path) -> int:
    return int("".join(c for c in p.stem if c.isdigit()))


def _frames_dir(seq_dir: Path, cam: str) -> Path:
    """Resized frames dir the pipeline consumed (originals deleted after resize)."""
    d = seq_dir / "exo" / cam / "images_undistorted" / "frames"
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} not found — expected the resized frames/ subdir the pipeline "
            f"ran on (body_data_clean kp2d live in that image space)."
        )
    return d


def _load_clean_npzs(clean_scene: Path, cams: list[str]) -> dict[str, dict[int, dict]]:
    """{cam: {pid: full npz dict + 'row_by_stem' map}} from body_data_clean/."""
    out: dict[str, dict[int, dict]] = {}
    for cam in cams:
        out[cam] = {}
        for pid in PIDS:
            f = clean_scene / cam / "body_data_clean" / f"person_{pid}.npz"
            if not f.exists():
                continue
            d = dict(np.load(str(f), allow_pickle=False))
            fi = d["frame_indices"].astype(int)
            d["row_by_stem"] = {int(s): t for t, s in enumerate(fi)}
            out[cam][pid] = d
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path,
                        default=Path("/capstor/scratch/cscs/tnanni/datasets/egohumans"))
    parser.add_argument("--scenes-root", type=Path,
                        default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new"),
                        help="Processed outputs root holding <scene>/cam/body_data_clean")
    parser.add_argument("--scene", type=str, default="03_fencing/005_fencing")
    parser.add_argument("--work-dir", type=Path,
                        default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/runtime_benchmark"),
                        help="Dummy output root — WIPED at every run")
    parser.add_argument("--frames", type=int, default=60, help="Frames per camera")
    parser.add_argument("--max-shift", type=int, default=10,
                        help="Max |random offset| in frames (±)")
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Base window start position (default: max-shift)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    activity, seq_name = args.scene.split("/")
    seq_dir = args.data_root / activity / _INNER / activity / seq_name
    clean_scene = args.scenes_root / args.scene
    cams = FENCING_CAMS if activity == "03_fencing" else sorted(
        d.name for d in clean_scene.iterdir()
        if d.is_dir() and (d / "body_data_clean").is_dir()
    )

    # Dummy work dir: wiped every run so no stage can skip-if-done.
    assert "runtime_benchmark" in str(args.work_dir), "refusing to wipe non-dummy dir"
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)

    # ── (untimed) 1. load frame lists ────────────────────────────────────────
    frames_all: dict[str, list[Path]] = {}
    for cam in cams:
        fdir = _frames_dir(seq_dir, cam)
        frames_all[cam] = sorted(
            p for p in fdir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )
        if not frames_all[cam]:
            raise FileNotFoundError(f"no images in {fdir}")

    # ── (untimed) 2. random per-camera offsets → async windows ───────────────
    rng = np.random.default_rng(args.seed)
    shifts = {cams[0]: 0}
    for cam in cams[1:]:
        shifts[cam] = int(rng.integers(-args.max_shift, args.max_shift + 1))
    base_pos = args.start_pos if args.start_pos is not None else args.max_shift

    window_files: dict[str, list[Path]] = {}
    window_stems: dict[str, list[int]] = {}
    async_root = args.work_dir / "frames_async"
    for cam in cams:
        pos = base_pos + shifts[cam]
        if pos < 0 or pos + args.frames > len(frames_all[cam]):
            raise ValueError(f"{cam}: window [{pos}, {pos + args.frames}) out of "
                             f"range (have {len(frames_all[cam])} frames)")
        files = frames_all[cam][pos:pos + args.frames]
        window_files[cam] = files
        window_stems[cam] = [_stem_num(p) for p in files]
        # Symlink farm: each camera's async 60-frame stream as its own dir, so
        # Video / VGGT / MapAnything all see exactly these frames and no others.
        d = async_root / cam
        d.mkdir(parents=True)
        for f in files:
            (d / f.name).symlink_to(f)

    logger.info(f"Scene {args.scene}: {len(cams)} cams, {args.frames} frames/cam, "
                f"persons {PIDS}")
    logger.info(f"Injected shifts (frames): {shifts}")

    scene_dummy = args.work_dir / "scene_dummy"     # VGGT/MA outputs + placer input
    scene_dummy.mkdir()

    # GPU sampler: physical device index (nvidia-smi ignores CUDA_VISIBLE_DEVICES,
    # so map the logical cuda index through it when set).
    cuda_idx = device.index or 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys_idx = int(cvd.split(",")[cuda_idx]) if cvd else cuda_idx
    sampler = GpuSampler(device_index=phys_idx, interval_s=0.5)
    sampler.start()

    clock = StageClock()
    clock.start()

    # ── [clock 1] SAM3 segmentation (dummy output) ───────────────────────────
    logger.info("── Stage 1: SAM3 segmentation ──")
    from preprocessing.segmentation import PersonSegmenter
    videos = []
    for cam in cams:
        v = Video(frames_dir=async_root / cam)
        v.video_id = cam
        videos.append(v)
    scene_obj = EgoHumansScene(scene_id="runtime", videos=videos, seq_dir=seq_dir)
    segmenter = PersonSegmenter(
        checkpoint_path=CONFIG.segmentation.checkpoint_path,
        text_prompt=CONFIG.segmentation.text_prompt,
        redetect_interval=CONFIG.segmentation.redetect_interval,
        new_det_thresh=CONFIG.segmentation.new_det_thresh,
        score_threshold_detection=CONFIG.segmentation.score_threshold_detection,
    )
    video_dirs = segmenter.segment_scene(
        scene=scene_obj, output_dir=args.work_dir / "pipeline_dummy", vis=False
    )
    _free_gpu(segmenter)
    clock.mark("segmentation")

    # ── [clock 2] SAM3D body parameter estimation (dummy output) ─────────────
    logger.info("── Stage 2: SAM3D body estimation ──")
    from preprocessing.parameters_extraction_v2 import ParametersExtractor
    estimator = ParametersExtractor(
        sam3d_hf_repo=CONFIG.parameters_extraction.sam3d_id,
        sam3d_step=CONFIG.parameters_extraction.sam3d_step,
        bbox_padding=CONFIG.parameters_extraction.bbox_padding,
        smplx_model_path=CONFIG.data.smplx_model_path,
        mhr_model_path=CONFIG.data.mhr_model_path,
        reid_threshold=CONFIG.parameters_extraction.reid_threshold,
        reid_match_window=getattr(CONFIG.parameters_extraction, "reid_match_window", 5),
        rich_data_root=None,
    )
    estimator.estimate_scene(scene=scene_obj, video_dirs=video_dirs)
    _free_gpu(estimator)
    clock.mark("sam3d")

    # ── [clock 3] synchronization ────────────────────────────────────────────
    # Downstream stages read the curated body_data_clean tracks (globally
    # consistent person ids), sliced to the SAME async windows as the images.
    logger.info("── Stage 3: synchronization ──")
    from synchronize_videos.synchronizer import Synchronizer
    clean = _load_clean_npzs(clean_scene, cams)

    joints_list: list[list[torch.Tensor]] = []
    confs_list: list[list[torch.Tensor]] = []
    for cam in cams:
        per_p_seq, per_p_conf = [], []
        for pid in PIDS:
            seq = np.zeros((args.frames, 51, 3), dtype=np.float32)
            conf = np.zeros((args.frames, 51), dtype=np.float32)
            d = clean[cam].get(pid)
            if d is not None:
                pose153 = np.concatenate([
                    d["smplx_body_pose"], d["smplx_left_hand_pose"],
                    d["smplx_right_hand_pose"],
                ], axis=1).astype(np.float32)              # (T_stored, 153)
                c51 = d["pred_joint_confidence"][:, 1:52].astype(np.float32)
                n_hit = 0
                for l, stem in enumerate(window_stems[cam]):
                    t = d["row_by_stem"].get(stem)
                    if t is not None:
                        seq[l] = pose153[t].reshape(51, 3)
                        conf[l] = c51[t]
                        n_hit += 1
                if n_hit == 0:
                    raise RuntimeError(
                        f"{cam}/person_{pid}: no body_data_clean rows inside the "
                        f"async window — frame_indices ↔ image-stem mapping broken?"
                    )
            per_p_seq.append(torch.from_numpy(seq).to(device))
            per_p_conf.append(torch.from_numpy(conf).to(device))
        joints_list.append(per_p_seq)
        confs_list.append(per_p_conf)

    sync = Synchronizer(
        use_acceleration_weights=False, device=str(device),
        min_overlap=max(20, args.frames - 3 * args.max_shift),
        max_shift=2 * args.max_shift, verbose=False,
    )
    offset_mat = sync.estimate_offset_matrix(joints_list, confs_list)
    weights = sync.cycle_consistency_weights(offset_mat)
    est_times = sync.estimate_initial_times(offset_mat, weights).cpu()
    est_anch = est_times - est_times.min()
    true_anch = np.array([shifts[c] for c in cams], dtype=np.float32)
    true_anch = true_anch - true_anch.min()
    clock.mark("sync")
    logger.info(f"  estimated start times: "
                f"{ {c: round(float(e), 1) for c, e in zip(cams, est_anch)} }")
    logger.info(f"  true start times:      "
                f"{ {c: float(t) for c, t in zip(cams, true_anch)} }")

    # ── [clock 4] VGGT + MapAnything on the async images ─────────────────────
    logger.info("── Stage 4: VGGT + MapAnything ──")
    from preprocessing.run_vggt import VGGTPreprocessor
    frame_paths = [
        [window_files[cam][t] for cam in cams] for t in range(args.frames)
    ]
    preprocessor = VGGTPreprocessor(
        weights=CONFIG.data.vggt_omega_checkpoint, device=str(device)
    )
    preprocessor.process_scene(
        frame_paths=frame_paths, camera_names=cams,
        output_dir=scene_dummy, devices=[str(device)],
    )
    _free_gpu(preprocessor)
    clock.mark("vggt")

    from preprocessing.run_mapanything import MapAnythingScaleEstimator
    # Camera-baseline ratio is the only scale protocol left, and what
    # BodyPlacer.load_mapanything_scale reads (mapanything_scale_baseline.npy).
    ma = MapAnythingScaleEstimator(device=str(device), batch_size=8)
    ma.process_scene(scene_dir=scene_dummy, img_root=async_root)
    _free_gpu(ma)
    clock.mark("mapanything")

    # ── [clock 5] fusion: geodesic median + shape averaging ──────────────────
    # Timeline = temporal intersection of the aligned streams. Within it, the
    # median runs over every camera that sees the person (>=1); placement later
    # needs >=2 (DLT) and leaves the rest NaN.
    logger.info("── Stage 5: fusion (geodesic median + mean betas) ──")
    inter = set(window_stems[cams[0]])
    for cam in cams[1:]:
        inter &= set(window_stems[cam])
    inter_stems = sorted(inter)
    if not inter_stems:
        raise RuntimeError("empty stream intersection — increase --frames or "
                           "decrease --max-shift")
    frame_start = inter_stems[0]
    T_int = len(inter_stems)
    K, P = len(cams), len(PIDS)
    logger.info(f"  intersection window: stems [{inter_stems[0]}, {inter_stems[-1]}]"
                f"  T={T_int}")

    pose = np.zeros((T_int, K, P, 54, 6), dtype=np.float32)
    mask = np.zeros((T_int, K, P), dtype=np.float32)
    betas_acc: dict[int, list[np.ndarray]] = {p: [] for p in PIDS}
    for k, cam in enumerate(cams):
        for s_idx, pid in enumerate(PIDS):
            d = clean[cam].get(pid)
            if d is None:
                continue
            for ti, stem in enumerate(inter_stems):
                t = d["row_by_stem"].get(stem)
                if t is None:
                    continue
                pose[ti, k, s_idx] = _pack_pose_54(
                    d["smplx_global_orient"][t], d["smplx_body_pose"][t],
                    d["smplx_left_hand_pose"][t], d["smplx_right_hand_pose"][t],
                )
                mask[ti, k, s_idx] = 1.0
                if "smplx_betas" in d:
                    betas_acc[pid].append(d["smplx_betas"][t][:10])

    with torch.no_grad():
        pt = torch.from_numpy(pose[None]).to(device)
        mt = torch.from_numpy(mask[None]).to(device)
        fused = median_fuse(pt, mt)                      # (1, T, P, 54, 6)
    fused_pose = fused[0].float().cpu().numpy().astype(np.float32)
    pred_shape = np.stack([
        (np.mean(betas_acc[p], 0) if betas_acc[p] else np.zeros(10, np.float32))
        for p in PIDS
    ]).astype(np.float32)                                # (P, 10) shape averaging
    clock.mark("fusion")

    # ── [clock 6] BodyPlacer (Procrustes DLT) ────────────────────────────────
    # scene_dummy gets cam/body_data → body_data_clean symlinks so the placer
    # finds VGGT outputs and body tracks under one root. raw rows are restricted
    # to the intersection stems. VGGT row t maps to each camera's own async
    # frame — off by <= the injected shift per camera; cameras are static, so
    # this slack does not change the placement cost.
    logger.info("── Stage 6: BodyPlacer ──")
    from inference import _run_placer
    for cam in cams:
        (scene_dummy / cam).mkdir(exist_ok=True)
        link = scene_dummy / cam / "body_data"
        if not link.exists():
            link.symlink_to(clean_scene / cam / "body_data_clean")
    cam_dirs = [scene_dummy / cam for cam in cams]

    inter_arr = np.array(inter_stems)
    raw_body: list[dict[int, dict]] = []
    for cam in cams:
        cam_persons: dict[int, dict] = {}
        for pid in PIDS:
            d = clean[cam].get(pid)
            if d is None:
                continue
            fi = d["frame_indices"].astype(int)
            m = np.isin(fi, inter_arr)
            if not m.any():
                continue
            cam_persons[pid] = {
                key: (arr[m] if isinstance(arr, np.ndarray)
                      and arr.ndim >= 1 and arr.shape[0] == len(fi) else arr)
                for key, arr in d.items() if key != "row_by_stem"
            }
        raw_body.append(cam_persons)

    root_translation, orient_R, vggt_cameras = _run_placer(
        scene_dir=scene_dummy,
        cam_dirs=cam_dirs,
        raw=raw_body,
        all_pids=PIDS,
        frame_start=frame_start,
        T=T_int,
        smplx_model_path=Path(CONFIG.data.smplx_model_path),
        fused_pose=fused_pose,
        crop_meta_path=None,
    )
    clock.mark("placer")

    n_placed = int(np.isfinite(root_translation[..., 0]).sum())
    logger.info(f"  placed {n_placed}/{T_int * P} (person, frame) slots "
                f"(rest <2-view or unseen)")

    # ── report ───────────────────────────────────────────────────────────────
    sampler.stop()
    sampler.join(timeout=5)
    dur = clock.durations()
    total = clock.total
    n_images = args.frames * K

    # Per-stage GPU stats by clock boundary. "4. VGGT + MapAnything" spans two
    # marks, so its window is stitched from both.
    mark_t = dict(clock.marks)
    t0 = mark_t["__start__"]
    stage_windows = {
        "segmentation": (t0, mark_t["segmentation"]),
        "sam3d":        (mark_t["segmentation"], mark_t["sam3d"]),
        "sync":         (mark_t["sam3d"], mark_t["sync"]),
        "vggt":         (mark_t["sync"], mark_t["vggt"]),
        "mapanything":  (mark_t["vggt"], mark_t["mapanything"]),
        "vggt+ma":      (mark_t["sync"], mark_t["mapanything"]),
        "fusion":       (mark_t["mapanything"], mark_t["fusion"]),
        "placer":       (mark_t["fusion"], mark_t["placer"]),
    }
    gpu = {k: sampler.stats_between(*w) for k, w in stage_windows.items()}
    gpu_all = sampler.stats_between(t0, mark_t["placer"])
    baseline = sampler.samples[0][1] if sampler.samples else float("nan")

    def _tstat(key: str, field: str) -> float:
        return clock.torch_stats.get(key, {}).get(field, float("nan"))

    def _g(key: str, field: str) -> float:
        st = gpu.get(key)
        return st[field] if st else float("nan")

    rows = [
        # (label, seconds, gpu-window key, torch-stats key)
        ("1. segmentation (SAM3)",     dur["segmentation"],             "segmentation", "segmentation"),
        ("2. SAM3D body params",       dur["sam3d"],                    "sam3d",        "sam3d"),
        ("3. synchronization",         dur["sync"],                     "sync",         "sync"),
        ("4. VGGT + MapAnything",      dur["vggt"] + dur["mapanything"], "vggt+ma",     "mapanything"),
        ("5. fusion (median + betas)", dur["fusion"],                   "fusion",       "fusion"),
        ("6. placer (Procrustes DLT)", dur["placer"],                   "placer",       "placer"),
    ]
    print("\n" + "=" * 110)
    print(f"RUNTIME — {args.scene}  "
          f"(T={args.frames} fr/cam, K={K} cams, P={P} persons, "
          f"{n_images} images, device={args.device})")
    print("=" * 110)
    print(f"  {'stage':<30} {'seconds':>9} {'share':>7} "
          f"{'gpu_peak':>9} {'gpu_mean':>9} {'util_peak':>9} {'util_mean':>9} "
          f"{'t_alloc':>8} {'t_resv':>8}")
    print(f"  {'':<30} {'':>9} {'':>7} "
          f"{'(MiB)':>9} {'(MiB)':>9} {'(%)':>9} {'(%)':>9} "
          f"{'(MiB)':>8} {'(MiB)':>8}")
    for name, s, gk, tk in rows:
        print(f"  {name:<30} {s:>9.1f} {s / total * 100:>6.1f}% "
              f"{_g(gk, 'gpu_mem_peak_mib'):>9.0f} {_g(gk, 'gpu_mem_mean_mib'):>9.0f} "
              f"{_g(gk, 'gpu_util_peak_pct'):>9.0f} {_g(gk, 'gpu_util_mean_pct'):>9.0f} "
              f"{_tstat(tk, 'torch_max_alloc_mib'):>8.0f} "
              f"{_tstat(tk, 'torch_max_reserved_mib'):>8.0f}")
    print("  " + "-" * 106)
    ga = gpu_all or {}
    print(f"  {'TOTAL (global clock)':<30} {total:>9.1f} {'100.0%':>7} "
          f"{ga.get('gpu_mem_peak_mib', float('nan')):>9.0f} "
          f"{ga.get('gpu_mem_mean_mib', float('nan')):>9.0f} "
          f"{ga.get('gpu_util_peak_pct', float('nan')):>9.0f} "
          f"{ga.get('gpu_util_mean_pct', float('nan')):>9.0f}")
    print(f"\n  4a. VGGT        {dur['vggt']:>8.1f} s   "
          f"gpu_peak {_g('vggt', 'gpu_mem_peak_mib'):.0f} MiB   "
          f"torch_alloc {_tstat('vggt', 'torch_max_alloc_mib'):.0f} MiB")
    print(f"  4b. MapAnything {dur['mapanything']:>8.1f} s   "
          f"gpu_peak {_g('mapanything', 'gpu_mem_peak_mib'):.0f} MiB   "
          f"torch_alloc {_tstat('mapanything', 'torch_max_alloc_mib'):.0f} MiB")
    print(f"\n  gpu_peak/gpu_mean/util: whole-GPU nvidia-smi samples @0.5s "
          f"({len(sampler.samples)} samples, baseline {baseline:.0f} MiB) — "
          f"covers subprocesses (SAM3D workers)")
    print(f"  t_alloc/t_resv: torch.cuda per-stage peak, parent process only")
    print(f"\n  per timeline frame : {total / args.frames:.2f} s")
    print(f"  per image          : {total / n_images:.2f} s")
    print("=" * 110)

    out = {
        "scene": args.scene, "frames_per_cam": args.frames, "cams": cams,
        "pids": PIDS, "shifts": shifts, "seed": args.seed,
        "device": args.device, "stages_s": {n: s for n, s, _, _ in rows},
        "vggt_s": dur["vggt"], "mapanything_s": dur["mapanything"],
        "total_s": total,
        "sync_estimated": [float(x) for x in est_anch],
        "sync_true": [float(x) for x in true_anch],
        "gpu_stage_stats": gpu,
        "gpu_overall": gpu_all,
        "gpu_baseline_mib": baseline,
        "torch_stage_stats": clock.torch_stats,
        "gpu_sample_interval_s": sampler.dt,
    }
    with open(args.work_dir / "runtime.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(args.work_dir / "gpu_samples.csv", "w") as f:
        f.write("t_rel_s,mem_used_mib,util_pct\n")
        for t, m, u in sampler.samples:
            f.write(f"{t - t0:.2f},{m:.0f},{u:.0f}\n")
    logger.info(f"Timings saved → {args.work_dir / 'runtime.json'} "
                f"+ gpu_samples.csv")


if __name__ == "__main__":
    main()
