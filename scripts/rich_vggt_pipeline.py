"""
RICH pipeline with VGGT camera + depth preprocessing.

Identical to rich_pipeline.py except:
  - Step 1.5: VGGT runs after segmentation and saves vggt_cameras.npz /
    vggt_depth.npz in the scene output directory.
  - Step 6 (CameraAlignment) is removed — VGGT provides camera geometry.

Multi-GPU for VGGT:
  T frames are split round-robin across all visible CUDA devices.
  Each device runs a separate VGGTPreprocessor instance in its own subprocess.
  Pass --vggt-devices cuda:0 cuda:1 ... to override; defaults to all GPUs.
"""

import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)

import json
import numpy as np
import torch

from configuration import CONFIG
from data.video_dataset import RichDataset
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset
from preprocessing.geometric_reidentifier import GeometricReidentifier
from preprocessing.run_vggt import VGGTPreprocessor
from preprocessing.segmentation import PersonSegmenter
from preprocessing.parameters_extraction import ParametersExtractor, CrossVideoReidentifier
from synchronize_videos.synchronizer import Synchronizer
from utilities.body_data import load_person_smplx_pose
from utilities.visualize_segmented_reids import visualize_reid

# ── Temporal-sync evaluation constants ────────────────────────────────────────
SYNC_MAX_SHIFT   = 148
SYNC_N_TRIALS    = 1
SYNC_SEED        = 42
SYNC_DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SYNC_MIN_OVERLAP = 100

VGGT_WEIGHTS = CONFIG.data.vggt_omega_checkpoint


def _load_body_data(
    video_dirs: dict[str, str],
) -> dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    for cam_id, video_dir in video_dirs.items():
        body_dir = Path(video_dir) / "body_data"
        if not body_dir.exists():
            continue
        persons: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for npz_path in sorted(body_dir.glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            result = load_person_smplx_pose(npz_path)
            if result is None:
                print(f"  WARNING: {npz_path.name} missing pose keys, skipping")
                continue
            persons[pid] = result
        if persons:
            cam_data[cam_id] = persons
    return cam_data


def _common_persons(cam_data: dict[str, dict[int, tuple]]) -> list[int]:
    sets = [set(persons.keys()) for persons in cam_data.values()]
    return sorted(set.intersection(*sets))


def _apply_shifts(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    shifts: dict[str, int],
    end_cuts: dict[str, int],
    pids: list[int],
    min_overlap: int = SYNC_MIN_OVERLAP,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]] | None:
    cam_ids = list(shifts.keys())
    max_s   = max(shifts.values())
    print(f"  shift_spread={max_s - min(shifts.values())}")

    joints_list: list[list[torch.Tensor]] = []
    confs_list:  list[list[torch.Tensor]] = []
    for cam_id in cam_ids:
        s  = max_s - shifts[cam_id]
        ec = end_cuts[cam_id]
        per_person_joints, per_person_confs = [], []
        for pid in pids:
            rotations, conf = cam_data[cam_id][pid]
            T   = rotations.shape[0]
            end = T - ec if ec > 0 else T
            remaining = end - s
            if remaining < min_overlap:
                print(
                    f"  WARNING: {cam_id} has only {remaining} frames after shift "
                    f"(need ≥{min_overlap}) — skipping sync"
                )
                return None
            per_person_joints.append(rotations[s:end].to(SYNC_DEVICE))
            per_person_confs .append(conf     [s:end].to(SYNC_DEVICE))
        joints_list.append(per_person_joints)
        confs_list .append(per_person_confs)
    return joints_list, confs_list


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _build_vggt_frame_paths(
    scene,
    video_dirs: dict[str, str],
) -> tuple[list[list[Path | None]], list[str]]:
    """Build frame_paths[t][k] and camera_names for VGGT from extracted frames.

    Cameras are ordered by video_id. T is the minimum number of frame files
    found across all cameras. Frame files are listed and sorted directly from
    each camera's frames_home directory — no naming convention is assumed.
    """
    sorted_videos = sorted(scene.videos, key=lambda v: v.video_id)
    camera_names  = [v.video_id for v in sorted_videos]

    # Collect sorted frame file lists per camera.
    per_cam_frames: list[list[Path]] = []
    for video in sorted_videos:
        fdir = video.frames_home
        if fdir is None or not fdir.is_dir():
            logging.warning(f"VGGT: no frames dir for {video.video_id} — using 0 frames")
            per_cam_frames.append([])
            continue
        frames = sorted(p for p in fdir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        per_cam_frames.append(frames)

    T = min((len(f) for f in per_cam_frames), default=0)
    if T == 0:
        logging.warning("VGGT: no frames found across cameras — skipping.")
        return [], camera_names

    frame_paths: list[list[Path | None]] = []
    for t in range(T):
        frame_paths.append([cam_frames[t] if cam_frames else None
                            for cam_frames in per_cam_frames])

    return frame_paths, camera_names


def process_scene(scene, segmenter, estimator, reidentifier, output_dir, vggt_weights, vggt_devices, skip_geo_reid=True):
    """Run the full pipeline (with VGGT) on a single scene."""
    print(f"\n=== Scene: {scene.scene_id} ({len(scene)} videos) ===")
    for v in scene:
        print(f"  {v}")

    # Step 1: Segment people in the scene.
    _seg_output_dir = Path(output_dir) / scene.scene_id
    _reid_already_done = (_seg_output_dir / "cross_view_reid.json").exists()
    _body_already_done = all(
        (_seg_output_dir / v.video_id / "body_data").exists()
        for v in scene.videos
    )

    if _body_already_done and _reid_already_done:
        print(f"\n--- Step 1: Segmentation (skipped — body + ReID already done) ---")
        video_dirs = {v.video_id: _seg_output_dir / v.video_id for v in scene.videos}
    else:
        print(f"\n--- Step 1: Segmentation ---")
        video_dirs = segmenter.segment_scene(
            scene=scene,
            output_dir=output_dir,
            vis=False,
        )
        print(f"\nSegmentation output dirs:")
        for video_id, vdir in video_dirs.items():
            print(f"  {video_id}: {vdir}")

        # Free the segmenter model before VGGT so GPU memory is available for VGGT workers.
        segmenter._predictor = None
        segmenter._models_ready = False
        torch.cuda.empty_cache()

    # Step 1.5: VGGT camera + depth estimation.
    # Frames are already extracted by Step 1. Cameras in the scene are fed
    # simultaneously to VGGT for each timestep so it can reason about their
    # relative poses and produce metric depth.
    scene_output_dir = Path(next(iter(video_dirs.values()))).parent
    vggt_cameras_path = scene_output_dir / "vggt_cameras.npz"
    vggt_depth_path   = scene_output_dir / "vggt_depth.npz"

    if vggt_cameras_path.exists() and vggt_depth_path.exists():
        print(f"\n--- Step 1.5: VGGT (already done, skipping) ---")
    else:
        print(f"\n--- Step 1.5: VGGT camera + depth estimation ---")
        frame_paths, camera_names = _build_vggt_frame_paths(scene, video_dirs)
        if frame_paths:
            print(f"  {len(frame_paths)} frames × {len(camera_names)} cameras")
            print(f"  Devices: {vggt_devices}")
            preprocessor = VGGTPreprocessor(weights=vggt_weights, device=vggt_devices[0])
            preprocessor.process_scene(
                frame_paths  = frame_paths,
                camera_names = camera_names,
                output_dir   = scene_output_dir,
                devices      = vggt_devices,
            )
            del preprocessor
            torch.cuda.empty_cache()
            print(f"  Saved → {vggt_cameras_path.name}, {vggt_depth_path.name}")
        else:
            print("  WARNING: no frames available — skipping VGGT.")

    # Step 2: Body parameter estimation.
    print(f"\n--- Step 2: Body parameter estimation ---")
    estimator.estimate_scene(
        scene=scene,
        video_dirs=video_dirs,
    )

    missing_body = [
        vid_id for vid_id, vid_dir in video_dirs.items()
        if not any((Path(vid_dir) / "body_data").glob("person_*.npz"))
    ]
    if missing_body:
        print(
            f"  WARNING: body estimation incomplete for cameras {missing_body} — "
            f"skipping cross-view ReID and subsequent steps for this scene."
        )
        return

    # Step 3: Cross-view person re-identification.
    print(f"\n--- Step 3: Cross-view ReID ---")
    _reid_already_done = (scene_output_dir / "cross_view_reid.json").exists()
    if _reid_already_done:
        print(f"  Already done (cross_view_reid.json exists), skipping.")
    else:
        reidentifier.match_across_views(
            scene=scene,
            video_dirs=video_dirs,
        )

    # Step 4: Verify MHR → SMPLX conversion.
    print(f"\n--- Step 4: Verifying MHR → SMPLX conversion ---")
    smplx_fields_found = {}
    for video_id, video_dir in video_dirs.items():
        body_dir = Path(video_dir) / "body_data"
        if not body_dir.exists():
            print(f"  WARNING: {body_dir} does not exist")
            continue
        for npz_path in sorted(body_dir.glob("person_*.npz")):
            data = dict(np.load(str(npz_path), allow_pickle=False))
            smplx_keys = [k for k in data if k.startswith("smplx_")]
            if smplx_keys:
                smplx_fields_found.setdefault(video_id, {})[npz_path.name] = smplx_keys
    if not smplx_fields_found:
        print(
            "  WARNING: No smplx_* fields found. "
            "Check that smplx_model_path and mhr_model_path are set in CONFIG."
        )

    # Step 5: Temporal synchronisation (optional).
    sync_cfg = getattr(CONFIG, "synchronization", None)
    if sync_cfg is not None and getattr(sync_cfg, "enabled", False):
        print(f"\n--- Step 5: Temporal synchronisation ---")
        cam_data = _load_body_data(video_dirs)
        if len(cam_data) < 2:
            print("  WARNING: fewer than 2 cameras with pose data — skipping sync eval")
        else:
            pids = _common_persons(cam_data)
            if not pids:
                print("  WARNING: no person ID common across all cameras — skipping sync eval")
            else:
                cam_ids = list(cam_data.keys())
                print(f"  Cameras: {cam_ids}")
                print(f"  Common persons: {pids}")
                sync = Synchronizer(method="cross_corr", device=SYNC_DEVICE, min_overlap=SYNC_MIN_OVERLAP)
                rng  = np.random.default_rng(SYNC_SEED)
                results = []
                for trial in range(SYNC_N_TRIALS):
                    raw_shifts  = [0] + rng.integers(-SYNC_MAX_SHIFT, SYNC_MAX_SHIFT + 1,
                                                      size=len(cam_ids) - 1).tolist()
                    true_shifts = {c: int(s) for c, s in zip(cam_ids, raw_shifts)}
                    end_cuts    = {c: int(e) for c, e in zip(cam_ids, rng.integers(
                        0, SYNC_MAX_SHIFT + 1, size=len(cam_ids)
                    ).tolist())}
                    print(f"  Trial {trial + 1}/{SYNC_N_TRIALS}  true shifts: {true_shifts}  end_cuts: {end_cuts}")

                    result = _apply_shifts(cam_data, true_shifts, end_cuts, pids)
                    if result is None:
                        continue
                    joints_list, confs_list = result
                    offset_mat = sync.estimate_offset_matrix(joints_list, confs_list)
                    weights    = sync.cycle_consistency_weights(offset_mat)
                    estimated  = sync.estimate_initial_times(offset_mat, weights)

                    true_t = torch.tensor([true_shifts[c] for c in cam_ids], dtype=torch.float32)
                    true_t = true_t - true_t.min()
                    errors = (estimated.cpu() - true_t).abs()
                    mae    = errors.mean().item()

                    for cam_id, tt, est, err in zip(cam_ids, true_t.tolist(),
                                                    estimated.cpu().tolist(), errors.tolist()):
                        print(f"    {cam_id}: true={tt:+.0f}  estimated={est:+.1f}  error={err:.1f}")
                    print(f"  MAE={mae:.2f}  "
                          f"within-1={((errors <= 1).float().mean().item()) * 100:.0f}%  "
                          f"within-2={((errors <= 2).float().mean().item()) * 100:.0f}%")
                    results.append({"mae": mae,
                                    "within_1": (errors <= 1).float().mean().item(),
                                    "within_2": (errors <= 2).float().mean().item()})

                if len(results) > 1:
                    all_mae = [r["mae"] for r in results]
                    print(f"\n  SUMMARY over {SYNC_N_TRIALS} trials:")
                    print(f"  MAE  mean={np.mean(all_mae):.2f}  "
                          f"median={np.median(all_mae):.2f}  max={np.max(all_mae):.2f}")
                    print(f"  Within 1fr  {np.mean([r['within_1'] for r in results]) * 100:.1f}%")
                    print(f"  Within 2fr  {np.mean([r['within_2'] for r in results]) * 100:.1f}%")

    # Step 6: Geometric post-ReID.
    if skip_geo_reid:
        print(f"\n--- Step 6: Geometric post-ReID (skipped via --skip-geo-reid) ---")
    else:
        geo_reid = GeometricReidentifier(
            distance_threshold=CONFIG.parameters_extraction.geo_reid_distance_threshold,
            min_overlap_frames=10,
            second_pass_threshold=CONFIG.parameters_extraction.geo_reid_second_pass_threshold,
        )
        geo_reid.reidentify(scene=scene, video_dirs=video_dirs)

    # Step 7: FusionDatapoint compatibility check.
    print(f"\n--- Step 7: FusionDatapoint compatibility check ---")
    try:
        fusion_dp = RICHFusionDatapoint(
            scene_dir=scene_output_dir,
            rich_data_root=CONFIG.data.rich_data_root,
        )
        ds = RICHFusionDataset([fusion_dp])
        inputs, targets = ds[0]
        print("  FusionDatapoint compatibility: OK")
    except Exception as e:
        print(f"  ERROR: FusionDatapoint failed to load: {e}")

    # Step 8: Visualise re-ID corrected segmentation (only if ReID ran this session).
    if not _reid_already_done:
        print(f"\n--- Step 8: Visualising re-ID corrected segmentation ---")
        for video in scene.videos:
            if video.video_id not in video_dirs:
                continue
            print(f"  {video.video_id}")
            try:
                visualize_reid(
                    video_dir=Path(video_dirs[video.video_id]),
                    fps=int(video.fps),
                    frames_dir=video.frames_home,
                )
            except FileNotFoundError as e:
                print(f"  WARNING: skipping visualisation — {e}")
    else:
        print(f"\n--- Skipping re-ID visualisation (cross-view ReID was already done) ---")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-start",   type=int,   default=None)
    parser.add_argument("--scene-end",     type=int,   default=None)
    parser.add_argument("--vggt-weights",  type=str,   default=VGGT_WEIGHTS,
                        help="HuggingFace repo ID or local path for VGGT weights")
    parser.add_argument("--vggt-devices",  type=str,   nargs="+", default=None,
                        help="CUDA device strings for VGGT, e.g. cuda:0 cuda:1. "
                             "Defaults to all available GPUs.")
    parser.add_argument("--skip-geo-reid", action="store_true", default=True,
                        help="Skip step 6 (geometric ReID). Use when VGGT cameras "
                             "are newly estimated and geo-reid should be re-run later.")
    args = parser.parse_args()

    # Auto-detect CUDA devices if not specified.
    if args.vggt_devices:
        vggt_devices = args.vggt_devices
    elif torch.cuda.is_available():
        vggt_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        vggt_devices = ["cpu"]
    print(f"VGGT devices: {vggt_devices}")

    rich_data_root = CONFIG.data.rich_data_root
    output_dir = CONFIG.data.output_directory
    scenes_slice = CONFIG.data.slice

    ds = RichDataset(
        data_root=rich_data_root,
        slice=scenes_slice,
        max_side=getattr(CONFIG.data, "rich_max_side", None),
        frames_base_dir=getattr(CONFIG.data, "rich_frames_base_dir", None),
    )

    if args.scene_start is not None or args.scene_end is not None:
        ds.scenes = ds.scenes[args.scene_start:args.scene_end]
        print(f"Processing scenes [{args.scene_start}:{args.scene_end}] → {len(ds.scenes)} scenes")

    import gc as _gc

    failed_videos_by_scene: dict[str, list[str]] = {}
    needs_reid: list[str] = []

    for scene in ds.scenes:
        segmenter = PersonSegmenter(
            checkpoint_path=CONFIG.segmentation.checkpoint_path,
            text_prompt=CONFIG.segmentation.text_prompt,
            redetect_interval=CONFIG.segmentation.redetect_interval,
            new_det_thresh=CONFIG.segmentation.new_det_thresh,
            score_threshold_detection=CONFIG.segmentation.score_threshold_detection,
        )
        estimator = ParametersExtractor(
            sam3d_hf_repo = CONFIG.parameters_extraction.sam3d_id,
            sam3d_step = CONFIG.parameters_extraction.sam3d_step,
            bbox_padding = CONFIG.parameters_extraction.bbox_padding,
            smplx_model_path = CONFIG.data.smplx_model_path,
            mhr_model_path  = CONFIG.data.mhr_model_path,
            reid_threshold = CONFIG.parameters_extraction.reid_threshold,
            reid_match_window = getattr(CONFIG.parameters_extraction, "reid_match_window", 5),
            rich_data_root = CONFIG.data.rich_data_root,
        )
        reidentifier = CrossVideoReidentifier(
            threshold = getattr(CONFIG.parameters_extraction, "cross_view_reid_threshold", 0.4),
            appearance_weight = getattr(CONFIG.parameters_extraction, "cross_view_appearance_weight", 0.5),
            shape_weight = getattr(CONFIG.parameters_extraction, "cross_view_shape_weight", 0.2),
            pose_weight = getattr(CONFIG.parameters_extraction, "cross_view_pose_weight", 0.3),
        )
        try:
            process_scene(
                scene, segmenter, estimator, reidentifier, output_dir,
                vggt_weights=args.vggt_weights,
                vggt_devices=vggt_devices,
                skip_geo_reid=args.skip_geo_reid,
            )
        except Exception as e:
            logging.error(
                f"Scene {scene.scene_id} raised an unexpected error: {e}", exc_info=True
            )
        finally:
            del segmenter, estimator, reidentifier
            _gc.collect()
            torch.cuda.empty_cache()

        scene_dir = Path(output_dir) / scene.scene_id
        missing = [
            v.video_id for v in scene.videos
            if not any((scene_dir / v.video_id / "body_data").glob("person_*.npz"))
        ]
        if missing:
            failed_videos_by_scene[scene.scene_id] = missing
        reid_done = (scene_dir / "cross_view_reid.json").exists()
        has_any_body = len(missing) < len(scene.videos)
        if has_any_body and not reid_done:
            needs_reid.append(scene.scene_id)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PIPELINE SUMMARY")
    print("=" * 64)
    if failed_videos_by_scene:
        print(f"\nScenes with missing body data ({len(failed_videos_by_scene)}):")
        for scene_id, cams in failed_videos_by_scene.items():
            print(f"  {scene_id}: missing cameras → {cams}")
    else:
        print("\nBody estimation completed for all cameras in all scenes.")
    if needs_reid:
        print(f"\nScenes with body data but no cross-view ReID ({len(needs_reid)}):")
        for scene_id in needs_reid:
            print(f"  {scene_id}")
    else:
        print("\nCross-view ReID completed for all scenes.")
    print("=" * 64)


if __name__ == "__main__":
    main()
