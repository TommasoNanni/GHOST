"""
RICH pipeline v3 — VGGT Omega cameras + combined appearance/affine ReID v2.

Differences from rich_vggt_pipeline.py:
  - Uses CrossViewReidentifierV4 (tiered canonical-pose + geometric RMSE matching).
  - Writes cross_view_reid_v2.json (does not overwrite v1 results).
  - Imports ParametersExtractor from parameters_extraction_v2.

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
from preprocessing.run_mapanything import MapAnythingScaleEstimator
from preprocessing.run_vggt import VGGTPreprocessor
from preprocessing.segmentation import PersonSegmenter
from preprocessing.parameters_extraction_v2 import ParametersExtractor
from preprocessing.cross_view_v4 import CrossViewReidentifierV4
from utilities.visualize_segmented_reids import visualize_reid


VGGT_WEIGHTS = CONFIG.data.vggt_omega_checkpoint


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


def process_scene(
    scene, segmenter, estimator, reidentifier,
    output_dir, vggt_weights, vggt_devices,
    skip_cams: set[str] | None = None,
    ma_estimator: MapAnythingScaleEstimator | None = None,
):
    """Run the full pipeline (with VGGT) on a single scene."""
    skip_cams = skip_cams or set()
    if skip_cams:
        removed = [v for v in scene.videos if v.video_id in skip_cams]
        if removed:
            print(f"  Skipping cameras: {[v.video_id for v in removed]}")
            scene.videos = [v for v in scene.videos if v.video_id not in skip_cams]
            scene.video_ids = [v.video_id for v in scene.videos]

    print(f"\n=== Scene: {scene.scene_id} ({len(scene)} videos) ===")
    for v in scene:
        print(f"  {v}")

    # Step 1: Segment people in the scene.
    _seg_output_dir = Path(output_dir) / scene.scene_id
    _reid_already_done = (_seg_output_dir / "cross_view_reid.json").exists()
    _body_already_done = all(
        any((_seg_output_dir / v.video_id / "body_data").glob("person_*.npz"))
        for v in scene.videos
    )

    if _body_already_done:
        print(f"\n--- Step 1: Segmentation (skipped — body already done) ---")
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

        # Free segmenter GPU memory before SAM3D workers start.
        segmenter._predictor = None
        segmenter._models_ready = False
        torch.cuda.empty_cache()

    scene_output_dir = Path(next(iter(video_dirs.values()))).parent

    # Step 2: Body parameter estimation.
    if _body_already_done:
        print(f"\n--- Step 2: Body parameter estimation (skipped — already done) ---")
    else:
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

    # Step 5: VGGT camera + depth estimation.
    vggt_cameras_path = scene_output_dir / "vggt_cameras.npz"
    vggt_depth_path   = scene_output_dir / "vggt_depth.npz"

    # A run is considered complete only if depth_valid has at least one True entry.
    # Old pipeline runs saved depth as all-zero with depth_valid all-False.
    def _depth_valid(path: Path) -> bool:
        try:
            return bool(np.load(path, mmap_mode="r")["depth_valid"].any())
        except Exception:
            return False

    _vggt_done = (vggt_cameras_path.exists() and vggt_depth_path.exists()
                  and _depth_valid(vggt_depth_path))

    if _vggt_done:
        print(f"\n--- Step 5: VGGT (already done, skipping) ---")
    else:
        if vggt_cameras_path.exists() and vggt_depth_path.exists():
            print(f"\n--- Step 5: VGGT depth stale/invalid — recomputing ---")
            vggt_cameras_path.unlink()
            vggt_depth_path.unlink()
        else:
            print(f"\n--- Step 5: VGGT camera + depth estimation ---")
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

    # Step 6: MapAnything metric scale estimation.
    print(f"\n--- Step 6: MapAnything scale estimation ---")
    if ma_estimator is None:
        print("  Skipped (--skip-mapanything).")
    elif not (vggt_cameras_path.exists() and vggt_depth_path.exists()):
        print("  Skipped — VGGT outputs missing.")
    else:
        ma_estimator.process_scene(
            scene_dir=scene_output_dir,
            img_root=Path(CONFIG.data.rich_data_root) / scene.scene_id,
        )

    # Step 7: Visualise re-ID corrected segmentation (only if ReID ran this session).
    if not _reid_already_done:
        print(f"\n--- Step 7: Visualising re-ID corrected segmentation ---")
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
        print(f"\n--- Skipping step 10: re-ID visualisation (cross-view ReID was already done) ---")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-start",   type=int,   default=None)
    parser.add_argument("--scene-end",     type=int,   default=None)
    parser.add_argument("--scene",         type=str,   default=None,
                        help="Process only the scene whose ID contains this string "
                             "(e.g. 'Pavallion_003'). Applied after --scene-start/end.")
    parser.add_argument("--vggt-weights",  type=str,   default=VGGT_WEIGHTS,
                        help="HuggingFace repo ID or local path for VGGT weights")
    parser.add_argument("--vggt-devices",  type=str,   nargs="+", default=None,
                        help="CUDA device strings for VGGT, e.g. cuda:0 cuda:1. "
                             "Defaults to all available GPUs.")
    parser.add_argument("--skip-mapanything", action="store_true", default=False,
                        help="Skip step 7 (MapAnything scale estimation).")
    parser.add_argument("--mapanything-device", type=str, default=None,
                        help="CUDA device for MapAnything (defaults to first VGGT device).")
    parser.add_argument("--mapanything-batch-size", type=int, default=8,
                        help="Number of consecutive frames per MapAnything call.")
    parser.add_argument("--skip-cameras", type=str, nargs="+", default=[],
                        metavar="SCENE_ID:CAM_ID",
                        help="Cameras to skip, as 'scene_id:cam_id' pairs "
                             "(e.g. Pavallion_003_plankjack:cam_03). "
                             "Can be repeated for multiple cameras.")
    args = parser.parse_args()

    skip_cams_map: dict[str, set[str]] = {}
    for entry in args.skip_cameras:
        if ":" not in entry:
            parser.error(f"--skip-cameras: expected 'scene_id:cam_id', got {entry!r}")
        scene_id, cam_id = entry.split(":", 1)
        skip_cams_map.setdefault(scene_id, set()).add(cam_id)

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

    if args.scene is not None:
        ds.scenes = [s for s in ds.scenes if args.scene in s.scene_id]
        print(f"Filtered to scene '{args.scene}' → {len(ds.scenes)} scene(s): "
              f"{[s.scene_id for s in ds.scenes]}")

    import gc as _gc

    if args.skip_mapanything:
        ma_estimator = None
    else:
        ma_device = args.mapanything_device or vggt_devices[0]
        ma_estimator = MapAnythingScaleEstimator(
            device=ma_device,
            batch_size=args.mapanything_batch_size,
        )
        print(f"MapAnything estimator: device={ma_device}  batch_size={args.mapanything_batch_size}")

    failed_videos_by_scene: dict[str, list[str]] = {}
    needs_reid: list[str] = []

    for scene in ds.scenes:
        scene_dir = Path(output_dir) / scene.scene_id
        if (
            (scene_dir / "cross_view_reid.json").exists()
            and (scene_dir / "vggt_cameras_centered.npz").exists()
            and (ma_estimator is None or (scene_dir / "mapanything_scale_centered.npy").exists())
        ):
            print(f"Scene {scene.scene_id}: already done, skipping.")
            continue

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
        reidentifier = CrossViewReidentifierV4(
            droid_weights=getattr(CONFIG.data, "droid_weights", None),
            slam_cams=getattr(CONFIG.parameters_extraction, "slam_cams", None),
        )
        try:
            process_scene(
                scene, segmenter, estimator, reidentifier, output_dir,
                vggt_weights=args.vggt_weights,
                vggt_devices=vggt_devices,
                skip_cams=skip_cams_map.get(scene.scene_id, set()),
                ma_estimator=ma_estimator,
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
