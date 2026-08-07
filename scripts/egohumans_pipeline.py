"""
EgoHumans pipeline — VGGT + body estimation + cross-view ReID.

Mirrors rich_pipeline_v3.py but uses EgoHumans exo-camera data.

Data layout (after undistortion by utilities/undistort_egohumans.py):

    data_root/
        01_tagging/
            media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/01_tagging/
                001_tagging/
                    exo/
                        cam01/
                            images_undistorted/   ← pipeline input
                            calibration.json      ← K_new (pinhole, D=0)
                        cam04/ cam06/ cam08/
                    processed_data/               ← GT annotations
                    colmap/                       ← COLMAP output
                002_tagging/ ...
        02_lego/ ...

Multi-GPU for VGGT:
  T frames are split round-robin across all visible CUDA devices.
  Pass --vggt-devices cuda:0 cuda:1 ... to override; defaults to all GPUs.
"""

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.append(str(_REPO_ROOT / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

import gc
import json
import numpy as np
import torch

from configuration import CONFIG
from data.video_dataset import Video, EgoHumansScene
from preprocessing.run_vggt import VGGTPreprocessor
from preprocessing.run_mapanything import MapAnythingScaleEstimator
from preprocessing.segmentation import PersonSegmenter
from preprocessing.parameters_extraction_v2 import ParametersExtractor
from preprocessing.cross_view_v4 import CrossViewReidentifierV4
from utilities.visualize_segmented_reids import visualize_reid

VGGT_WEIGHTS     = CONFIG.data.vggt_omega_checkpoint


# Deep inner path baked into every EgoHumans activity folder.
_INNER = Path("media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready")

# Exo cameras to use per activity (others were deleted during preprocessing).
KEEP_CAMS: dict[str, list[str]] = {
    "01_tagging":    ["cam01", "cam04", "cam06", "cam08"],
    "02_lego":       ["cam02", "cam03", "cam04", "cam06"],
    "03_fencing":    ["cam04", "cam05", "cam10", "cam13"],
    "04_basketball": ["cam01", "cam03", "cam04", "cam08"],
    "05_volleyball": ["cam02", "cam04", "cam08", "cam11"],
    "06_badminton":  ["cam01", "cam02", "cam05", "cam07"],
    "07_tennis":     ["cam04", "cam09", "cam12", "cam20"],
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _build_scenes(
    data_root: Path,
    activities: list[str] | None = None,
    seq_filter: str | None = None,
) -> list[EgoHumansScene]:
    """Walk data_root and build EgoHumansScene objects for all sequences."""
    all_activities = sorted(KEEP_CAMS.keys())
    if activities:
        all_activities = [a for a in all_activities if a in activities]

    scenes: list[EgoHumansScene] = []
    for activity in all_activities:
        cam_ready = data_root / activity / _INNER / activity
        if not cam_ready.is_dir():
            logging.warning(f"Activity dir not found: {cam_ready}")
            continue
        keep = KEEP_CAMS[activity]
        for seq_dir in sorted(cam_ready.iterdir()):
            if not seq_dir.is_dir():
                continue
            if seq_filter and seq_filter not in seq_dir.name:
                continue
            videos: list[Video] = []
            for cam_name in keep:
                undist_dir = seq_dir / "exo" / cam_name / "images_undistorted"
                if not undist_dir.is_dir():
                    logging.warning(f"Missing {undist_dir.relative_to(data_root)}")
                    continue
                _has_images = any(p.suffix.lower() in _IMAGE_EXTS for p in undist_dir.iterdir())
                if not _has_images:
                    # After resizing, originals are deleted and only frames/ subdir remains.
                    _frames_sub = undist_dir / "frames"
                    _has_images = _frames_sub.is_dir() and any(p.suffix.lower() in _IMAGE_EXTS for p in _frames_sub.iterdir())
                if not _has_images:
                    logging.warning(f"Empty {undist_dir.relative_to(data_root)}")
                    continue
                _frames_sub = undist_dir / "frames"
                _effective_dir = _frames_sub if (_frames_sub.is_dir() and any(p.suffix.lower() in _IMAGE_EXTS for p in _frames_sub.iterdir())) else undist_dir
                video = Video(frames_dir=_effective_dir, max_side=getattr(CONFIG.data, "egohumans_max_side", None) if _effective_dir == undist_dir else None)
                video.video_id = cam_name
                videos.append(video)
            if not videos:
                logging.warning(f"No valid cameras for {activity}/{seq_dir.name}, skipping.")
                continue
            scene_id = f"{activity}/{seq_dir.name}"
            scenes.append(EgoHumansScene(
                scene_id=scene_id,
                videos=videos,
                seq_dir=seq_dir,
            ))
    return scenes


def _build_vggt_frame_paths(
    scene: EgoHumansScene,
    video_dirs: dict[str, str],
) -> tuple[list[list[Path | None]], list[str]]:
    sorted_videos = sorted(scene.videos, key=lambda v: v.video_id)
    camera_names = [v.video_id for v in sorted_videos]

    per_cam_frames: list[list[Path]] = []
    for video in sorted_videos:
        fdir = video.frames_home
        if fdir is None or not fdir.is_dir():
            logging.warning(f"VGGT: no frames dir for {video.video_id}")
            per_cam_frames.append([])
            continue
        frames = sorted(p for p in fdir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        per_cam_frames.append(frames)

    T = min((len(f) for f in per_cam_frames), default=0)
    if T == 0:
        logging.warning("VGGT: no frames found — skipping.")
        return [], camera_names

    frame_paths = [
        [cam_frames[t] if cam_frames else None for cam_frames in per_cam_frames]
        for t in range(T)
    ]
    return frame_paths, camera_names


def process_scene(
    scene: EgoHumansScene,
    segmenter: PersonSegmenter,
    estimator: ParametersExtractor,
    reidentifier: CrossViewReidentifierV4,
    output_dir: str | Path,
    vggt_weights: str,
    vggt_devices: list[str],
    ma_estimator: MapAnythingScaleEstimator | None = None,
    skip_cams: set[str] | None = None,
):
    """Run the full pipeline on a single EgoHumans scene."""
    skip_cams = skip_cams or set()
    if skip_cams:
        removed = [v for v in scene.videos if v.video_id in skip_cams]
        if removed:
            print(f"  Skipping cameras: {[v.video_id for v in removed]}")
            scene.videos = [v for v in scene.videos if v.video_id not in skip_cams]

    print(f"\n=== Scene: {scene.scene_id} ({len(scene.videos)} cameras) ===")
    for v in scene.videos:
        print(f"  {v}")

    # scene_id is "activity/seq_name"; Path handles the slash correctly.
    _seg_output_dir = Path(output_dir) / scene.scene_id
    _reid_already_done = (_seg_output_dir / "cross_view_reid.json").exists()
    _body_already_done = all(
        any((_seg_output_dir / v.video_id / "body_data").glob("person_*.npz"))
        for v in scene.videos
    )

    # Step 1: Segmentation.
    if _body_already_done:
        print(f"\n--- Step 1: Segmentation (skipped — body already done) ---")
        video_dirs = {v.video_id: str(_seg_output_dir / v.video_id) for v in scene.videos}
    else:
        print(f"\n--- Step 1: Segmentation ---")
        # segmenter creates output_dir/scene.scene_id/video_id/ internally
        video_dirs = segmenter.segment_scene(scene=scene, output_dir=output_dir, vis=False)
        segmenter._predictor = None
        segmenter._models_ready = False
        torch.cuda.empty_cache()

    # After segmentation, originals may have been deleted and replaced by frames/ subdir.
    # Update each Video's frames_dir so body estimator finds the right path.
    for video in scene.videos:
        if video.frames_dir is not None:
            frames_sub = video.frames_dir / "frames"
            if frames_sub.is_dir() and any(p.suffix.lower() in _IMAGE_EXTS for p in frames_sub.iterdir()):
                video.frames_dir = frames_sub

    scene_output_dir = Path(next(iter(video_dirs.values()))).parent

    # Step 2: Body parameter estimation.
    if _body_already_done:
        print(f"\n--- Step 2: Body parameter estimation (skipped — already done) ---")
    else:
        print(f"\n--- Step 2: Body parameter estimation ---")
        estimator.estimate_scene(scene=scene, video_dirs=video_dirs)

    missing_body = [
        vid_id for vid_id, vid_dir in video_dirs.items()
        if not any((Path(vid_dir) / "body_data").glob("person_*.npz"))
    ]
    if missing_body:
        print(f"  WARNING: body estimation incomplete for {missing_body} — skipping scene.")
        return

    # Step 3: Cross-view ReID.
    print(f"\n--- Step 3: Cross-view ReID ---")
    _reid_already_done = (scene_output_dir / "cross_view_reid.json").exists()
    if _reid_already_done:
        print(f"  Already done, skipping.")
    else:
        reidentifier.match_across_views(scene=scene, video_dirs=video_dirs)

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
    vggt_cameras_path = scene_output_dir / "vggt_cameras_centered.npz"
    vggt_depth_path   = scene_output_dir / "vggt_depth_centered.npz"

    def _depth_valid(path: Path) -> bool:
        try:
            return bool(np.load(path, mmap_mode="r")["depth_valid"].any())
        except Exception:
            return False

    _vggt_done = (
        vggt_cameras_path.exists()
        and vggt_depth_path.exists()
        and _depth_valid(vggt_depth_path)
    )

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
                frame_paths=frame_paths,
                camera_names=camera_names,
                output_dir=scene_output_dir,
                devices=vggt_devices,
            )
            del preprocessor
            torch.cuda.empty_cache()
        else:
            print("  WARNING: no frames — skipping VGGT.")

    # Step 6: MapAnything metric scale estimation.
    print(f"\n--- Step 6: MapAnything scale estimation ---")
    if ma_estimator is None:
        print("  Skipped (--skip-mapanything).")
    elif not (vggt_cameras_path.exists() and vggt_depth_path.exists()):
        print("  Skipped — VGGT outputs missing.")
    else:
        # Images live at seq_dir/exo/<cam_name>/images_undistorted/
        # MapAnythingScaleEstimator expects img_root/<cam_name>/ (searches one level deep).
        ma_estimator.process_scene(
            scene_dir=scene_output_dir,
            img_root=scene.seq_dir / "exo",
        )

    # Step 7: Visualise re-ID (only if ReID ran this session).
    if not _reid_already_done:
        print(f"\n--- Step 7: Visualising re-ID corrected segmentation ---")
        for video in scene.videos:
            if video.video_id not in video_dirs:
                continue
            try:
                visualize_reid(
                    video_dir=Path(video_dirs[video.video_id]),
                    fps=int(video.fps),
                    frames_dir=video.frames_home,
                )
            except FileNotFoundError as e:
                print(f"  WARNING: skipping visualisation — {e}")
    else:
        print(f"\n--- Step 7: Skipping re-ID visualisation (already done) ---")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",     type=str, required=True,
                        help="EgoHumans root (contains 01_tagging/, 02_lego/, ...)")
    parser.add_argument("--output-dir",    type=str, required=True,
                        help="Pipeline output directory")
    parser.add_argument("--activity",      type=str, nargs="+", default=None,
                        help="Restrict to these activities (e.g. 01_tagging 02_lego)")
    parser.add_argument("--seq",           type=str, default=None,
                        help="Only process sequences whose name contains this string")
    parser.add_argument("--scene-start",   type=int, default=None)
    parser.add_argument("--scene-end",     type=int, default=None)
    parser.add_argument("--vggt-weights",  type=str, default=VGGT_WEIGHTS)
    parser.add_argument("--vggt-devices",  type=str, nargs="+", default=None)
    parser.add_argument("--skip-mapanything",      action="store_true", default=False)
    parser.add_argument("--mapanything-device",    type=str, default=None,
                        help="CUDA device for MapAnything (defaults to first VGGT device).")
    parser.add_argument("--mapanything-batch-size", type=int, default=8)
    parser.add_argument("--skip-cameras",  type=str, nargs="+", default=[],
                        metavar="SCENE_ID:CAM_ID")
    args = parser.parse_args()

    skip_cams_map: dict[str, set[str]] = {}
    for entry in args.skip_cameras:
        if ":" not in entry:
            parser.error(f"--skip-cameras: expected 'scene_id:cam_id', got {entry!r}")
        scene_id, cam_id = entry.split(":", 1)
        skip_cams_map.setdefault(scene_id, set()).add(cam_id)

    if args.vggt_devices:
        vggt_devices = args.vggt_devices
    elif torch.cuda.is_available():
        vggt_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        vggt_devices = ["cpu"]
    print(f"VGGT devices: {vggt_devices}")

    if args.skip_mapanything:
        ma_estimator = None
        print("MapAnything: skipped (--skip-mapanything).")
    else:
        ma_device = args.mapanything_device or vggt_devices[0]
        ma_estimator = MapAnythingScaleEstimator(
            device=ma_device,
            batch_size=args.mapanything_batch_size,
        )
        print(f"MapAnything estimator: device={ma_device}  batch_size={args.mapanything_batch_size}")

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = _build_scenes(data_root, activities=args.activity, seq_filter=args.seq)
    if args.scene_start is not None or args.scene_end is not None:
        scenes = scenes[args.scene_start:args.scene_end]
    print(f"Processing {len(scenes)} scenes")

    failed: dict[str, list[str]] = {}
    needs_reid: list[str] = []

    for scene in scenes:
        scene_dir = output_dir / scene.scene_id
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
            sam3d_hf_repo=CONFIG.parameters_extraction.sam3d_id,
            sam3d_step=CONFIG.parameters_extraction.sam3d_step,
            bbox_padding=CONFIG.parameters_extraction.bbox_padding,
            smplx_model_path=CONFIG.data.smplx_model_path,
            mhr_model_path=CONFIG.data.mhr_model_path,
            reid_threshold=CONFIG.parameters_extraction.reid_threshold,
            reid_match_window=getattr(CONFIG.parameters_extraction, "reid_match_window", 5),
            rich_data_root=None,
        )
        reidentifier = CrossViewReidentifierV4(
            droid_weights=getattr(CONFIG.data, "droid_weights", None),
            slam_cams=getattr(CONFIG.parameters_extraction, "slam_cams", None),
        )
        try:
            process_scene(
                scene, segmenter, estimator, reidentifier,
                output_dir=output_dir,
                vggt_weights=args.vggt_weights,
                vggt_devices=vggt_devices,
                ma_estimator=ma_estimator,
                skip_cams=skip_cams_map.get(scene.scene_id, set()),
            )
        except Exception as e:
            logging.error(f"Scene {scene.scene_id} failed: {e}", exc_info=True)
        finally:
            del segmenter, estimator, reidentifier
            gc.collect()
            torch.cuda.empty_cache()

        scene_dir = output_dir / scene.scene_id
        missing = [
            v.video_id for v in scene.videos
            if not any((scene_dir / v.video_id / "body_data").glob("person_*.npz"))
        ]
        if missing:
            failed[scene.scene_id] = missing
        reid_done = (scene_dir / "cross_view_reid.json").exists()
        if len(missing) < len(scene.videos) and not reid_done:
            needs_reid.append(scene.scene_id)

    print("\n" + "=" * 64)
    print("PIPELINE SUMMARY")
    print("=" * 64)
    if failed:
        print(f"\nScenes with missing body data ({len(failed)}):")
        for sid, cams in failed.items():
            print(f"  {sid}: {cams}")
    else:
        print("\nBody estimation complete for all scenes.")
    if needs_reid:
        print(f"\nScenes missing cross-view ReID ({len(needs_reid)}):")
        for sid in needs_reid:
            print(f"  {sid}")
    else:
        print("\nCross-view ReID complete for all scenes.")
    print("=" * 64)


if __name__ == "__main__":
    main()
