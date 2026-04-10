"""
Run cross-view person re-identification on pre-computed body_data.

Copies body_data/ from SOURCE_DIR into OUTPUT_DIR at the start of each run,
then runs CrossVideoReidentifier on OUTPUT_DIR (in-place). SOURCE_DIR is never
modified, so experiments can be repeated cleanly.

mask_data.npz and json_data/ are also copied since match_across_views rewrites them in-place.
"""
import logging
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

from configuration import CONFIG
from preprocessing.parameters_extraction import CrossVideoReidentifier

# --- configuration -----------------------------------------------------------
SOURCE_DIR  = Path("/cluster/project/cvg/students/tnanni/ghost/preprocessing_outputs/backup_scenes")
OUTPUT_DIR  = Path("/cluster/project/cvg/students/tnanni/ghost/preprocessing_outputs/new_reid_test")
SCENE       = "ParkingLot1_004_005_greetingchattingeating1"      # set to a scene ID string to process only that scene, e.g. "scene_01"
FORCE       = True      # re-run even if cross_view_reid.json already exists
VISUALIZE   = False      # generate *_segmentation_reid.mp4 videos after ReID
DATA_ROOT   = "/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train"
# -----------------------------------------------------------------------------


def scan_scenes(output_dir: Path) -> list[tuple[str, dict[str, Path]]]:
    """Return (scene_id, {video_id: video_dir}) for every scene that has body_data."""
    scenes = []
    for scene_dir in sorted(output_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        video_dirs: dict[str, Path] = {}
        for cam_dir in sorted(scene_dir.iterdir()):
            if cam_dir.is_dir() and (cam_dir / "body_data").exists():
                video_dirs[cam_dir.name] = cam_dir
        if video_dirs:
            scenes.append((scene_dir.name, video_dirs))
    return scenes


def copy_scene_data(source_dir: Path, output_dir: Path, scene_id: str) -> None:
    """Copy body_data/, mask_data.npz and json_data/ from source_dir into output_dir.

    mask_data.npz and json_data/ are also copied (not just body_data) because
    match_across_views rewrites them in-place via _apply_reid_remap.
    """
    src_scene = source_dir / scene_id
    dst_scene = output_dir / scene_id
    for cam_dir in sorted(src_scene.iterdir()):
        if not cam_dir.is_dir() or not (cam_dir / "body_data").exists():
            continue
        dst_cam = dst_scene / cam_dir.name

        for item in ("body_data", "json_data"):
            src = cam_dir / item
            dst = dst_cam / item
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        mask_src = cam_dir / "mask_data.npz"
        if mask_src.exists():
            dst_cam.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_src, dst_cam / "mask_data.npz")

        print(f"  Copied {cam_dir.name}")


def main() -> None:
    if not SOURCE_DIR.exists():
        print(f"ERROR: SOURCE_DIR does not exist: {SOURCE_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if VISUALIZE and not DATA_ROOT:
        print("ERROR: VISUALIZE=True requires DATA_ROOT to be set.")
        sys.exit(1)

    reidentifier = CrossVideoReidentifier(
        threshold=getattr(CONFIG.parameters_extraction, "cross_view_reid_threshold", 0.4),
        appearance_weight=getattr(CONFIG.parameters_extraction, "cross_view_appearance_weight", 0.5),
        shape_weight=getattr(CONFIG.parameters_extraction, "cross_view_shape_weight", 0.2),
        pose_weight=getattr(CONFIG.parameters_extraction, "cross_view_pose_weight", 0.3),
    )

    scenes = scan_scenes(SOURCE_DIR)
    if not scenes:
        print(f"No scenes with body_data found in {SOURCE_DIR}")
        sys.exit(0)

    if SCENE:
        scenes = [(sid, vd) for sid, vd in scenes if sid == SCENE]
        if not scenes:
            print(f"Scene '{SCENE}' not found or has no body_data in {SOURCE_DIR}")
            sys.exit(1)

    print(f"Found {len(scenes)} scene(s) with body_data.")

    for scene_id, _ in scenes:
        # Resolve video_dirs to OUTPUT_DIR (where ReID will write in-place)
        video_dirs = {
            cam_dir.name: OUTPUT_DIR / scene_id / cam_dir.name
            for cam_dir in sorted((SOURCE_DIR / scene_id).iterdir())
            if (cam_dir / "body_data").exists()
        }
        print(f"\n=== Scene: {scene_id} ({len(video_dirs)} cameras) ===")
        for vid_id in sorted(video_dirs):
            print(f"  {vid_id}")

        print(f"  Copying data from source...")
        copy_scene_data(SOURCE_DIR, OUTPUT_DIR, scene_id)

        reid_marker = OUTPUT_DIR / scene_id / "cross_view_reid.json"
        if FORCE and reid_marker.exists():
            print(f"  FORCE: removing existing {reid_marker}")
            reid_marker.unlink()

        # match_across_views only reads scene.scene_id; no full Scene object needed.
        mock_scene = SimpleNamespace(scene_id=scene_id)
        reidentifier.match_across_views(scene=mock_scene, video_dirs=video_dirs)

        if VISUALIZE:
            from utilities.visualize_segmented_reids import visualize_reid
            print(f"\n  Generating ReID visualizations...")
            for vid_id, video_dir in sorted(video_dirs.items()):
                _cam_dir = Path(DATA_ROOT) / scene_id / vid_id
                frames_dir = _cam_dir / "frames" if (_cam_dir / "frames").exists() else _cam_dir if _cam_dir.exists() else None
                try:
                    out = visualize_reid(video_dir=video_dir, fps=30, frames_dir=frames_dir)
                    print(f"    {vid_id}: saved {out}")
                except FileNotFoundError as e:
                    print(f"    WARNING: skipping {vid_id} — {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
