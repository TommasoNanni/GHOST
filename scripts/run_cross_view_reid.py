"""
Run cross-view person re-identification on pre-computed body_data.

Scans an output directory for scenes (subdirs where at least one camera
subdir contains body_data/) and runs CrossVideoReidentifier on each,
writing cross_view_reid.json and remapping body_data / mask files in-place.

Usage:
    pixi run python -m scripts.run_cross_view_reid \\
        --output_dir /path/to/output \\
        [--scene SCENE_ID] \\
        [--force] \\
        [--visualize --data_root /path/to/rich/train]
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from configuration import CONFIG
from preprocessing.parameters_extraction import CrossVideoReidentifier


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cross-view ReID on existing body_data without re-running the full pipeline.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory containing scene subdirs (e.g. the pipeline output_dir).",
    )
    parser.add_argument(
        "--scene",
        help="Process only this scene ID (default: all scenes found).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if cross_view_reid.json already exists.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate *_segmentation_reid.mp4 videos after ReID.",
    )
    parser.add_argument(
        "--data_root",
        help="RICH data root containing original frames (required for --visualize).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: output_dir does not exist: {output_dir}")
        sys.exit(1)

    if args.visualize and not args.data_root:
        print("ERROR: --visualize requires --data_root to locate original frames.")
        sys.exit(1)

    reidentifier = CrossVideoReidentifier(
        threshold=getattr(CONFIG.parameters_extraction, "cross_view_reid_threshold", 0.4),
        appearance_weight=getattr(CONFIG.parameters_extraction, "cross_view_appearance_weight", 0.7),
        shape_weight=getattr(CONFIG.parameters_extraction, "cross_view_shape_weight", 0.3),
    )

    scenes = scan_scenes(output_dir)
    if not scenes:
        print(f"No scenes with body_data found in {output_dir}")
        sys.exit(0)

    if args.scene:
        scenes = [(sid, vd) for sid, vd in scenes if sid == args.scene]
        if not scenes:
            print(f"Scene '{args.scene}' not found or has no body_data in {output_dir}")
            sys.exit(1)

    print(f"Found {len(scenes)} scene(s) with body_data.")

    for scene_id, video_dirs in scenes:
        print(f"\n=== Scene: {scene_id} ({len(video_dirs)} cameras) ===")
        for vid_id in sorted(video_dirs):
            print(f"  {vid_id}")

        reid_marker = output_dir / scene_id / "cross_view_reid.json"
        if args.force and reid_marker.exists():
            print(f"  --force: removing existing {reid_marker}")
            reid_marker.unlink()

        # match_across_views only reads scene.scene_id; no full Scene object needed.
        mock_scene = SimpleNamespace(scene_id=scene_id)
        reidentifier.match_across_views(scene=mock_scene, video_dirs=video_dirs)

        if args.visualize:
            from utilities.visualize_segmented_reids import visualize_reid
            data_root = Path(args.data_root)
            print(f"\n  Generating ReID visualizations...")
            for vid_id, video_dir in sorted(video_dirs.items()):
                frames_dir = data_root / scene_id / vid_id / "frames"
                if not frames_dir.exists():
                    frames_dir = None
                try:
                    out = visualize_reid(
                        video_dir=video_dir,
                        fps=30,
                        frames_dir=frames_dir,
                    )
                    print(f"    {vid_id}: saved {out}")
                except FileNotFoundError as e:
                    print(f"    WARNING: skipping {vid_id} — {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
