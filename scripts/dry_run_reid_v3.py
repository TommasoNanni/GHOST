"""Dry-run cross-view ReID v3–v7 on one or all training scenes."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration import CONFIG
from data.video_dataset import Scene, Video
from preprocessing.cross_view_v3 import CrossViewReidentifierV3
from preprocessing.cross_view_v4 import CrossViewReidentifierV4
from preprocessing.cross_view_v5 import CrossViewReidentifierV5
from preprocessing.cross_view_v6 import CrossViewReidentifierV6
from preprocessing.cross_view_v7 import CrossViewReidentifierV7

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")

OUTPUT_DIR = getattr(CONFIG.data, "output_dir", None)
DATA_ROOT  = getattr(CONFIG.data, "data_root", None)


def _build_scene(scene_name: str, scene_dir: Path):
    video_dirs: dict[str, Path] = {}
    for cam_dir in sorted(scene_dir.iterdir()):
        if not cam_dir.is_dir() or cam_dir.name == "cam_10":
            continue
        body_dir = cam_dir / "body_data"
        if body_dir.exists() and any(body_dir.glob("person_*.npz")):
            video_dirs[cam_dir.name] = cam_dir
    if len(video_dirs) < 2:
        return None
    videos = [Video(frames_dir=vdir) for vdir in video_dirs.values()]
    scene = Scene(scene_id=scene_name, videos=videos)
    return scene, video_dirs


# ── v7 scoring vs manual GT ───────────────────────────────────────────────────

def _cam_num(cam: str) -> int:
    """Same rule as scripts/build_clean_body_data.py: digits in the dir name.
    Returns -1 for camera names without digits (unmappable to cXpY tokens)."""
    digits = re.sub(r"\D", "", cam)
    return int(digits) if digits else -1


def _score_scene(scene_name: str, clusters: list[set], manual_reid_path: Path,
                 manual_ops_path: Path | None) -> None:
    """Pair-level precision/recall of predicted clusters vs manual_reid.json.

    Caveat: manual_reid tokens use POST-manual-ops local ids.  On cams with
    non-empty manual_operations the on-disk ids may differ → scores there are
    only indicative (a warning is printed).
    """
    mr = json.loads(manual_reid_path.read_text())
    groups = None
    for dataset, scenes in mr.items():
        if dataset.startswith("_"):
            continue
        if scene_name in scenes:
            groups = scenes[scene_name].get("groups", {})
            break
    if groups is None:
        print(f"  [score] {scene_name}: not in manual_reid.json — skipped")
        return

    if manual_ops_path and manual_ops_path.exists():
        mo = json.loads(manual_ops_path.read_text())
        for dataset, scenes in mo.items():
            if dataset.startswith("_") or scene_name not in scenes:
                continue
            dirty = [c for c, ops in scenes[scene_name].items()
                     if any(v for v in ops.values())]
            if dirty:
                print(f"  [score] WARNING: manual ops exist for {dirty} — "
                      f"ids there may not match on-disk tracks")

    tok_re = re.compile(r"c(\d+)p(\d+)")
    gt_nodes: set[tuple[int, int]] = set()
    gt_pairs: set[frozenset] = set()
    for gid, tokens in groups.items():
        members = []
        for t in tokens:
            m = tok_re.fullmatch(t.strip())
            if m:
                members.append((int(m.group(1)), int(m.group(2))))
        gt_nodes.update(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if members[i][0] != members[j][0]:
                    gt_pairs.add(frozenset((members[i], members[j])))

    pred_pairs: set[frozenset] = set()
    for cl in clusters:
        members = [(_cam_num(v), p) for v, p in cl if _cam_num(v) >= 0]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if (members[i][0] != members[j][0]
                        and members[i] in gt_nodes and members[j] in gt_nodes):
                    pred_pairs.add(frozenset((members[i], members[j])))

    tp = len(pred_pairs & gt_pairs)
    prec = tp / len(pred_pairs) if pred_pairs else float("nan")
    rec = tp / len(gt_pairs) if gt_pairs else float("nan")
    print(f"  [score] {scene_name}: pairs pred={len(pred_pairs)} gt={len(gt_pairs)} "
          f"tp={tp}  precision={prec:.3f}  recall={rec:.3f}")
    for fp in sorted(pred_pairs - gt_pairs):
        a, b = sorted(fp)
        print(f"  [score]   FALSE MERGE c{a[0]}p{a[1]} ↔ c{b[0]}p{b[1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--version", choices=["v3", "v4", "v5", "v6", "v7"], default="v4")
    parser.add_argument("--reid_ckpt", default=None,
                        help="v5/v6/v7: TransReID ckpt for cross-view appearance")
    parser.add_argument("--frames_root", default=None,
                        help="root of frames, <root>/<scene>/<cam>/*.jpg")
    parser.add_argument("--camera_pass", action="store_true",
                        help="v7: run ReidCameraPass (GPU) when reid_cameras.npz "
                             "is missing; default = skip with a message")
    parser.add_argument("--moving_cams", nargs="*", default=[],
                        help="v7 camera pass: cameras treated as moving")
    parser.add_argument("--score", action="store_true",
                        help="v7: score clusters vs manual_reid.json")
    args = parser.parse_args()

    if args.output_dir is None:
        print("ERROR: --output_dir required"); sys.exit(1)

    output_dir = Path(args.output_dir)
    if args.version in ("v5", "v6", "v7"):
        cls = {"v5": CrossViewReidentifierV5, "v6": CrossViewReidentifierV6,
               "v7": CrossViewReidentifierV7}[args.version]
        reidentifier = cls(reid_ckpt=args.reid_ckpt)
    else:
        reidentifier = {"v3": CrossViewReidentifierV3, "v4": CrossViewReidentifierV4}[args.version]()

    scene_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and (args.scene is None or d.name == args.scene)
    )
    for scene_dir in scene_dirs:
        result = _build_scene(scene_dir.name, scene_dir)
        if result is None:
            print(f"\nSkipping {scene_dir.name}: fewer than 2 cameras")
            continue
        scene, video_dirs = result
        frames_dirs = None
        if args.frames_root:
            fr = Path(args.frames_root) / scene_dir.name
            frames_dirs = {cam: fr / cam for cam in video_dirs if (fr / cam).is_dir()}
        print(f"\n{'='*60}")
        print(f"Scene: {scene_dir.name}  ({len(video_dirs)} cameras)  [{args.version}]")

        if (args.version == "v7" and args.camera_pass
                and not (scene_dir / "reid_cameras.npz").exists()):
            if not frames_dirs:
                print("  --camera_pass needs --frames_root; skipping camera pass")
            else:
                from preprocessing.reid_cameras import ReidCameraPass
                ReidCameraPass(
                    vggt_weights=CONFIG.data.vggt_omega_checkpoint,
                    droid_weights=getattr(CONFIG.data, "droid_weights", None),
                ).run_scene(scene_dir, video_dirs, frames_dirs,
                            moving_cams=args.moving_cams)

        clusters = reidentifier.match_across_views(
            scene=scene, video_dirs=video_dirs,
            frames_dirs=frames_dirs, dry_run=True)

        if args.score and args.version == "v7":
            if clusters is None:
                print("  [score] no clusters returned — skipped")
            else:
                repo = Path(__file__).parent.parent
                _score_scene(scene_dir.name, clusters,
                             repo / "manual_reid.json",
                             repo / "manual_operations.json")


if __name__ == "__main__":
    main()
