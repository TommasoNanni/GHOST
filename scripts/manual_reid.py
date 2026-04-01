"""Manual cross-view re-identification and downstream reprocessing.

After the automated pipeline runs, person IDs may be inconsistent across
camera views. This script lets you specify the correct global assignment
manually in a YAML file, then applies it to every scene and redoes the
Kabsch camera alignment so the data is ready for fusion training.

Workflow
--------
1.  Generate a template YAML showing current person IDs in every scene/camera:

        pixi run python -m scripts.manual_reid \\
            --scenes-root preprocessing_outputs/rich11_segmentation_test \\
            --print-template > reid_assignments.yaml

2.  Edit reid_assignments.yaml — fill in the correct {old_id: new_id} entries
    for cameras where the automated ReID was wrong.

3.  Apply all remappings and redo Kabsch for every listed scene:

        pixi run python -m scripts.manual_reid \\
            --scenes-root preprocessing_outputs/rich11_segmentation_test \\
            --assignments reid_assignments.yaml

4.  Preview without touching files:

        pixi run python -m scripts.manual_reid ... --dry-run

YAML format
-----------
Top-level keys are scene names.  Under each scene, keys are camera directory
names, each mapping {old_person_id: new_global_person_id}.
Only scenes/cameras/IDs that need correction must appear; the rest are skipped.

    BBQ_001_guitar:
      cam_01:
        3: 1    # person 3 in cam_01 is global person 1
        4: 2    # person 4 in cam_01 is global person 2
      cam_02:
        5: 1    # person 5 in cam_02 is global person 1
    BBQ_001_juggle:
      cam_00:
        2: 0
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tyro
import yaml

from preprocessing.camera_alignment import CameraAlignment

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_person_ids(cam_dir: Path) -> list[int]:
    body_dir = cam_dir / "body_data"
    if not body_dir.exists():
        return []
    ids = []
    for p in body_dir.glob("person_*.npz"):
        try:
            ids.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(ids)


def _scene_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))


def _cam_dirs(scene_dir: Path) -> list[Path]:
    return sorted(d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("cam_"))


# ── template printer ───────────────────────────────────────────────────────────

def _print_template(root: Path) -> None:
    scenes = _scene_dirs(root)
    if not scenes:
        logger.error(f"No scene directories found in {root}")
        return

    lines: list[str] = []
    lines.append(f"# Manual ReID assignments for: {root}")
    lines.append("# Format: scene -> camera -> {old_person_id: new_global_person_id}")
    lines.append("# IDs not listed keep their current value.\n")

    for scene_dir in scenes:
        cams = _cam_dirs(scene_dir)
        if not cams:
            continue
        lines.append(f"{scene_dir.name}:")
        for cam_dir in cams:
            ids = _read_person_ids(cam_dir)
            id_str = ", ".join(str(i) for i in ids) if ids else "none"
            lines.append(f"  {cam_dir.name}:  # current IDs: [{id_str}]")
            for pid in ids:
                lines.append(f"    {pid}: {pid}")
        lines.append("")

    content = "\n".join(lines)
    print(content)

    out_dir = Path(__file__).parent.parent / "manual_reid_yaml"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{root.name}.yaml"
    out_path.write_text(content)
    logger.info(f"Template saved to {out_path}")


# ── validation ─────────────────────────────────────────────────────────────────

def _validate_remap(cam_dir: Path, remap: dict[int, int]) -> list[str]:
    errors = []
    current_ids = set(_read_person_ids(cam_dir))
    new_ids: set[int] = set()
    for old_id, new_id in remap.items():
        if old_id not in current_ids:
            errors.append(
                f"  {cam_dir.name}: person_{old_id}.npz does not exist "
                f"(available: {sorted(current_ids)})"
            )
        new_ids.add(new_id)
    # IDs unchanged — check collisions with remapped targets
    for uid in current_ids - set(remap.keys()):
        if uid in new_ids:
            errors.append(
                f"  {cam_dir.name}: new_id={uid} collides with unchanged person_{uid}.npz"
            )
    return errors


# ── per-camera apply ───────────────────────────────────────────────────────────

def _apply_body_data_remap(body_dir: Path, remap: dict[int, int], dry_run: bool) -> None:
    tmp_renames: list[tuple[Path, Path]] = []
    for old_id, new_id in remap.items():
        src = body_dir / f"person_{old_id}.npz"
        if not src.exists():
            continue
        tmp = body_dir / f"person_{old_id}._manual_reid_tmp.npz"
        dst = body_dir / f"person_{new_id}.npz"
        logger.info(f"      {src.name} → {dst.name}")
        if not dry_run:
            src.rename(tmp)
            tmp_renames.append((tmp, dst))
    if not dry_run:
        for tmp, dst in tmp_renames:
            tmp.rename(dst)


def _apply_summary_remap(body_dir: Path, remap: dict[int, int], dry_run: bool) -> None:
    summary_path = body_dir / "body_params_summary.json"
    if not summary_path.exists():
        return
    with open(summary_path) as f:
        summary = json.load(f)
    persons = summary.get("persons", {})
    new_persons: dict[str, object] = {}
    for str_id, info in persons.items():
        old_id = int(str_id)
        new_persons[str(remap.get(old_id, old_id))] = info
    summary["persons"] = new_persons
    if not dry_run:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


def _apply_mask_remap(cam_dir: Path, remap: dict[int, int], dry_run: bool) -> None:
    npz_path = cam_dir / "mask_data.npz"
    if not npz_path.exists():
        logger.warning(f"      mask_data.npz not found, skipping")
        return
    if dry_run:
        return
    tmp_path = npz_path.with_suffix(".tmp.npz")
    with (
        zipfile.ZipFile(str(npz_path), "r") as zf_in,
        zipfile.ZipFile(str(tmp_path), "w",
                        compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf_out,
    ):
        for name in sorted(zf_in.namelist()):
            with zf_in.open(name) as fh:
                mask = np.load(io.BytesIO(fh.read()))
            new_mask = mask.copy()
            # Two-pass remap to avoid collisions mid-rename
            offset = int(mask.max()) + 10000
            for old_id, new_id in remap.items():
                new_mask[mask == old_id] = offset + new_id
            for _, new_id in remap.items():
                new_mask[new_mask == offset + new_id] = new_id
            buf = io.BytesIO()
            np.save(buf, new_mask)
            zf_out.writestr(name, buf.getvalue())
    tmp_path.replace(npz_path)


def _apply_json_remap(cam_dir: Path, remap: dict[int, int], dry_run: bool) -> None:
    json_dir = cam_dir / "json_data"
    if not json_dir.exists():
        return
    if dry_run:
        return
    for json_path in sorted(json_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        if "labels" in data:
            new_labels: dict[str, object] = {}
            for str_id, info in data["labels"].items():
                old_id = int(str_id)
                new_labels[str(remap.get(old_id, old_id))] = info
            data["labels"] = new_labels
        with open(json_path, "w") as f:
            json.dump(data, f)


def _apply_remap_for_camera(cam_dir: Path, remap: dict[int, int], dry_run: bool) -> None:
    logger.info(f"    {cam_dir.name}: {remap}")
    body_dir = cam_dir / "body_data"
    _apply_body_data_remap(body_dir, remap, dry_run)
    _apply_summary_remap(body_dir, remap, dry_run)
    _apply_mask_remap(cam_dir, remap, dry_run)
    _apply_json_remap(cam_dir, remap, dry_run)


# ── Kabsch ─────────────────────────────────────────────────────────────────────

def _redo_kabsch(scene_dir: Path, dry_run: bool) -> None:
    video_dirs = {
        d.name: d for d in _cam_dirs(scene_dir)
        if (d / "body_data").exists()
    }
    if len(video_dirs) < 2:
        logger.warning(f"  {scene_dir.name}: fewer than 2 cameras with body_data — skipping Kabsch")
        return

    aligner = CameraAlignment()
    alignment = aligner.estimate(video_dirs)
    if not alignment:
        logger.warning(f"  {scene_dir.name}: Kabsch produced no pairs")
        return

    logger.info(f"  {scene_dir.name}: {len(alignment)} pair(s) estimated")
    if not dry_run:
        CameraAlignment.save(alignment, scene_dir)


# ── main ───────────────────────────────────────────────────────────────────────

def main(
    scenes_root:    Path,
    assignments:    Path | None = None,
    print_template: bool = False,
    dry_run:        bool = False,
) -> None:
    """
    Parameters
    ----------
    scenes_root    : folder containing all scene directories
                     (e.g. preprocessing_outputs/rich11_segmentation_test)
    assignments    : YAML file with {scene: {cam: {old_id: new_id}}} remappings
    print_template : print a YAML template showing current IDs for all scenes and exit
    dry_run        : preview changes without modifying any files
    """
    scenes_root = Path(scenes_root)
    if not scenes_root.exists():
        raise FileNotFoundError(f"Root not found: {scenes_root}")

    if print_template:
        _print_template(scenes_root)
        return

    if assignments is None:
        logger.error("Provide --assignments <file.yaml> or use --print-template")
        sys.exit(1)

    with open(assignments) as f:
        raw = yaml.safe_load(f) or {}

    # Normalise to {scene_name: {cam_name: {int: int}}}
    parsed: dict[str, dict[str, dict[int, int]]] = {}
    for scene_name, cam_block in raw.items():
        if not cam_block:
            continue
        scene_remaps: dict[str, dict[int, int]] = {}
        for cam_name, id_block in (cam_block or {}).items():
            if not id_block:
                continue
            remap = {int(k): int(v) for k, v in id_block.items() if int(k) != int(v)}
            if remap:
                scene_remaps[cam_name] = remap
        if scene_remaps:
            parsed[scene_name] = scene_remaps

    if not parsed:
        logger.info("No non-identity remappings found — only redoing Kabsch for all scenes.")

    # ── validate all scenes before touching anything ───────────────────────────
    all_errors: list[str] = []
    for scene_name, cam_remaps in parsed.items():
        scene_dir = scenes_root / scene_name
        if not scene_dir.exists():
            all_errors.append(f"{scene_name}: directory not found in {scenes_root}")
            continue
        for cam_name, remap in cam_remaps.items():
            cam_dir = scene_dir / cam_name
            if not cam_dir.exists():
                all_errors.append(f"{scene_name}/{cam_name}: directory not found")
                continue
            all_errors.extend(_validate_remap(cam_dir, remap))

    if all_errors:
        logger.error("Validation failed:\n" + "\n".join(all_errors))
        sys.exit(1)

    if dry_run:
        logger.info("[DRY RUN] no files will be modified\n")

    # ── apply remaps scene by scene ────────────────────────────────────────────
    scenes_to_process = sorted(
        set(parsed.keys()) | {d.name for d in _scene_dirs(scenes_root)}
    ) if not parsed else sorted(parsed.keys())

    # If assignments only cover some scenes, also redo Kabsch on the rest
    all_scene_dirs = {d.name: d for d in _scene_dirs(scenes_root)}

    for scene_name in scenes_to_process:
        scene_dir = scenes_root / scene_name
        if not scene_dir.exists():
            continue
        cam_remaps = parsed.get(scene_name, {})
        if cam_remaps:
            logger.info(f"\n[{scene_name}] applying remaps …")
            for cam_name, remap in cam_remaps.items():
                _apply_remap_for_camera(scene_dir / cam_name, remap, dry_run)
        else:
            logger.info(f"\n[{scene_name}] no remaps — redoing Kabsch only")

        logger.info(f"[{scene_name}] redoing Kabsch …")
        _redo_kabsch(scene_dir, dry_run)

    action = "would update" if dry_run else "updated"
    logger.info(f"\nDone — {action} {len(scenes_to_process)} scene(s).")


if __name__ == "__main__":
    tyro.cli(main)
