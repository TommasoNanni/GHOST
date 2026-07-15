"""Manual surgery on within-view (single-camera) person tracks.

Cross-view ReID (``manual_reid.json`` + ``scripts/manual_reid.py``) assumes
one clean track per real person per camera. That assumption regularly breaks
*within* a single camera: SAM2/SAM3 loses a person to occlusion and
re-acquires them under a new id (one real person → two ids), or a track
silently swaps identity when two people cross paths (two real people → one
id), or a spurious id appears from a hand/reflection/background artifact.
This module fixes exactly that, directly on disk, before cross-view ReID
ever runs.

For one camera's ``video_dir`` (e.g. ``<scene>/<video_id>``), every operation
below keeps these in sync:

    body_data/person_<id>.npz            per-frame body params + frame_indices
    body_data/body_params_summary.json   frame ranges / shapes per person
    body_data/appearance_gallery.npz     per-person DINOv3/TransReID gallery
    mask_data.npz                        per-frame segmentation masks
    json_data/mask_<frame>.json          per-frame bbox/label metadata

Operations
----------
merge_tracks   fuse two ids into one, optionally at an explicit split frame
split_track    cut one id into two at a frame boundary (inverse of merge)
swap_tracks    exchange two ids over a frame window (id-steal at a crossing)
remove_track   delete a spurious id entirely
rename_track   relabel one id, without touching any other track
list_tracks    inspect current per-camera ids and their frame ranges

All mutating operations accept ``dry_run=True`` to log the plan without
touching any file, and raise on ambiguous input (id collisions, overlapping
frame ranges without a split point, etc.) rather than guessing.

mask_data.npz / json_data are only decoded and rewritten for the frames a
given operation actually touches (found via each track's frame_indices, not
the full frame range) — cheap even on long videos.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── low-level: per-person .npz ──────────────────────────────────────────────

def _load_person(body_dir: Path, pid: int) -> dict[str, np.ndarray]:
    path = body_dir / f"person_{pid}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _save_person(body_dir: Path, pid: int, arrays: dict[str, np.ndarray]) -> None:
    np.savez(str(body_dir / f"person_{pid}.npz"), **arrays)


# ── low-level: body_params_summary.json ─────────────────────────────────────

def _remove_from_summary(body_dir: Path, ids_to_drop: list[int]) -> None:
    summary_path = body_dir / "body_params_summary.json"
    if not summary_path.exists():
        return
    with open(summary_path) as f:
        summary = json.load(f)
    for pid in ids_to_drop:
        summary.get("persons", {}).pop(str(pid), None)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def _update_summary_entry(body_dir: Path, pid: int, arrays: dict[str, np.ndarray]) -> None:
    summary_path = body_dir / "body_params_summary.json"
    summary = {"video_id": body_dir.parent.name, "persons": {}}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    frame_indices = arrays["frame_indices"]
    summary.setdefault("persons", {})[str(pid)] = {
        "num_frames": int(len(frame_indices)),
        "frame_range": [int(frame_indices.min()), int(frame_indices.max())],
        "param_shapes": {k: list(v.shape) for k, v in arrays.items()},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


# ── low-level: appearance_gallery.npz ───────────────────────────────────────

def _remove_from_gallery(body_dir: Path, ids_to_drop: list[int]) -> None:
    gallery_path = body_dir / "appearance_gallery.npz"
    if not gallery_path.exists():
        return
    with np.load(gallery_path) as data:
        arrays = {k: data[k] for k in data.files}
    for pid in ids_to_drop:
        arrays.pop(str(pid), None)
        arrays.pop(f"{pid}_conf", None)
    np.savez(str(gallery_path), **arrays)


def _reassign_gallery(body_dir: Path, id_map: dict[int, int]) -> None:
    """Rewrite gallery keys per id_map; ids mapping to the same output id have
    their feature/conf samples concatenated (used by merge_tracks/rename_track)."""
    gallery_path = body_dir / "appearance_gallery.npz"
    if not gallery_path.exists():
        return
    with np.load(gallery_path) as data:
        arrays = {k: data[k] for k in data.files}

    untouched_pids = {k for k in arrays if not k.endswith("_conf") and int(k) not in id_map}
    new_arrays = {k: arrays[k] for k in untouched_pids}
    new_arrays.update({f"{k}_conf": arrays[f"{k}_conf"] for k in untouched_pids if f"{k}_conf" in arrays})

    feats: dict[int, list[np.ndarray]] = {}
    confs: dict[int, list[np.ndarray]] = {}
    for old_id, new_id in id_map.items():
        if str(old_id) not in arrays:
            continue
        feats.setdefault(new_id, []).append(arrays[str(old_id)])
        if f"{old_id}_conf" in arrays:
            confs.setdefault(new_id, []).append(arrays[f"{old_id}_conf"])

    for new_id, parts in feats.items():
        new_arrays[str(new_id)] = np.concatenate(parts, axis=0)
        if new_id in confs:
            new_arrays[f"{new_id}_conf"] = np.concatenate(confs[new_id], axis=0)

    np.savez(str(gallery_path), **new_arrays)


# ── low-level: mask_data.npz + json_data/, touched frames only ─────────────

def _frame_idx_from_stem(stem: str) -> int:
    s = stem.replace("mask_", "", 1)
    return int(s.split("_")[0])


def _frame_index_maps(video_dir: Path) -> tuple[dict[int, Path], dict[int, str]]:
    """Map frame_idx -> json_data path and frame_idx -> mask_data.npz key.

    Built from filenames/archive namelist only (no decoding), so this is
    cheap even for long videos.
    """
    json_map: dict[int, Path] = {}
    json_dir = video_dir / "json_data"
    if json_dir.exists():
        for p in json_dir.glob("*.json"):
            json_map[_frame_idx_from_stem(p.stem)] = p

    mask_map: dict[int, str] = {}
    npz_path = video_dir / "mask_data.npz"
    if npz_path.exists():
        with zipfile.ZipFile(str(npz_path), "r") as zf:
            for name in zf.namelist():
                stem = name[:-4] if name.endswith(".npy") else name
                mask_map[_frame_idx_from_stem(stem)] = name

    return json_map, mask_map


def _apply_frame_actions(
    video_dir: Path,
    actions: dict[int, dict[int, int | None]],
    dry_run: bool = False,
) -> None:
    """Rewrite mask_data.npz + json_data/*.json for a set of frames.

    actions: {frame_idx: {old_person_id: new_person_id_or_None}}.
    new_person_id=None deletes that person's pixels/label in that frame; an
    int remaps old_person_id -> new_person_id. Frames absent from `actions`
    are copied through unchanged (raw bytes, no decode/re-encode).
    """
    if not actions:
        return
    json_map, mask_map = _frame_index_maps(video_dir)

    npz_path = video_dir / "mask_data.npz"
    if npz_path.exists() and not dry_run:
        tmp_path = npz_path.with_suffix(".tmp.npz")
        with (
            zipfile.ZipFile(str(npz_path), "r") as zf_in,
            zipfile.ZipFile(str(tmp_path), "w",
                            compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf_out,
        ):
            for name in zf_in.namelist():
                stem = name[:-4] if name.endswith(".npy") else name
                action = actions.get(_frame_idx_from_stem(stem))
                if action is None:
                    zf_out.writestr(name, zf_in.read(name))
                    continue
                with zf_in.open(name) as f:
                    mask_img = np.load(io.BytesIO(f.read()))
                new_mask = mask_img.copy()
                for old_id, new_id in action.items():
                    region = mask_img == old_id
                    new_mask[region] = 0 if new_id is None else new_id
                buf = io.BytesIO()
                np.save(buf, new_mask)
                zf_out.writestr(name, buf.getvalue())
        tmp_path.replace(npz_path)

    if dry_run:
        return

    for frame_idx, action in actions.items():
        json_path = json_map.get(frame_idx)
        if json_path is None:
            continue
        with open(json_path) as f:
            data = json.load(f)
        labels = data.get("labels", {})
        if not any(str(oid) in labels for oid in action):
            continue

        # Destination-based remap computed from the *original* labels, so a
        # simultaneous swap {a: b, b: a} resolves correctly (the pop-in-place
        # form would see b still present while remapping a and wrongly drop it).
        # Identity mappings claim their slot first, so a remap onto an
        # untouched id is the one flagged as a collision (never the reverse).
        new_labels: dict[str, object] = {}
        items = sorted(
            labels.items(),
            key=lambda kv: 0 if action.get(int(kv[0]), int(kv[0])) == int(kv[0]) else 1,
        )
        for str_id, info in items:
            old_id = int(str_id)
            new_id = action.get(old_id, old_id)
            if new_id is None:
                continue
            dest = str(new_id)
            if dest in new_labels:
                logger.warning(
                    f"{json_path}: label collision remapping {old_id}->{new_id}, "
                    f"keeping original id {old_id} instead"
                )
                info["instance_id"] = old_id
                new_labels[str_id] = info
                continue
            info["instance_id"] = new_id
            new_labels[dest] = info
        data["labels"] = new_labels
        with open(json_path, "w") as f:
            json.dump(data, f)


# ── public API ───────────────────────────────────────────────────────────────

def list_tracks(video_dir: Path | str) -> dict[int, dict]:
    """Return {person_id: {"num_frames", "frame_range"}} for one camera."""
    summary_path = Path(video_dir) / "body_data" / "body_params_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        summary = json.load(f)
    return {int(pid): info for pid, info in summary.get("persons", {}).items()}


def remove_track(
    video_dir: Path | str,
    person_id: int,
    start_frame: int | None = None,
    end_frame: int | None = None,
    dry_run: bool = False,
) -> None:
    """Delete a spurious track, whole or over a frame window.

    Without start/end the entire track is deleted (a fully spurious id, e.g. a
    hand/background artifact). With start_frame/end_frame (inclusive) only the
    frames in that window are removed — the surviving frames stay under the same
    id. Use the windowed form to excise a spurious stretch where a real track
    briefly latched onto a tiny/background blob (e.g. a track whose bbox
    collapses for a range of frames) without discarding the good frames.
    """
    video_dir = Path(video_dir)
    body_dir = video_dir / "body_data"
    npz_path = body_dir / f"person_{person_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} does not exist")

    arrays = _load_person(body_dir, person_id)
    frame_indices = arrays["frame_indices"]

    if start_frame is None and end_frame is None:
        remove_mask = np.ones(len(frame_indices), dtype=bool)
    else:
        lo = start_frame if start_frame is not None else int(frame_indices.min())
        hi = end_frame if end_frame is not None else int(frame_indices.max())
        remove_mask = (frame_indices >= lo) & (frame_indices <= hi)
        if not remove_mask.any():
            logger.warning(
                f"{video_dir.name}: remove person {person_id} [{lo},{hi}] "
                f"matches no frames — nothing to do"
            )
            return

    remove_frames = frame_indices[remove_mask].tolist()
    logger.info(
        f"{video_dir.name}: removing person {person_id} "
        f"({len(remove_frames)}/{len(frame_indices)} frames"
        + ("" if remove_mask.all() else f", window {remove_frames[0]}-{remove_frames[-1]}")
        + ")" + (" [dry run]" if dry_run else "")
    )

    actions = {f: {person_id: None} for f in remove_frames}
    _apply_frame_actions(video_dir, actions, dry_run=dry_run)
    if dry_run:
        return

    if remove_mask.all():
        npz_path.unlink()
        _remove_from_summary(body_dir, [person_id])
        _remove_from_gallery(body_dir, [person_id])
    else:
        kept = {k: v[~remove_mask] for k, v in arrays.items()}
        _save_person(body_dir, person_id, kept)
        _update_summary_entry(body_dir, person_id, kept)


def rename_track(video_dir: Path | str, old_id: int, new_id: int, dry_run: bool = False) -> None:
    """Relabel one track's person id, without touching any other track."""
    video_dir = Path(video_dir)
    body_dir = video_dir / "body_data"
    if old_id == new_id:
        return
    if (body_dir / f"person_{new_id}.npz").exists():
        raise ValueError(f"person_{new_id}.npz already exists — choose an unused id")

    arrays = _load_person(body_dir, old_id)
    frame_indices = arrays["frame_indices"].tolist()

    logger.info(
        f"{video_dir.name}: renaming person {old_id} -> {new_id} ({len(frame_indices)} frames)"
        + (" [dry run]" if dry_run else "")
    )

    actions = {f: {old_id: new_id} for f in frame_indices}
    _apply_frame_actions(video_dir, actions, dry_run=dry_run)
    if dry_run:
        return

    (body_dir / f"person_{old_id}.npz").unlink()
    _save_person(body_dir, new_id, arrays)
    _remove_from_summary(body_dir, [old_id])
    _update_summary_entry(body_dir, new_id, arrays)
    _reassign_gallery(body_dir, {old_id: new_id})


def merge_tracks(
    video_dir: Path | str,
    id_a: int,
    id_b: int,
    output_id: int | None = None,
    split_frame: int | None = None,
    dry_run: bool = False,
) -> int:
    """Fuse two within-view tracks into one person id.

    Typical case: SAM2/SAM3 lost the person mid-video (occlusion, motion
    blur) and re-acquired them under a new id — `id_a` and `id_b` are the
    same real person, split across two ids with non-overlapping frame ranges.

    split_frame : if given, frames < split_frame are taken from `id_a` and
        frames >= split_frame from `id_b`; any frames on the wrong side of
        the cut are dropped (pixels/labels erased, not kept under either id).
        Required if the two tracks' frame ranges overlap. If None, the frame
        ranges must be disjoint and their union is taken as-is.

    Returns the output person id (== id_a unless `output_id` is given).
    """
    video_dir = Path(video_dir)
    body_dir = video_dir / "body_data"
    if id_a == id_b:
        raise ValueError("id_a and id_b must differ")
    output_id = id_a if output_id is None else output_id
    if output_id not in (id_a, id_b) and (body_dir / f"person_{output_id}.npz").exists():
        raise ValueError(f"person_{output_id}.npz already exists — choose an unused output_id")

    arrays_a = _load_person(body_dir, id_a)
    arrays_b = _load_person(body_dir, id_b)
    frames_a = arrays_a["frame_indices"].tolist()
    frames_b = arrays_b["frame_indices"].tolist()

    common_keys = sorted(k for k in arrays_a if k != "frame_indices" and k in arrays_b)
    dropped_keys = sorted(
        (set(arrays_a) | set(arrays_b)) - set(common_keys) - {"frame_indices"}
    )
    if dropped_keys:
        logger.warning(
            f"merge_tracks {id_a}+{id_b}: keys not present in both tracks, "
            f"dropped from merge: {dropped_keys}"
        )

    idx_a = {f: i for i, f in enumerate(frames_a)}
    idx_b = {f: i for i, f in enumerate(frames_b)}
    frame_actions: dict[int, dict[int, int | None]] = {}
    plan: list[tuple[int, str]] = []  # (frame_idx, "a" | "b"), final order

    if split_frame is None:
        overlap = sorted(set(frames_a) & set(frames_b))
        if overlap:
            raise ValueError(
                f"merge_tracks {id_a}+{id_b}: frame ranges overlap at "
                f"{overlap[:10]}{'...' if len(overlap) > 10 else ''} "
                f"({len(overlap)} frames) — pass split_frame to resolve precedence."
            )
        for f in frames_a:
            plan.append((f, "a"))
            frame_actions.setdefault(f, {})[id_a] = output_id
        for f in frames_b:
            plan.append((f, "b"))
            frame_actions.setdefault(f, {})[id_b] = output_id
    else:
        dropped_a = [f for f in frames_a if f >= split_frame]
        dropped_b = [f for f in frames_b if f < split_frame]
        if dropped_a:
            logger.info(
                f"merge_tracks: dropping {len(dropped_a)} frame(s) of person {id_a} "
                f"at/after split_frame={split_frame}"
            )
        if dropped_b:
            logger.info(
                f"merge_tracks: dropping {len(dropped_b)} frame(s) of person {id_b} "
                f"before split_frame={split_frame}"
            )
        for f in frames_a:
            if f < split_frame:
                plan.append((f, "a"))
                frame_actions.setdefault(f, {})[id_a] = output_id
            else:
                frame_actions.setdefault(f, {})[id_a] = None
        for f in frames_b:
            if f >= split_frame:
                plan.append((f, "b"))
                frame_actions.setdefault(f, {})[id_b] = output_id
            else:
                frame_actions.setdefault(f, {})[id_b] = None

    plan.sort(key=lambda t: t[0])
    combined_frames = [f for f, _ in plan]

    logger.info(
        f"{video_dir.name}: merging person {id_b} into {id_a} -> output person "
        f"{output_id} ({len(combined_frames)} frames)" + (" [dry run]" if dry_run else "")
    )

    _apply_frame_actions(video_dir, frame_actions, dry_run=dry_run)
    if dry_run:
        return output_id

    new_arrays: dict[str, np.ndarray] = {"frame_indices": np.array(combined_frames, dtype=np.int32)}
    for key in common_keys:
        rows = []
        for f, src in plan:
            arr, idx = (arrays_a, idx_a) if src == "a" else (arrays_b, idx_b)
            rows.append(arr[key][idx[f]])
        new_arrays[key] = np.stack(rows, axis=0)

    for pid in {id_a, id_b} - {output_id}:
        (body_dir / f"person_{pid}.npz").unlink()
    _save_person(body_dir, output_id, new_arrays)

    _remove_from_summary(body_dir, list({id_a, id_b} - {output_id}))
    _update_summary_entry(body_dir, output_id, new_arrays)
    _reassign_gallery(body_dir, {id_a: output_id, id_b: output_id})

    return output_id


def split_track(
    video_dir: Path | str,
    person_id: int,
    split_frame: int,
    new_id: int,
    dry_run: bool = False,
) -> None:
    """Cut one track into two: frames >= split_frame move to `new_id`.

    Inverse of merge_tracks. Use when a single id silently swapped identity
    mid-track (e.g. two people crossed paths and SAM2/SAM3 followed the
    wrong one afterwards) — frames after the swap need a different person id
    than the frames before it, without disturbing any other track.
    """
    video_dir = Path(video_dir)
    body_dir = video_dir / "body_data"
    if (body_dir / f"person_{new_id}.npz").exists():
        raise ValueError(f"person_{new_id}.npz already exists — choose an unused id")

    arrays = _load_person(body_dir, person_id)
    frame_indices = arrays["frame_indices"]
    move_mask = frame_indices >= split_frame
    if not move_mask.any():
        raise ValueError(f"no frames of person {person_id} are >= split_frame={split_frame}")
    if move_mask.all():
        raise ValueError(
            f"all frames of person {person_id} are >= split_frame={split_frame} "
            "— nothing stays behind"
        )

    logger.info(
        f"{video_dir.name}: splitting person {person_id} at frame {split_frame} "
        f"-> new person {new_id} ({int(move_mask.sum())} frames)"
        + (" [dry run]" if dry_run else "")
    )

    move_frames = frame_indices[move_mask].tolist()
    frame_actions = {f: {person_id: new_id} for f in move_frames}
    _apply_frame_actions(video_dir, frame_actions, dry_run=dry_run)
    if dry_run:
        return

    keep_arrays = {k: v[~move_mask] for k, v in arrays.items()}
    move_arrays = {k: v[move_mask] for k, v in arrays.items()}

    _save_person(body_dir, person_id, keep_arrays)
    _save_person(body_dir, new_id, move_arrays)
    _update_summary_entry(body_dir, person_id, keep_arrays)
    _update_summary_entry(body_dir, new_id, move_arrays)

    # Gallery samples have no per-frame correspondence to reconstruct which
    # half they came from, so both halves keep the same appearance evidence
    # until the next segmentation/estimation run refreshes it.
    gallery_path = body_dir / "appearance_gallery.npz"
    if gallery_path.exists():
        with np.load(gallery_path) as data:
            g = {k: data[k] for k in data.files}
        if str(person_id) in g:
            g[str(new_id)] = g[str(person_id)].copy()
            if f"{person_id}_conf" in g:
                g[f"{new_id}_conf"] = g[f"{person_id}_conf"].copy()
        np.savez(str(gallery_path), **g)


def swap_tracks(
    video_dir: Path | str,
    id_a: int,
    id_b: int,
    start_frame: int,
    end_frame: int,
    dry_run: bool = False,
) -> None:
    """Exchange two ids over a closed frame window [start_frame, end_frame].

    Use for the id-steal / label-swap case: two people cross and, for a few
    frames, SAM2/SAM3 assigns each person the other's id. Within the window,
    every frame currently labelled id_a becomes id_b and vice versa (bodies,
    masks and json labels). Frames outside the window are untouched.

    Unlike merge/split this is non-destructive: no frames are dropped, the two
    ids both survive, only the ownership of the windowed frames is exchanged.
    Chain several swaps (one per crossing) to unpick a whole swap cascade.

    end_frame is inclusive. Frames a track does not have in the window are
    simply absent from the exchange (a swap only moves detections that exist).
    """
    video_dir = Path(video_dir)
    body_dir = video_dir / "body_data"
    if id_a == id_b:
        raise ValueError("id_a and id_b must differ")
    if start_frame > end_frame:
        raise ValueError(f"start_frame ({start_frame}) > end_frame ({end_frame})")

    arrays_a = _load_person(body_dir, id_a)
    arrays_b = _load_person(body_dir, id_b)
    frames_a = arrays_a["frame_indices"].tolist()
    frames_b = arrays_b["frame_indices"].tolist()

    def _in_win(f: int) -> bool:
        return start_frame <= f <= end_frame

    win_a = [f for f in frames_a if _in_win(f)]
    win_b = [f for f in frames_b if _in_win(f)]
    if not win_a and not win_b:
        logger.warning(
            f"{video_dir.name}: swap {id_a}<->{id_b} on [{start_frame},{end_frame}] "
            f"touches no frames of either track — nothing to do"
        )
        return

    common_keys = sorted(k for k in arrays_a if k != "frame_indices" and k in arrays_b)
    dropped_keys = sorted(
        (set(arrays_a) | set(arrays_b)) - set(common_keys) - {"frame_indices"}
    )
    if dropped_keys:
        logger.warning(
            f"swap {id_a}<->{id_b}: keys not present in both tracks, "
            f"dropped from swap: {dropped_keys}"
        )

    logger.info(
        f"{video_dir.name}: swapping {id_a}<->{id_b} on frames [{start_frame},{end_frame}] "
        f"({len(win_a)} frame(s) of {id_a} <-> {len(win_b)} frame(s) of {id_b})"
        + (" [dry run]" if dry_run else "")
    )

    # mask + json: windowed frames get {a->b, b->a}; both directions in one
    # action so _apply_frame_actions swaps them atomically per frame.
    frame_actions: dict[int, dict[int, int | None]] = {}
    for f in win_a:
        frame_actions.setdefault(f, {})[id_a] = id_b
    for f in win_b:
        frame_actions.setdefault(f, {})[id_b] = id_a
    _apply_frame_actions(video_dir, frame_actions, dry_run=dry_run)
    if dry_run:
        return

    idx_a = {int(f): i for i, f in enumerate(frames_a)}
    idx_b = {int(f): i for i, f in enumerate(frames_b)}

    # New id_a = a's out-of-window frames + b's in-window frames (and vice versa).
    plan_a = [(f, "a") for f in frames_a if not _in_win(f)] + [(f, "b") for f in win_b]
    plan_b = [(f, "b") for f in frames_b if not _in_win(f)] + [(f, "a") for f in win_a]

    def _build(plan: list[tuple[int, str]]) -> dict[str, np.ndarray] | None:
        if not plan:
            return None
        plan = sorted(plan, key=lambda t: t[0])
        out: dict[str, np.ndarray] = {
            "frame_indices": np.array([f for f, _ in plan], dtype=np.int32)
        }
        for key in common_keys:
            rows = []
            for f, src in plan:
                arr, idx = (arrays_a, idx_a) if src == "a" else (arrays_b, idx_b)
                rows.append(arr[key][idx[f]])
            out[key] = np.stack(rows, axis=0)
        return out

    new_a = _build(plan_a)
    new_b = _build(plan_b)

    for pid, new_arrays in ((id_a, new_a), (id_b, new_b)):
        npz_path = body_dir / f"person_{pid}.npz"
        if new_arrays is None:
            # every frame of this id moved to the other track and nothing came
            # back — the id no longer exists after the swap.
            if npz_path.exists():
                npz_path.unlink()
            _remove_from_summary(body_dir, [pid])
            _remove_from_gallery(body_dir, [pid])
        else:
            _save_person(body_dir, pid, new_arrays)
            _update_summary_entry(body_dir, pid, new_arrays)

    # appearance_gallery has no per-frame correspondence to re-split, so it is
    # left as-is; only cross-view ReID consumes it and swaps are a within-view
    # fix applied after that stage is (or is not) used.


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="op", required=True)

    p_list = sub.add_parser("list", help="show person ids and frame ranges for one camera")
    p_list.add_argument("video_dir", type=Path)

    p_remove = sub.add_parser("remove", help="delete a spurious track (whole, or a frame window)")
    p_remove.add_argument("video_dir", type=Path)
    p_remove.add_argument("person_id", type=int)
    p_remove.add_argument("--start-frame", type=int, default=None)
    p_remove.add_argument("--end-frame", type=int, default=None, help="inclusive")
    p_remove.add_argument("--dry-run", action="store_true")

    p_rename = sub.add_parser("rename", help="relabel one track's person id")
    p_rename.add_argument("video_dir", type=Path)
    p_rename.add_argument("old_id", type=int)
    p_rename.add_argument("new_id", type=int)
    p_rename.add_argument("--dry-run", action="store_true")

    p_merge = sub.add_parser("merge", help="fuse two tracks into one")
    p_merge.add_argument("video_dir", type=Path)
    p_merge.add_argument("id_a", type=int)
    p_merge.add_argument("id_b", type=int)
    p_merge.add_argument("--output-id", type=int, default=None)
    p_merge.add_argument("--split-frame", type=int, default=None,
                          help="frames < split_frame from id_a, >= from id_b")
    p_merge.add_argument("--dry-run", action="store_true")

    p_split = sub.add_parser("split", help="cut one track into two at a frame boundary")
    p_split.add_argument("video_dir", type=Path)
    p_split.add_argument("person_id", type=int)
    p_split.add_argument("split_frame", type=int)
    p_split.add_argument("new_id", type=int)
    p_split.add_argument("--dry-run", action="store_true")

    p_swap = sub.add_parser("swap", help="exchange two ids over a frame window")
    p_swap.add_argument("video_dir", type=Path)
    p_swap.add_argument("id_a", type=int)
    p_swap.add_argument("id_b", type=int)
    p_swap.add_argument("start_frame", type=int)
    p_swap.add_argument("end_frame", type=int, help="inclusive")
    p_swap.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.op == "list":
        for pid, info in sorted(list_tracks(args.video_dir).items()):
            print(f"person {pid}: {info['num_frames']} frames, range {info['frame_range']}")
    elif args.op == "remove":
        remove_track(
            args.video_dir, args.person_id,
            start_frame=args.start_frame, end_frame=args.end_frame, dry_run=args.dry_run,
        )
    elif args.op == "rename":
        rename_track(args.video_dir, args.old_id, args.new_id, dry_run=args.dry_run)
    elif args.op == "merge":
        merge_tracks(
            args.video_dir, args.id_a, args.id_b,
            output_id=args.output_id, split_frame=args.split_frame, dry_run=args.dry_run,
        )
    elif args.op == "split":
        split_track(args.video_dir, args.person_id, args.split_frame, args.new_id, dry_run=args.dry_run)
    elif args.op == "swap":
        swap_tracks(
            args.video_dir, args.id_a, args.id_b,
            args.start_frame, args.end_frame, dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
