#!/usr/bin/env python
"""Materialise cleaned, globally-reid'd body tracks into ``body_data_clean/``.

Pipeline, per scene / camera (originals are never modified):
  1. copy ``body_data/`` to a scratch dir
  2. apply the within-view ops from ``manual_operations.json``
     (order: split -> remove -> merge -> swap -> rename), body_data only
  3. relabel the surviving tracks to their GLOBAL person id using the
     ``groups`` in ``manual_reid.json`` (token ``cXpY`` = camera X, post-ops
     local id Y -> the group key is the global id), dropping any track that
     is not referenced by a group
  4. write ``person_<GLOBAL>.npz`` into ``<cam>/body_data_clean/`` plus a
     ``reid_map.json`` recording the local->global mapping.

So ``body_data_clean/person_4.npz`` means the same real person across every
camera of that scene — which is what evaluation consumes.

Usage:
  pixi run python scripts/build_clean_body_data.py                 # all egohumans scenes
  pixi run python scripts/build_clean_body_data.py --scene 009_basketball
  pixi run python scripts/build_clean_body_data.py --activity 06_badminton
  pixi run python scripts/build_clean_body_data.py --dry-run       # report only, write nothing
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/users/tnanni/ghost")
import utilities.within_reid_operations as w  # noqa: E402

# body_data only: never rewrite mask_data.npz / json_data (legacy, slow, stale ids)
w._apply_frame_actions = lambda *a, **k: None

REPO = Path("/users/tnanni/ghost")
DEFAULT_OUT = "/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new"
OPS_PATH = REPO / "manual_operations.json"
REID_PATH = REPO / "manual_reid.json"


def cam_num(cam: str) -> int:
    return int(re.sub(r"\D", "", cam))


def apply_ops(cam_dir: Path, ops: dict) -> None:
    for e in ops.get("split", []):
        w.split_track(cam_dir, e["person_id"], e["split_frame"], e["new_id"])
    for e in ops.get("remove", []):
        w.remove_track(cam_dir, e["person_id"], e.get("start_frame"), e.get("end_frame"))
    for e in ops.get("merge", []):
        w.merge_tracks(cam_dir, e["id_a"], e["id_b"],
                       output_id=e.get("output_id"), split_frame=e.get("split_frame"))
    for e in ops.get("swap", []):
        w.swap_tracks(cam_dir, e["id_a"], e["id_b"], e["start_frame"], e["end_frame"])
    for e in ops.get("rename", []):
        w.rename_track(cam_dir, e["old_id"], e["new_id"])


def local_to_global(groups: dict, cnum: int) -> dict[int, int]:
    """post-ops local id -> global id, for one camera."""
    l2g: dict[int, int] = {}
    for gid, toks in groups.items():
        for t in toks:
            m = re.fullmatch(r"c(\d+)p(\d+)", t.strip())
            if m and int(m.group(1)) == cnum:
                l2g[int(m.group(2))] = int(gid)
    return l2g


def build_cam(cam_dir_src: Path, ops: dict, groups: dict, dry: bool):
    body = cam_dir_src / "body_data"
    if not body.is_dir():
        return None  # nothing to do

    tmp_root = Path(tempfile.mkdtemp(prefix="clean_"))
    try:
        work = tmp_root / cam_dir_src.name
        work.mkdir(parents=True)
        shutil.copytree(body, work / "body_data")
        apply_ops(work, ops)

        l2g = local_to_global(groups, cam_num(cam_dir_src.name))
        present = {int(os.path.basename(f).split("_")[1][:-4])
                   for f in glob.glob(str(work / "body_data" / "person_*.npz"))}

        written, missing = [], []
        for local, gid in sorted(l2g.items()):
            src = work / "body_data" / f"person_{local}.npz"
            (written if src.exists() else missing).append((local, gid) if src.exists() else local)

        if not dry:
            clean = cam_dir_src / "body_data_clean"
            shutil.rmtree(clean, ignore_errors=True)
            clean.mkdir(parents=True)
            for local, gid in written:
                shutil.copy(work / "body_data" / f"person_{local}.npz", clean / f"person_{gid}.npz")
            json.dump(
                {"local_to_global": {str(k): v for k, v in sorted(l2g.items())},
                 "written_global_ids": sorted(g for _, g in written),
                 "missing_local_ids": missing},
                open(clean / "reid_map.json", "w"), indent=2)
        return {"written": written, "missing": missing,
                "dropped": sorted(present - set(l2g))}
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="ghost_outputs root")
    ap.add_argument("--dataset", default="egohumans", help="manual_reid top-level key")
    ap.add_argument("--activity", default=None, help="restrict to one activity dir (e.g. 06_badminton)")
    ap.add_argument("--scene", default=None, help="restrict to one scene (e.g. 009_basketball)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    out = Path(args.out)
    ops_all = json.load(open(OPS_PATH)).get(args.dataset, {})
    reid_all = json.load(open(REID_PATH))[args.dataset]

    # scene -> activity dir
    scene_act = {}
    for act in sorted(os.listdir(out)):
        ad = out / act
        if ad.is_dir():
            for sc in os.listdir(ad):
                if (ad / sc).is_dir():
                    scene_act[sc] = act

    n_scene = n_cam = n_track = 0
    warnings = []
    for scene in sorted(reid_all):
        if args.scene and scene != args.scene:
            continue
        if scene not in scene_act:
            warnings.append(f"{scene}: not found under {out}")
            continue
        if args.activity and scene_act[scene] != args.activity:
            continue
        scene_dir = out / scene_act[scene] / scene
        groups = reid_all[scene].get("groups", {})
        ops_scene = ops_all.get(scene, {})
        cams = sorted(p.name for p in scene_dir.iterdir()
                      if p.is_dir() and (p / "body_data").is_dir())
        if not cams:
            continue
        n_scene += 1
        for cam in cams:
            res = build_cam(scene_dir / cam, ops_scene.get(cam, {}), groups, args.dry_run)
            if res is None:
                continue
            n_cam += 1
            n_track += len(res["written"])
            tag = "DRY " if args.dry_run else ""
            note = ""
            if res["missing"]:
                note += f"  !! MISSING local {res['missing']}"
                warnings.append(f"{scene}/{cam}: missing local ids {res['missing']}")
            print(f"{tag}{scene}/{cam}: {len(res['written'])} clean tracks "
                  f"-> global {sorted(g for _, g in res['written'])}"
                  f"  (dropped {res['dropped']}){note}", flush=True)

    print(f"\n=== {'DRY-RUN ' if args.dry_run else ''}DONE: "
          f"{n_scene} scenes, {n_cam} cams, {n_track} clean tracks ===")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for wmsg in warnings:
            print("  ", wmsg)


if __name__ == "__main__":
    main()
