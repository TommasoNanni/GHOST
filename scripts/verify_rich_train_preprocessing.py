"""Verify regenerated RICH train preprocessing.

Three independent checks:

1. CENTERED SOURCE  — the per-frame json_data mask dims must equal crop_meta's
   crop_w/crop_h from centered_train.sqsh. Raw images would give src_w/src_h instead.
   (Needs --centered_root pointing at a mount of centered_train.sqsh.)

2. COMPLETENESS     — per camera, every person id that appears in json_data detections
   should also have a body_data/person_<id>.npz. Reports detections with no body
   estimate and body files with no detections.

3. CROSS-VIEW MATCH — the ground truth annotates one subject in 3D (multi-cam frame).
   Project its root into every camera with the GT calibration, map the pixel into
   centered-crop space, and check which ghost person's bbox contains it. Cross-view
   ReID is correct iff the SAME global person id is hit in every camera.

Coordinate chain for check 3 (this is where mistakes hide):
    X_multicam --[R|t]--> camera --K--> ORIGINAL pixels
               --* scale--> archive pixels (1440 max side)
               --- off_x/off_y --> centered-crop pixels  == ghost bbox space
`scale` is src_w / orig_w; RICH originals are 4112 wide (4112/3008 matches the archive's
1440/1053 aspect), overridable with --orig_width. A wrong scale puts projections far
outside every bbox, so it cannot pass silently.

Usage
-----
    pixi run python scripts/verify_rich_train_preprocessing.py \\
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_root  /capstor/scratch/cscs/tnanni/datasets/rich \\
        --centered_root /tmp/centered_train --body_split train_body
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_calibration(rich_root: Path, scene: str) -> dict[int, dict]:
    """{cam_index: {'K': (3,3), 'RT': (3,4)}} from scan_calibration/<LOCATION>/calibration."""
    location = scene.split("_")[0]
    calib_dir = rich_root / "scan_calibration" / location / "calibration"
    out: dict[int, dict] = {}
    for xml_path in sorted(calib_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
            mats = {}
            for node in root:
                data = node.find("data")
                if data is None or data.text is None:
                    continue
                vals = [float(v) for v in data.text.split()]
                rows = int(node.find("rows").text)
                cols = int(node.find("cols").text)
                mats[node.tag] = np.array(vals, dtype=np.float64).reshape(rows, cols)
            if "Intrinsics" in mats and "CameraMatrix" in mats:
                out[int(xml_path.stem)] = {"K": mats["Intrinsics"], "RT": mats["CameraMatrix"]}
        except Exception:
            continue
    return out


def load_crop_meta(centered_root: Path | None, scene: str) -> dict[str, dict]:
    if centered_root is None:
        return {}
    p = centered_root / scene / "crop_meta.json"
    if not p.exists():
        return {}
    return json.load(open(p)).get("cameras", {})


def detections_by_frame(cam_dir: Path) -> dict[int, dict[int, tuple]]:
    """{frame: {person_id: (x1,y1,x2,y2)}} plus the mask dims seen."""
    out: dict[int, dict[int, tuple]] = {}
    dims: set[tuple[int, int]] = set()
    for jp in sorted((cam_dir / "json_data").glob("mask_*.json")):
        m = re.match(r"mask_(\d+)", jp.stem)
        if not m:
            continue
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        dims.add((d.get("mask_width"), d.get("mask_height")))
        frame = int(m.group(1))
        out[frame] = {
            int(pid): (lab["x1"], lab["y1"], lab["x2"], lab["y2"])
            for pid, lab in (d.get("labels") or {}).items()
        }
    out["_dims"] = dims          # type: ignore[index]
    return out


def gt_root_translation(rich_root: Path, body_split: str, scene: str, frame: int) -> np.ndarray | None:
    """GT root translation (3,) in the multi-cam frame, or None."""
    fdir = rich_root / body_split / scene / f"{frame:05d}"
    if not fdir.is_dir():
        return None
    pkls = sorted(fdir.glob("*.pkl"))
    if not pkls:
        return None
    try:
        d = pickle.load(open(pkls[0], "rb"))
    except Exception:
        return None
    t = d.get("transl")
    return np.asarray(t, dtype=np.float64).reshape(-1)[:3] if t is not None else None


def project(X: np.ndarray, K: np.ndarray, RT: np.ndarray) -> tuple[np.ndarray, float]:
    """3-D point in multi-cam frame -> ORIGINAL-resolution pixel, plus depth."""
    Xc = RT[:, :3] @ X + RT[:, 3]
    z = float(Xc[2])
    if z <= 1e-6:
        return np.array([np.nan, np.nan]), z
    uv = K @ (Xc / z)
    return uv[:2], z


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_scene(scene_dir: Path, rich_root: Path, centered_root: Path | None,
                body_split: str, orig_width: int, n_frames: int) -> dict:
    scene = scene_dir.name
    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and (d / "body_data").is_dir())
    crop_meta = load_crop_meta(centered_root, scene)
    calib = load_calibration(rich_root, scene)

    res = {"scene": scene, "n_cams": len(cam_dirs), "centered": [], "completeness": [],
           "match": {"checked": 0, "consistent": 0, "per_cam_hits": defaultdict(int),
                     "no_bbox_hit": 0, "ids": defaultdict(int)}}

    det_cache: dict[str, dict] = {}
    for cam_dir in cam_dirs:
        dets = detections_by_frame(cam_dir)
        dims = dets.pop("_dims", set())                     # type: ignore[arg-type]
        det_cache[cam_dir.name] = dets

        # --- 1. centered source ---
        cm = crop_meta.get(cam_dir.name)
        if cm and dims:
            got = next(iter(dims))
            exp_crop = (cm["crop_w"], cm["crop_h"])
            exp_src = (cm["src_w"], cm["src_h"])
            verdict = ("CENTERED" if got == exp_crop else
                       "RAW/SRC" if got == exp_src else f"UNKNOWN {got}")
            res["centered"].append((cam_dir.name, verdict))

        # --- 2. completeness ---
        det_ids = {pid for f, pm in dets.items() for pid in pm}
        body_ids = {int(p.stem.split("_")[1])
                    for p in (cam_dir / "body_data").glob("person_*.npz")}
        res["completeness"].append({
            "cam": cam_dir.name, "detected": len(det_ids), "estimated": len(body_ids),
            "missing_body": sorted(det_ids - body_ids),
            "orphan_body": sorted(body_ids - det_ids),
        })

    # --- 3. cross-view matching via GT projection ---
    gt_dir = rich_root / body_split / scene
    if gt_dir.is_dir() and calib:
        frames = sorted(int(p.name) for p in gt_dir.iterdir() if p.name.isdigit())
        step = max(1, len(frames) // max(n_frames, 1))
        for frame in frames[::step][:n_frames]:
            X = gt_root_translation(rich_root, body_split, scene, frame)
            if X is None:
                continue
            hits: dict[str, int] = {}
            for cam_dir in cam_dirs:
                cam_idx = int(re.sub(r"\D", "", cam_dir.name))
                cal, cm = calib.get(cam_idx), crop_meta.get(cam_dir.name)
                dets = det_cache[cam_dir.name].get(frame)
                if cal is None or cm is None or not dets:
                    continue
                uv, z = project(X, cal["K"], cal["RT"])
                if not np.isfinite(uv).all():
                    continue
                scale = cm["src_w"] / float(orig_width)
                u = uv[0] * scale - cm["off_x"]
                v = uv[1] * scale - cm["off_y"]
                # Among the bboxes containing the projection, take the one whose centre
                # is nearest. Overlapping people otherwise make the pick dict-order
                # dependent, which would fabricate cross-view disagreements.
                cands = [(pid, bb) for pid, bb in dets.items()
                         if bb[0] <= u <= bb[2] and bb[1] <= v <= bb[3]]
                if cands:
                    hits[cam_dir.name] = min(
                        cands,
                        key=lambda pb: ((u - (pb[1][0] + pb[1][2]) / 2) ** 2
                                        + (v - (pb[1][1] + pb[1][3]) / 2) ** 2),
                    )[0]
            if len(hits) >= 2:
                res["match"]["checked"] += 1
                ids = set(hits.values())
                if len(ids) == 1:
                    res["match"]["consistent"] += 1
                for c, pid in hits.items():
                    res["match"]["per_cam_hits"][c] += 1
                    res["match"]["ids"][pid] += 1
                    res["match"].setdefault("cam_ids", defaultdict(lambda: defaultdict(int)))
                    res["match"]["cam_ids"][c][pid] += 1
            elif len(hits) == 0:
                res["match"]["no_bbox_hit"] += 1
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_root", required=True, type=Path)
    ap.add_argument("--rich_root", required=True, type=Path)
    ap.add_argument("--centered_root", type=Path, default=None,
                    help="mount of centered_train.sqsh (needed for checks 1 and 3)")
    ap.add_argument("--body_split", default="train_body")
    ap.add_argument("--orig_width", type=int, default=4112,
                    help="true original image width the GT intrinsics refer to")
    ap.add_argument("--frames", type=int, default=8, help="GT frames sampled per scene")
    ap.add_argument("--scenes", default="", help="comma-separated subset")
    ap.add_argument("--only_complete", action="store_true",
                    help="only scenes with cross_view_reid.json (finished by the pipeline)")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scenes = sorted(d for d in args.ghost_root.iterdir() if d.is_dir())
    if wanted:
        scenes = [d for d in scenes if d.name in wanted]
    if args.only_complete:
        scenes = [d for d in scenes if (d / "cross_view_reid.json").exists()]

    rows = [check_scene(d, args.rich_root, args.centered_root,
                        args.body_split, args.orig_width, args.frames) for d in scenes]

    # ---- report ----
    print(f"\n{'='*78}\n1. CENTERED SOURCE\n{'='*78}")
    tally = defaultdict(int)
    for r in rows:
        for _, v in r["centered"]:
            tally[v.split()[0]] += 1
    if tally:
        for k, v in tally.items():
            print(f"  {k:10s} {v} cameras")
    else:
        print("  (skipped — pass --centered_root)")

    print(f"\n{'='*78}\n2. COMPLETENESS (detections vs body estimates)\n{'='*78}")
    bad = 0
    for r in rows:
        for c in r["completeness"]:
            if c["missing_body"] or c["orphan_body"]:
                bad += 1
                print(f"  {r['scene']:<32} {c['cam']:<8} det={c['detected']:<3} "
                      f"est={c['estimated']:<3} missing={c['missing_body']} "
                      f"orphan={c['orphan_body']}")
    print(f"  cameras with a mismatch: {bad} / {sum(len(r['completeness']) for r in rows)}")

    print(f"\n{'='*78}\n3. CROSS-VIEW MATCH (GT projected into every view)\n{'='*78}")
    print("  A scene passes when every camera's bbox hit carries the SAME global id")
    print("  (the id itself is arbitrary — it is whatever ReID assigned).\n")
    print(f"  {'scene':<34}{'frames':>7}{'allcams':>9}{'cam-agree':>11}  majority / dissenting cams")
    tot_c = tot_k = 0
    tot_hit = tot_ok = 0
    dissent_tally: dict[str, int] = defaultdict(int)
    for r in rows:
        m = r["match"]
        if not m["checked"]:
            continue
        tot_c += m["consistent"]; tot_k += m["checked"]
        maj = max(m["ids"].items(), key=lambda kv: kv[1])[0] if m["ids"] else None
        cam_ids = m.get("cam_ids", {})
        bad = []
        for cam, counter in sorted(cam_ids.items()):
            cam_maj = max(counter.items(), key=lambda kv: kv[1])[0]
            n = sum(counter.values())
            tot_hit += n
            tot_ok += counter.get(maj, 0)
            if cam_maj != maj:
                bad.append(f"{cam}->{cam_maj}")
                dissent_tally[cam] += 1
        agree = 100.0 * sum(c.get(maj, 0) for c in cam_ids.values()) / max(
            sum(sum(c.values()) for c in cam_ids.values()), 1)
        flag = "" if not bad else "  " + ",".join(bad)
        print(f"  {r['scene']:<34}{m['checked']:>7}{m['consistent']:>9}"
              f"{agree:>10.0f}%  id={maj}{flag}")
    if tot_k:
        print(f"\n  frames where ALL cameras agree : {tot_c}/{tot_k} ({100*tot_c/tot_k:.1f}%)")
        print(f"  camera-level agreement          : {tot_ok}/{tot_hit} "
              f"({100*tot_ok/max(tot_hit,1):.1f}%)")
        if dissent_tally:
            print("  cameras that dissent, by frequency:")
            for cam, n in sorted(dissent_tally.items(), key=lambda kv: -kv[1]):
                print(f"    {cam}: {n} scenes")
    else:
        print("  (no frames checked — need --centered_root and GT)")


if __name__ == "__main__":
    main()
