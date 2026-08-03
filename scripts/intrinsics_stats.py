#!/usr/bin/env python3
"""Raw camera-intrinsics statistics for the three evaluation datasets.

Reports, per dataset, the *raw released calibration* (before any undistortion,
cropping or resizing done by the ghost pipeline) of the cameras that ghost
actually used — i.e. the camera directories present in the ghost output tree.

Sources
-------
EgoExo4D   <gt_root>/<take>/gopro_calibs.csv
           cols: cam_uid, image_width, image_height, intrinsics_type,
                 intrinsics_0..7 (fx, fy, cx, cy, k1..k4), start/end_frame_idx
           model: KANNALABRANDTK3 (Kannala-Brandt fisheye, 4 radial terms)

EgoHumans  <activity>.sqsh : <INNER>/<activity>/<scene>/colmap/workplace/
                 cameras.txt  -> CAMERA_ID MODEL W H fx fy cx cy k1..k4
                 images.txt   -> image-name prefix ("cam01", "aria01") -> CAMERA_ID
           model: OPENCV_FISHEYE

RICH       <rich_root>/scan_calibration/<location>/calibration/<NNN>.xml
           cam_NN -> NNN.xml ; Intrinsics 3x3 + Distortion 8x1 (all zero -> pinhole)
           native resolution 4112x3008

Field of view
-------------
Pinhole   half-angle = atan(d / f)          with d = distance pp -> image border
Fisheye   r(theta) = f * (theta + k1 th^3 + k2 th^5 + k3 th^7 + k4 th^9)
          inverted numerically for r = d (equidistant Kannala-Brandt / OpenCV
          fisheye convention).  The naive pinhole-equivalent 2*atan(W/2fx) is
          printed alongside for reference — it badly under-estimates fisheye FOV.

Both horizontal and vertical FOV use the true principal point (left + right
half-angles summed), so an off-centre pp is accounted for.

Usage
-----
    pixi run python -m scripts.intrinsics_stats
    pixi run python -m scripts.intrinsics_stats --datasets rich egoexo
    pixi run python -m scripts.intrinsics_stats --out /tmp/intrinsics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# Default paths (CSCS Alps layout)
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:          # for utilities.rich_gender_plugin
    sys.path.insert(0, str(_REPO_ROOT))

EGOEXO_GT_ROOT = Path("/capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt")
EGOEXO_GHOST_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d")

EGOHUMANS_GHOST_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new")
# Unpacked GT tree (<activity>/<scene>/{processed_data/smpl, colmap/workplace}).
# Preferred source for the distances: it covers tennis and badminton 001-030,
# which the per-activity archives below do not.  It carries no cameras.txt, so
# the raw intrinsics still come from the archives.
EGOHUMANS_GT_DIR = Path("/iopsstor/scratch/cscs/tnanni/egohumans_gt_full")
EGOHUMANS_SQSH_DIR = Path("/capstor/scratch/cscs/tnanni/datasets")
EGOHUMANS_INNER = "media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"
# activity dir name -> candidate squashfs archives (first that provides the
# scene wins; badminton/tennis were packed in two parts)
EGOHUMANS_SQSH = {
    "01_tagging": ["egohumans_01_tagging.sqsh"],
    "02_lego": ["egohumans_02_lego.sqsh"],
    "03_fencing": ["egohumans_03_fencing.sqsh"],
    "04_basketball": ["egohumans_04_basketball.sqsh"],
    "05_volleyball": ["egohumans_05_volleyball.sqsh"],
    "06_badminton": ["egohumans_06_badminton_b.sqsh", "egohumans_06_badminton_a.sqsh"],
    "07_tennis": ["egohumans_07_tennis_new.sqsh", "egohumans_07_tennis.sqsh"],
}

RICH_ROOT = Path("/capstor/scratch/cscs/tnanni/datasets/rich")
RICH_GHOST_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test")
RICH_NATIVE_W, RICH_NATIVE_H = 4112, 3008  # full-res BMP size (verified on disk)


# --------------------------------------------------------------------------- #
# Camera record
# --------------------------------------------------------------------------- #

@dataclass
class Cam:
    dataset: str
    group: str          # activity / location / take — the "rig" grouping key
    scene: str
    name: str           # ghost camera dir name
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = field(default_factory=list)

    # ---- derived ---------------------------------------------------------- #
    @property
    def is_fisheye(self) -> bool:
        m = self.model.upper()
        return "FISHEYE" in m or "KANNALA" in m

    def fov(self) -> dict[str, float]:
        """Horizontal / vertical / diagonal FOV in degrees (pp-aware)."""
        dx_l, dx_r = self.cx, self.width - self.cx
        dy_t, dy_b = self.cy, self.height - self.cy
        dd = math.hypot(max(dx_l, dx_r), max(dy_t, dy_b))
        f_diag = math.sqrt(self.fx * self.fy)

        if self.is_fisheye:
            th = lambda r, f: _kb_theta(r, f, self.dist)
        else:
            th = lambda r, f: math.atan(r / f)

        h = math.degrees(th(dx_l, self.fx) + th(dx_r, self.fx))
        v = math.degrees(th(dy_t, self.fy) + th(dy_b, self.fy))
        d = math.degrees(2.0 * th(dd, f_diag))
        # pinhole-equivalent (ignores distortion) — reference value
        h_pin = math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))
        return {"hfov": h, "vfov": v, "dfov": d, "hfov_pinhole_equiv": h_pin}


def _kb_theta(r: float, f: float, k: list[float], n: int = 4096) -> float:
    """Invert the Kannala-Brandt / OpenCV-fisheye radial map r(theta) -> theta.

    r(theta) = f * (theta + k1 th^3 + k2 th^5 + k3 th^7 + k4 th^9)

    The polynomial is only physically meaningful while it is monotonically
    increasing, so we tabulate it up to the first turning point and invert by
    linear interpolation.  If the requested r lies beyond the tabulated range
    the last valid theta is returned (FOV then reads as a lower bound).
    """
    kk = (list(k) + [0.0, 0.0, 0.0, 0.0])[:4]
    th = np.linspace(0.0, math.pi, n)          # up to 360 deg total FOV
    poly = th + kk[0] * th**3 + kk[1] * th**5 + kk[2] * th**7 + kk[3] * th**9
    rr = f * poly
    # keep the strictly increasing prefix
    inc = np.diff(rr) > 0
    stop = int(np.argmin(inc)) + 1 if not inc.all() else n
    th, rr = th[:stop], rr[:stop]
    if r <= rr[0]:
        return 0.0
    if r >= rr[-1]:
        return float(th[-1])
    return float(np.interp(r, rr, th))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ghost_cam_dirs(scene_dir: Path) -> list[str]:
    """Camera directories ghost actually produced output for."""
    cams = [d.name for d in scene_dir.iterdir()
            if d.is_dir() and (d / "body_data").is_dir()]
    if not cams:  # tolerate trees where body_data was pruned
        cams = [d.name for d in scene_dir.iterdir()
                if d.is_dir() and not d.name.startswith(("_", "."))]
    return sorted(cams)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _stats(vals: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(vals), dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        "mean": float(a.mean()), "std": float(a.std()),
        "median": float(np.median(a)),
        "min": float(a.min()), "max": float(a.max()),
    }


# --------------------------------------------------------------------------- #
# EgoExo4D
# --------------------------------------------------------------------------- #

def collect_egoexo(gt_root: Path, ghost_root: Path) -> tuple[list[Cam], dict]:
    cams: list[Cam] = []
    skipped: dict[str, list[str]] = defaultdict(list)
    takes = sorted(d for d in ghost_root.iterdir() if d.is_dir())

    for take_dir in takes:
        take = take_dir.name
        gt_take = gt_root / take
        csv_path = gt_take / "gopro_calibs.csv"
        kp_path = gt_take / "keypoints_gt.json"
        if not csv_path.exists():
            skipped["no_calib_csv"].append(take)
            continue
        if not kp_path.exists():
            # eval requires GT keypoints — such takes are not in the eval set
            skipped["no_gt_keypoints"].append(take)
            continue

        # frame index used to pick the calibration row (same rule as the eval)
        gt_frame = 0
        try:
            with open(kp_path) as fh:
                kp = json.load(fh)
            if kp:
                gt_frame = int(next(iter(kp)))
            else:
                skipped["empty_gt_keypoints"].append(take)
                continue
        except (ValueError, json.JSONDecodeError):
            skipped["bad_gt_keypoints"].append(take)
            continue

        rows_by_cam: dict[str, list[dict]] = defaultdict(list)
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                rows_by_cam[row["cam_uid"]].append(row)

        used = _ghost_cam_dirs(take_dir)
        take_cams: list[Cam] = []
        for cam_name in used:
            rows = rows_by_cam.get(cam_name)
            if rows is None:
                for uid, r in rows_by_cam.items():
                    if _norm(uid) == _norm(cam_name):
                        rows = r
                        break
            if rows is None:
                skipped["cam_without_calib"].append(f"{take}/{cam_name}")
                continue

            row = rows[0]
            for r in rows:  # row whose frame range covers the GT frame
                try:
                    s, e = int(r["start_frame_idx"]), int(r["end_frame_idx"])
                except (KeyError, ValueError):
                    continue
                if s <= gt_frame <= e:
                    row = r
                    break

            take_cams.append(Cam(
                dataset="egoexo4d", group=take, scene=take, name=cam_name,
                model=row.get("intrinsics_type", "UNKNOWN"),
                width=int(float(row["image_width"])),
                height=int(float(row["image_height"])),
                fx=float(row["intrinsics_0"]), fy=float(row["intrinsics_1"]),
                cx=float(row["intrinsics_2"]), cy=float(row["intrinsics_3"]),
                dist=[float(row[f"intrinsics_{i}"]) for i in range(4, 8)
                      if row.get(f"intrinsics_{i}") not in (None, "")],
            ))

        if len(take_cams) < 2:
            skipped["fewer_than_2_calibrated_cams"].append(take)
            continue
        cams.extend(take_cams)

    return cams, dict(skipped)


# --------------------------------------------------------------------------- #
# EgoHumans (squashfs archives)
# --------------------------------------------------------------------------- #

class SqshMount:
    """Mount a squashfs image with squashfuse; unmount on exit."""

    def __init__(self, image: Path):
        self.image = image
        self.mnt: Path | None = None

    def __enter__(self) -> Path:
        self.mnt = Path(tempfile.mkdtemp(prefix="intr_sqsh_"))
        res = subprocess.run(["squashfuse", "-o", "ro", str(self.image), str(self.mnt)],
                             capture_output=True, text=True)
        if res.returncode != 0 or not any(self.mnt.iterdir()):
            self.__exit__(None, None, None)
            raise RuntimeError(f"squashfuse failed for {self.image.name}: "
                               f"{res.stderr.strip() or 'empty mount'}")
        return self.mnt

    def __exit__(self, *exc: Any) -> None:
        if self.mnt is None:
            return
        subprocess.run(["fusermount", "-u", str(self.mnt)], capture_output=True)
        shutil.rmtree(self.mnt, ignore_errors=True)
        self.mnt = None


def _parse_colmap_cameras(path: Path) -> dict[int, tuple[str, int, int, list[float]]]:
    """CAMERA_ID -> (model, width, height, params)."""
    out: dict[int, tuple[str, int, int, list[float]]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            out[int(p[0])] = (p[1], int(p[2]), int(p[3]), [float(x) for x in p[4:]])
    return out


def _parse_colmap_image_prefixes(path: Path, wanted: set[str]) -> dict[str, int]:
    """Image-name prefix -> CAMERA_ID, streamed, stops once all wanted found."""
    found: dict[str, int] = {}
    is_image_line = True
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if is_image_line:
                p = line.split()
                if len(p) >= 10:
                    prefix = p[9].split("/")[0]
                    if prefix not in found:
                        found[prefix] = int(p[8])
                        if wanted and wanted <= found.keys():
                            return found
            is_image_line = not is_image_line
    return found


def collect_egohumans(ghost_root: Path, sqsh_dir: Path) -> tuple[list[Cam], dict]:
    cams: list[Cam] = []
    skipped: dict[str, list[str]] = defaultdict(list)

    for act_dir in sorted(d for d in ghost_root.iterdir() if d.is_dir()):
        activity = act_dir.name
        scenes = sorted(d for d in act_dir.iterdir() if d.is_dir())
        pending = {s.name: s for s in scenes}

        for sqsh_name in EGOHUMANS_SQSH.get(activity, []):
            if not pending:
                break
            image = sqsh_dir / sqsh_name
            if not image.exists():
                skipped["missing_sqsh"].append(sqsh_name)
                continue
            try:
                with SqshMount(image) as mnt:
                    base = mnt / EGOHUMANS_INNER / activity
                    if not base.is_dir():
                        # the badminton parts keep their own activity folder name
                        cands = [p for p in (mnt / EGOHUMANS_INNER).iterdir()
                                 if p.is_dir() and p.name.startswith(activity[:2])]
                        base = cands[0] if cands else base
                    for scene_name in list(pending):
                        wp = base / scene_name / "colmap" / "workplace"
                        cam_txt, img_txt = wp / "cameras.txt", wp / "images.txt"
                        if not cam_txt.exists() or not img_txt.exists():
                            continue
                        used = _ghost_cam_dirs(pending[scene_name])
                        intr = _parse_colmap_cameras(cam_txt)
                        pref = _parse_colmap_image_prefixes(img_txt, set(used))
                        got = 0
                        for cam_name in used:
                            cid = pref.get(cam_name)
                            if cid is None or cid not in intr:
                                skipped["cam_without_colmap"].append(
                                    f"{activity}/{scene_name}/{cam_name}")
                                continue
                            model, w, h, prm = intr[cid]
                            cams.append(Cam(
                                dataset="egohumans", group=activity,
                                scene=f"{activity}/{scene_name}", name=cam_name,
                                model=model, width=w, height=h,
                                fx=prm[0], fy=prm[1], cx=prm[2], cy=prm[3],
                                dist=prm[4:],
                            ))
                            got += 1
                        if got:
                            pending.pop(scene_name)
            except RuntimeError as exc:
                skipped["sqsh_mount_failed"].append(f"{sqsh_name}: {exc}")

        for scene_name in pending:
            skipped["scene_without_colmap"].append(f"{activity}/{scene_name}")

    return cams, dict(skipped)


# --------------------------------------------------------------------------- #
# RICH
# --------------------------------------------------------------------------- #

def _rich_location(scene: str) -> str:
    m = re.match(r"^(.+?)_\d{3}_", scene)
    return m.group(1) if m else scene


def _parse_rich_xml(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(str(path)).getroot()

    def _mat(tag: str) -> np.ndarray | None:
        node = root.find(tag)
        if node is None:
            return None
        r, c = int(node.findtext("rows", "1")), int(node.findtext("cols", "1"))
        data = list(map(float, node.findtext("data", "").split()))
        return np.asarray(data, dtype=np.float64).reshape(r, c)

    return _mat("Intrinsics"), _mat("Distortion")


def collect_rich(rich_root: Path, ghost_root: Path) -> tuple[list[Cam], dict]:
    cams: list[Cam] = []
    skipped: dict[str, list[str]] = defaultdict(list)

    for scene_dir in sorted(d for d in ghost_root.iterdir() if d.is_dir()):
        scene = scene_dir.name
        loc = _rich_location(scene)
        calib_dir = rich_root / "scan_calibration" / loc / "calibration"
        if not calib_dir.is_dir():
            skipped["no_calib_dir"].append(scene)
            continue
        for cam_name in _ghost_cam_dirs(scene_dir):
            m = re.search(r"(\d+)$", cam_name)
            if m is None:
                skipped["unparsable_cam_name"].append(f"{scene}/{cam_name}")
                continue
            xml_path = calib_dir / f"{int(m.group(1)):03d}.xml"
            if not xml_path.exists():
                skipped["cam_without_calib"].append(f"{scene}/{cam_name}")
                continue
            K, D = _parse_rich_xml(xml_path)
            if K is None:
                skipped["bad_xml"].append(str(xml_path))
                continue
            dist = [] if D is None else [float(x) for x in D.reshape(-1)]
            model = "PINHOLE" if not any(abs(x) > 0 for x in dist) else "OPENCV_RATIONAL"
            cams.append(Cam(
                dataset="rich", group=loc, scene=scene, name=cam_name,
                model=model, width=RICH_NATIVE_W, height=RICH_NATIVE_H,
                fx=float(K[0, 0]), fy=float(K[1, 1]),
                cx=float(K[0, 2]), cy=float(K[1, 2]), dist=dist,
            ))

    return cams, dict(skipped)


# --------------------------------------------------------------------------- #
# Camera <-> subject distance
#
# One sample per (scene, used camera, subject, frame):
#     dist = || camera centre  -  subject pelvis ||   in metres
#
# Subject anchor per dataset:
#   EgoExo4D   mid-hip of the annotated 3D keypoints (centroid fallback)
#   EgoHumans  SMPL joint 0 of every GT person, aria01 world frame
#   RICH       SMPL-X J0(betas, gender) + transl  (= pelvis in world frame)
# --------------------------------------------------------------------------- #

def _record(dataset: str, group: str, scene: str, cam: str, subject: str,
            frame: int, centre: np.ndarray, pelvis: np.ndarray) -> dict:
    """One distance sample; ``ray`` is the unit subject->camera direction, kept
    so the inter-camera view angles can be computed afterwards."""
    v = np.asarray(centre, dtype=np.float64) - np.asarray(pelvis, dtype=np.float64)
    d = float(np.linalg.norm(v))
    return {"dataset": dataset, "group": group, "scene": scene, "cam": cam,
            "subject": subject, "frame": frame, "dist": d,
            "ray": (v / d).tolist() if d > 1e-9 else [0.0, 0.0, 0.0]}


def _sample(seq: list, k: int | None) -> list:
    """Deterministic uniform subsample of at most k elements, order preserved."""
    seq = list(seq)
    if k is None or k <= 0 or len(seq) <= k:
        return seq
    idx = sorted({int(round(i)) for i in np.linspace(0, len(seq) - 1, k)})
    return [seq[i] for i in idx]


def dist_egoexo(gt_root: Path, ghost_root: Path) -> tuple[list[dict], dict]:
    recs: list[dict] = []
    skipped: dict[str, list[str]] = defaultdict(list)

    for take_dir in sorted(d for d in ghost_root.iterdir() if d.is_dir()):
        take = take_dir.name
        kp_path = gt_root / take / "keypoints_gt.json"
        csv_path = gt_root / take / "gopro_calibs.csv"
        if not kp_path.exists() or not csv_path.exists():
            skipped["no_gt"].append(take)
            continue
        with open(kp_path) as fh:
            kp_raw = json.load(fh)
        if not kp_raw:
            skipped["empty_gt"].append(take)
            continue

        frame_str, joints_raw = next(iter(kp_raw.items()))
        pts, hips = [], []
        for jname, v in joints_raw.items():
            if not isinstance(v, dict) or v.get("num_views_for_3d", 0) <= 0:
                continue
            p = np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
            pts.append(p)
            if "hip" in jname.lower():
                hips.append(p)
        if not pts:
            skipped["no_valid_joints"].append(take)
            continue
        pelvis = np.mean(hips, axis=0) if len(hips) >= 2 else np.mean(pts, axis=0)

        centres: dict[str, np.ndarray] = {}
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                centres.setdefault(row["cam_uid"], np.array(
                    [float(row["tx_world_cam"]), float(row["ty_world_cam"]),
                     float(row["tz_world_cam"])], dtype=np.float64))

        for cam_name in _ghost_cam_dirs(take_dir):
            C = centres.get(cam_name)
            if C is None:
                for uid, c in centres.items():
                    if _norm(uid) == _norm(cam_name):
                        C = c
                        break
            if C is None:
                skipped["cam_without_calib"].append(f"{take}/{cam_name}")
                continue
            recs.append(_record("egoexo4d", take, take, cam_name,
                                "annotated_person", int(frame_str), C, pelvis))

    return recs, dict(skipped)


def _colmap_to_aria(workplace: Path) -> np.ndarray | None:
    """4x4 colmap-world -> aria01-world (similarity, scale included)."""
    p = workplace / "colmap_from_aria_transforms.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    if "aria01" not in d:
        return None
    return np.linalg.inv(np.asarray(d["aria01"], dtype=np.float64))


def _colmap_cam_centres(images_txt: Path, T_c2a: np.ndarray) -> dict[str, np.ndarray]:
    """{camera prefix: centre in aria world} from images.txt."""
    from scipy.spatial.transform import Rotation as SciR
    out: dict[str, np.ndarray] = {}
    is_image_line = True
    with open(images_txt) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if is_image_line:
                p = line.split()
                if len(p) >= 10:
                    prefix = p[9].split("/")[0]
                    if prefix not in out:
                        qw, qx, qy, qz = (float(x) for x in p[1:5])
                        t = np.array([float(x) for x in p[5:8]], dtype=np.float64)
                        R = SciR.from_quat([qx, qy, qz, qw]).as_matrix()
                        C = -R.T @ t                       # colmap world
                        out[prefix] = T_c2a[:3, :3] @ C + T_c2a[:3, 3]
            is_image_line = not is_image_line
    return out


def _egohumans_scene_records(gt_scene: Path, ghost_scene: Path, activity: str,
                             scene: str, max_frames: int, recs: list[dict],
                             skipped: dict[str, list[str]]) -> int:
    """Append the distance samples of one scene; return how many were added."""
    tag = f"{activity}/{scene}"
    smpl_dir = gt_scene / "processed_data" / "smpl"
    wp = gt_scene / "colmap" / "workplace"
    if not smpl_dir.is_dir() or not any(smpl_dir.glob("*.npy")):
        return 0
    T_c2a = _colmap_to_aria(wp)
    if T_c2a is None or not (wp / "images.txt").exists():
        skipped["no_colmap_alignment"].append(tag)
        return 0

    centres = _colmap_cam_centres(wp / "images.txt", T_c2a)
    used = _ghost_cam_dirs(ghost_scene)
    got = 0

    for npy in _sample(sorted(smpl_dir.glob("*.npy")), max_frames):
        try:
            frame = int(npy.stem)
        except ValueError:
            continue
        arr = np.load(str(npy), allow_pickle=True)
        d = arr.item() if arr.dtype == object and arr.shape == () else arr
        if not isinstance(d, dict):
            continue
        for aid, params in d.items():
            if not isinstance(params, dict) or "joints" not in params:
                continue
            j = np.asarray(params["joints"], dtype=np.float64)
            if j.ndim != 2 or j.shape[0] < 24 or j.shape[1] < 3:
                continue
            pelvis = j[0, :3]
            for cam_name in used:
                C = centres.get(cam_name)
                if C is None:
                    skipped["cam_without_colmap"].append(f"{tag}/{cam_name}")
                    continue
                recs.append(_record("egohumans", activity, tag, cam_name,
                                    str(aid), frame, C, pelvis))
                got += 1
    return got


def dist_egohumans(ghost_root: Path, sqsh_dir: Path, max_frames: int,
                   gt_dir: Path | None = None) -> tuple[list[dict], dict]:
    """Distances from the unpacked GT tree, falling back to the per-activity
    squashfs archives for whatever the tree does not provide."""
    recs: list[dict] = []
    skipped: dict[str, list[str]] = defaultdict(list)

    for act_dir in sorted(d for d in ghost_root.iterdir() if d.is_dir()):
        activity = act_dir.name
        pending = {d.name: d for d in sorted(act_dir.iterdir()) if d.is_dir()}

        # 1) unpacked GT tree
        if gt_dir is not None and (gt_dir / activity).is_dir():
            for scene in list(pending):
                if _egohumans_scene_records(gt_dir / activity / scene, pending[scene],
                                            activity, scene, max_frames, recs, skipped):
                    pending.pop(scene)

        # 2) squashfs archives for the remainder
        for sqsh_name in EGOHUMANS_SQSH.get(activity, []):
            if not pending:
                break
            image = sqsh_dir / sqsh_name
            if not image.exists():
                skipped["missing_sqsh"].append(sqsh_name)
                continue
            try:
                with SqshMount(image) as mnt:
                    base = mnt / EGOHUMANS_INNER / activity
                    if not base.is_dir():
                        cands = [p for p in (mnt / EGOHUMANS_INNER).iterdir()
                                 if p.is_dir() and p.name.startswith(activity[:2])]
                        base = cands[0] if cands else base
                    for scene in list(pending):
                        if _egohumans_scene_records(base / scene, pending[scene],
                                                    activity, scene, max_frames,
                                                    recs, skipped):
                            pending.pop(scene)
            except RuntimeError as exc:
                skipped["sqsh_mount_failed"].append(f"{sqsh_name}: {exc}")

        for scene in pending:
            skipped["scene_without_gt_smpl"].append(f"{activity}/{scene}")

    return recs, dict(skipped)


def _rich_cam_centres(calib_dir: Path, used: list[str]) -> dict[str, np.ndarray]:
    """{cam name: centre in RICH world} from the XML CameraMatrix ([R|t], w2c)."""
    out: dict[str, np.ndarray] = {}
    for cam_name in used:
        m = re.search(r"(\d+)$", cam_name)
        if m is None:
            continue
        xml_path = calib_dir / f"{int(m.group(1)):03d}.xml"
        if not xml_path.exists():
            continue
        node = ET.parse(str(xml_path)).getroot().find("CameraMatrix")
        if node is None:
            continue
        vals = list(map(float, node.findtext("data", "").split()))
        ext = np.asarray(vals, dtype=np.float64).reshape(3, 4)
        out[cam_name] = -ext[:3, :3].T @ ext[:3, 3]
    return out


def _smplx_pelvis_offsets(jobs: dict[Path, list[np.ndarray]]) -> dict[tuple, np.ndarray]:
    """{(model_path, betas_key): J0} — SMPL-X root joint in the rest pose."""
    import torch
    import smplx as smplx_lib

    out: dict[tuple, np.ndarray] = {}
    for model_path, betas_list in jobs.items():
        uniq = sorted({tuple(np.round(b, 5)) for b in betas_list})
        for start in range(0, len(uniq), 64):
            chunk = uniq[start:start + 64]
            betas = np.asarray(chunk, dtype=np.float32)
            model = smplx_lib.create(
                str(model_path), model_type="smplx",
                ext=Path(model_path).suffix.lstrip("."),
                num_betas=betas.shape[1], use_pca=False, flat_hand_mean=True,
                batch_size=betas.shape[0],
            ).eval()
            with torch.no_grad():
                joints = model(betas=torch.as_tensor(betas)).joints[:, 0].numpy()
            for key, j0 in zip(chunk, joints):
                out[(model_path, key)] = np.asarray(j0, dtype=np.float64)
    return out


def dist_rich(rich_root: Path, ghost_root: Path, split: str, max_frames: int,
              body_models_dir: Path, gender_json: Path) -> tuple[list[dict], dict]:
    recs: list[dict] = []
    skipped: dict[str, list[str]] = defaultdict(list)

    neutral = body_models_dir / "SMPLX_NEUTRAL.pkl"
    try:
        from utilities.rich_gender_plugin import resolve_smplx_models
    except Exception:                                     # noqa: BLE001
        resolve_smplx_models = None
        skipped["gender_plugin_unavailable"].append("falling back to SMPLX_NEUTRAL")

    # ---- pass 1: gather GT params + camera centres ------------------------ #
    pending: list[dict] = []
    jobs: dict[Path, list[np.ndarray]] = defaultdict(list)

    for scene_dir in sorted(d for d in ghost_root.iterdir() if d.is_dir()):
        scene = scene_dir.name
        gt_root = rich_root / f"{split}_body" / scene
        if not gt_root.is_dir():
            skipped["no_gt_body"].append(scene)
            continue
        calib_dir = rich_root / "scan_calibration" / _rich_location(scene) / "calibration"
        centres = _rich_cam_centres(calib_dir, _ghost_cam_dirs(scene_dir))
        if not centres:
            skipped["no_camera_centres"].append(scene)
            continue

        models: dict[int, Path] = {}
        if resolve_smplx_models is not None and gender_json.exists():
            try:
                models = resolve_smplx_models(scene, body_models_dir, gender_json)
            except Exception:                             # noqa: BLE001
                skipped["gender_lookup_failed"].append(scene)

        frame_dirs = _sample(
            [d for d in sorted(gt_root.iterdir()) if d.is_dir() and d.name.isdigit()],
            max_frames)
        for frame_dir in frame_dirs:
            frame = int(frame_dir.name)
            for pkl_path in sorted(frame_dir.glob("*.pkl")):
                pid = int(pkl_path.stem)
                with open(pkl_path, "rb") as fh:
                    data = pickle.load(fh)
                betas = np.asarray(data.get("betas", np.zeros(10)),
                                   dtype=np.float32).reshape(-1)[:10]
                transl = np.asarray(data["transl"], dtype=np.float64).reshape(3)
                mp = Path(models.get(pid, neutral))
                jobs[mp].append(betas)
                pending.append({"scene": scene, "pid": pid, "frame": frame,
                                "model": mp, "betas": betas, "transl": transl,
                                "centres": centres})

    if not pending:
        return recs, dict(skipped)

    # ---- pass 2: SMPL-X rest-pose root joint, then the distances ---------- #
    j0_lut = _smplx_pelvis_offsets(jobs)

    for e in pending:
        j0 = j0_lut.get((e["model"], tuple(np.round(e["betas"], 5))))
        if j0 is None:
            skipped["missing_j0"].append(f"{e['scene']}/{e['pid']}")
            continue
        pelvis = j0 + e["transl"]
        for cam_name, C in e["centres"].items():
            recs.append(_record("rich", _rich_location(e["scene"]), e["scene"],
                                cam_name, str(e["pid"]), e["frame"], C, pelvis))

    return recs, dict(skipped)


def view_angles(recs: list[dict]) -> list[dict]:
    """Inter-camera view angles, one entry per (scene, subject, frame).

    For every pair of cameras that sees the same subject in the same frame, the
    angle between the two subject->camera rays.  Gravity-independent: it says
    how far apart the viewpoints are, not where they sit relative to the floor.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        if "ray" in r:
            groups[(r["scene"], r["subject"], r["frame"])].append(r)

    out: list[dict] = []
    for (scene, subject, frame), rs in groups.items():
        if len(rs) < 2:
            continue
        V = np.asarray([r["ray"] for r in rs], dtype=np.float64)
        cos = np.clip(V @ V.T, -1.0, 1.0)
        iu = np.triu_indices(len(rs), k=1)
        ang = np.degrees(np.arccos(cos[iu]))
        out.append({"group": rs[0]["group"], "scene": scene, "subject": subject,
                    "frame": frame, "n_views": len(rs),
                    "mean_pair_deg": float(ang.mean()),
                    "max_pair_deg": float(ang.max())})
    return out


def print_distances(name: str, recs: list[dict], skipped: dict,
                    group_label: str) -> dict:
    def _gaps() -> None:
        if not skipped:
            return
        print("\n**Distance coverage gaps**")
        for reason, items in sorted(skipped.items()):
            head = ", ".join(items[:5]) + (" …" if len(items) > 5 else "")
            print(f"- `{reason}`: {len(items)} ({head})")

    print(f"\n### {name} — camera↔subject distance\n")
    if not recs:
        print("_no samples_")
        _gaps()
        print()
        return {}

    s = _stats(r["dist"] for r in recs)
    per_scene = defaultdict(list)
    for r in recs:
        per_scene[r["scene"]].append(r["dist"])
    scene_means = _stats(np.mean(v) for v in per_scene.values())

    print(f"- samples: **{len(recs)}**, scenes: **{len(per_scene)}**, "
          f"subjects: **{len({(r['scene'], r['subject']) for r in recs})}**")
    print("\n| quantity | value |")
    print("|---|---|")
    print(f"| distance (m) | {_fmt(s, 2)} |")
    print(f"| median (m) | {s['median']:.2f} |")
    print(f"| per-scene mean (m) | {_fmt(scene_means, 2)} |")

    va = view_angles(recs)
    va_mean = _stats(v["mean_pair_deg"] for v in va)
    va_max = _stats(v["max_pair_deg"] for v in va)
    va_n = _stats(v["n_views"] for v in va)
    if va:
        print(f"| inter-camera view angle (deg) | {_fmt(va_mean, 1)} |")
        print(f"| widest camera pair (deg) | {_fmt(va_max, 1)} |")
        print(f"| views per subject-frame | {_fmt(va_n, 2)} |")

    by_group = defaultdict(list)
    for r in recs:
        by_group[r["group"]].append(r["dist"])
    va_by_group = defaultdict(list)
    for v in va:
        va_by_group[v["group"]].append(v)
    if 1 < len(by_group) <= 40:
        print(f"\n| {group_label} | samples | mean (m) | median (m) | min | max "
              f"| view angle (deg) | widest pair (deg) |")
        print("|---|---|---|---|---|---|---|---|")
        for g in sorted(by_group):
            gs = _stats(by_group[g])
            gv = va_by_group.get(g, [])
            ang = f"{np.mean([v['mean_pair_deg'] for v in gv]):.1f}" if gv else "-"
            wide = f"{np.mean([v['max_pair_deg'] for v in gv]):.1f}" if gv else "-"
            print(f"| {g} | {len(by_group[g])} | {gs['mean']:.2f} | "
                  f"{gs['median']:.2f} | {gs['min']:.2f} | {gs['max']:.2f} | "
                  f"{ang} | {wide} |")

    _gaps()
    return {"overall": s, "per_scene_mean": scene_means,
            "n_samples": len(recs), "n_scenes": len(per_scene),
            "view_angle_mean_pair_deg": va_mean,
            "view_angle_max_pair_deg": va_max,
            "views_per_subject_frame": va_n}


# --------------------------------------------------------------------------- #
# Aggregation / reporting
# --------------------------------------------------------------------------- #

def summarize(cams: list[Cam]) -> dict[str, Any]:
    if not cams:
        return {}
    fovs = [c.fov() for c in cams]
    rigs = {(c.model, c.width, c.height,
             round(c.fx, 1), round(c.fy, 1), round(c.cx, 1), round(c.cy, 1))
            for c in cams}
    ndist = max(len(c.dist) for c in cams)
    dist_stats = {
        f"k{i + 1}": _stats([c.dist[i] for c in cams if len(c.dist) > i])
        for i in range(ndist)
    }
    return {
        "n_scenes": len({c.scene for c in cams}),
        "n_camera_instances": len(cams),
        "n_unique_intrinsics": len(rigs),
        "models": sorted({c.model for c in cams}),
        "resolutions": sorted({f"{c.width}x{c.height}" for c in cams}),
        "fx": _stats(c.fx for c in cams),
        "fy": _stats(c.fy for c in cams),
        "cx": _stats(c.cx for c in cams),
        "cy": _stats(c.cy for c in cams),
        "fx_over_width": _stats(c.fx / c.width for c in cams),
        "fy_over_height": _stats(c.fy / c.height for c in cams),
        "aspect_fy_over_fx": _stats(c.fy / c.fx for c in cams),
        "pp_offset_px": _stats(math.hypot(c.cx - c.width / 2, c.cy - c.height / 2)
                               for c in cams),
        "pp_offset_pct_width": _stats(
            100.0 * math.hypot(c.cx - c.width / 2, c.cy - c.height / 2) / c.width
            for c in cams),
        "hfov_deg": _stats(f["hfov"] for f in fovs),
        "vfov_deg": _stats(f["vfov"] for f in fovs),
        "dfov_deg": _stats(f["dfov"] for f in fovs),
        "hfov_pinhole_equiv_deg": _stats(f["hfov_pinhole_equiv"] for f in fovs),
        "distortion": dist_stats,
    }


def _fmt(s: dict[str, float], prec: int = 1) -> str:
    if not s:
        return "-"
    return (f"{s['mean']:.{prec}f} ± {s['std']:.{prec}f} "
            f"[{s['min']:.{prec}f}, {s['max']:.{prec}f}]")


def print_dataset(name: str, cams: list[Cam], skipped: dict, group_label: str) -> dict:
    print(f"\n## {name}\n")
    if not cams:
        print("_no cameras collected_\n")
        return {}

    summ = summarize(cams)
    print(f"- scenes: **{summ['n_scenes']}**, camera instances: "
          f"**{summ['n_camera_instances']}**, unique intrinsics: "
          f"**{summ['n_unique_intrinsics']}**")
    print(f"- model(s): {', '.join(summ['models'])}")
    print(f"- resolution(s): {', '.join(summ['resolutions'])}\n")

    rows = [
        ("fx (px)", _fmt(summ["fx"])),
        ("fy (px)", _fmt(summ["fy"])),
        ("cx (px)", _fmt(summ["cx"])),
        ("cy (px)", _fmt(summ["cy"])),
        ("fx / W", _fmt(summ["fx_over_width"], 3)),
        ("fy / H", _fmt(summ["fy_over_height"], 3)),
        ("fy / fx", _fmt(summ["aspect_fy_over_fx"], 4)),
        ("pp offset from centre (px)", _fmt(summ["pp_offset_px"])),
        ("pp offset (% of W)", _fmt(summ["pp_offset_pct_width"], 2)),
        ("HFOV (deg)", _fmt(summ["hfov_deg"], 2)),
        ("VFOV (deg)", _fmt(summ["vfov_deg"], 2)),
        ("DFOV (deg)", _fmt(summ["dfov_deg"], 2)),
        ("HFOV pinhole-equiv (deg)", _fmt(summ["hfov_pinhole_equiv_deg"], 2)),
    ]
    for k, v in summ["distortion"].items():
        rows.append((f"distortion {k}", _fmt(v, 4)))

    print("| quantity | mean ± std [min, max] |")
    print("|---|---|")
    for k, v in rows:
        print(f"| {k} | {v} |")

    # per-group breakdown
    by_group: dict[str, list[Cam]] = defaultdict(list)
    for c in cams:
        by_group[c.group].append(c)
    if len(by_group) > 1 and len(by_group) <= 40:
        print(f"\n### Per-{group_label} breakdown\n")
        print(f"| {group_label} | cams | model | res | fx | HFOV | VFOV | DFOV |")
        print("|---|---|---|---|---|---|---|---|")
        for g in sorted(by_group):
            gc = by_group[g]
            gs = summarize(gc)
            print(f"| {g} | {len(gc)} | {'/'.join(gs['models'])} | "
                  f"{'/'.join(gs['resolutions'])} | {gs['fx']['mean']:.1f} | "
                  f"{gs['hfov_deg']['mean']:.1f} | {gs['vfov_deg']['mean']:.1f} | "
                  f"{gs['dfov_deg']['mean']:.1f} |")

    if skipped:
        print("\n**Coverage gaps**")
        for reason, items in sorted(skipped.items()):
            head = ", ".join(items[:5]) + (" …" if len(items) > 5 else "")
            print(f"- `{reason}`: {len(items)} ({head})")
    return summ


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["egoexo", "egohumans", "rich"],
                    choices=["egoexo", "egohumans", "rich"])
    ap.add_argument("--egoexo_gt_root", type=Path, default=EGOEXO_GT_ROOT)
    ap.add_argument("--egoexo_ghost_root", type=Path, default=EGOEXO_GHOST_ROOT)
    ap.add_argument("--egohumans_ghost_root", type=Path, default=EGOHUMANS_GHOST_ROOT)
    ap.add_argument("--egohumans_sqsh_dir", type=Path, default=EGOHUMANS_SQSH_DIR)
    ap.add_argument("--rich_root", type=Path, default=RICH_ROOT)
    ap.add_argument("--rich_ghost_root", type=Path, default=RICH_GHOST_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="write JSON dump here")
    ap.add_argument("--distance", action="store_true",
                    help="also compute camera<->subject distances from the GT")
    ap.add_argument("--egohumans_gt_dir", type=Path, default=EGOHUMANS_GT_DIR,
                    help="unpacked EgoHumans GT tree used for the distances")
    ap.add_argument("--rich_split", default="test",
                    help="RICH GT body split, i.e. <split>_body/")
    ap.add_argument("--max_frames", type=int, default=20,
                    help="max GT frames sampled per scene for the distances")
    ap.add_argument("--body_models", type=Path, default=_REPO_ROOT / "body_models")
    ap.add_argument("--gender_json", type=Path,
                    default=_REPO_ROOT / "resource" / "rich_gender.json")
    args = ap.parse_args()

    print("# Raw camera-intrinsics statistics\n")
    print("Raw released calibration (pre-undistortion / pre-crop), restricted to the "
          "cameras used by the ghost pipeline in each evaluation set.")

    dump: dict[str, Any] = {}
    all_cams: dict[str, list[Cam]] = {}
    all_dists: dict[str, list[dict]] = {}

    if "egoexo" in args.datasets:
        cams, sk = collect_egoexo(args.egoexo_gt_root, args.egoexo_ghost_root)
        dump["egoexo4d"] = {"summary": print_dataset("EgoExo4D", cams, sk, "take"),
                            "skipped": sk}
        all_cams["egoexo4d"] = cams
        if args.distance:
            recs, dsk = dist_egoexo(args.egoexo_gt_root, args.egoexo_ghost_root)
            dump["egoexo4d"]["distance"] = print_distances("EgoExo4D", recs, dsk, "take")
            all_dists["egoexo4d"] = recs

    if "egohumans" in args.datasets:
        cams, sk = collect_egohumans(args.egohumans_ghost_root, args.egohumans_sqsh_dir)
        dump["egohumans"] = {"summary": print_dataset("EgoHumans", cams, sk, "activity"),
                             "skipped": sk}
        all_cams["egohumans"] = cams
        if args.distance:
            recs, dsk = dist_egohumans(args.egohumans_ghost_root, args.egohumans_sqsh_dir,
                                       args.max_frames, args.egohumans_gt_dir)
            dump["egohumans"]["distance"] = print_distances("EgoHumans", recs, dsk,
                                                            "activity")
            all_dists["egohumans"] = recs

    if "rich" in args.datasets:
        cams, sk = collect_rich(args.rich_root, args.rich_ghost_root)
        dump["rich_test"] = {"summary": print_dataset("RICH (test)", cams, sk, "location"),
                             "skipped": sk}
        all_cams["rich_test"] = cams
        if args.distance:
            recs, dsk = dist_rich(args.rich_root, args.rich_ghost_root, args.rich_split,
                                  args.max_frames, args.body_models, args.gender_json)
            dump["rich_test"]["distance"] = print_distances("RICH (test)", recs, dsk,
                                                            "location")
            all_dists["rich_test"] = recs

    if args.out:
        dump["cameras"] = {k: [asdict(c) | c.fov() for c in v]
                           for k, v in all_cams.items()}
        if all_dists:
            dump["distances"] = all_dists
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(dump, fh, indent=2)
        print(f"\n_JSON dump written to {args.out}_")


if __name__ == "__main__":
    main()
