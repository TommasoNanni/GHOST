"""Animate the IRLS iterations of the geodesic-median rotation fusion.

GHOST fuses the per-camera SAM3D body poses with a geodesic median on SO(3),
solved by Weiszfeld/IRLS: seed with the chordal mean, then repeatedly re-take a
chordal mean with each camera weighted by 1/(theta_k + eps), theta_k being its
geodesic distance to the current estimate.  A camera that disagrees loses weight,
which is why one bad view drags the mean but not the median.

Three panels, left to right:

  * the exponential chart at the seed — every camera's rotation as an
    axis x angle vector, which is the space the algorithm actually measures in.
    Nothing is discarded here: SO(3) is three-dimensional and this is all three
    dimensions, so the "sphere" is a ball and distances from its centre are the
    thetas that set the weights.
  * a real unit sphere, showing where each camera's rotation sends the bone
    axis of the joint.  This is the intuitive picture, but it is a projection:
    a rotation about the bone axis does not move the dot, so two different
    rotations can coincide.
  * theta and weight per camera, updated each iteration.

Body-pose joints are parent-relative, so the per-camera rotations are directly
comparable without any camera alignment — only the root orientation would need
it, and the root is not fused this way.

The estimator is copied from median_fuse in evaluation/evaluate_egohumans_sync.py
rather than imported, so this animation cannot drift from what it draws.

Usage:
    pixi run python visualize/animate_irls_rotations.py \\
        --scene-dir /.../egohumans_new/02_lego/001_legoassemble \\
        --frame 140 --pid 1 --out out_irls --mp4 irls.mp4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#ffffff"
INK = "#1a1a1a"
MUTED = "#8a9098"
ACCENT = "#c62828"
SEED_C = "#0f9ed5"
CAM_C = ["#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#c2185b", "#5d4037"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": BG,
})


# ── SO(3) helpers ───────────────────────────────────────────────────────────
def aa_to_R(aa: np.ndarray) -> np.ndarray:
    """(...,3) axis-angle -> (...,3,3) rotation matrices (Rodrigues)."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.divide(aa, theta, out=np.zeros_like(aa), where=theta > 1e-12)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    t = theta[..., None]
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def log_R(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,3) axis-angle."""
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2],
                  R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    s = np.sin(theta)[..., None]
    small = np.abs(s) < 1e-8
    out = np.divide(v * theta[..., None], 2 * s, out=v * 0.5, where=~small)
    return out


def geodesic(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Angle in radians between rotations, elementwise over leading dims."""
    cos = (np.trace(A @ np.swapaxes(B, -1, -2), axis1=-2, axis2=-1) - 1.0) * 0.5
    return np.arccos(np.clip(cos, -1 + 1e-7, 1 - 1e-7))


def chordal_mean(R: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted chordal mean: the SVD projection of the weighted average matrix.

    Copied from median_fuse._chordal — every IRLS step is this same closed form.
    """
    M = (R * w[:, None, None]).sum(0) / max(w.sum(), 1e-8)
    U, _, Vh = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vh))
    return U @ D @ Vh


def irls(R: np.ndarray, iters: int, eps: float = 1e-3):
    """Weiszfeld geodesic median; returns the seed and every iterate with its
    thetas and weights, which is what the animation walks through."""
    w0 = np.ones(len(R))
    seed = chordal_mean(R, w0)
    steps = []
    R_bar = seed
    for _ in range(iters):
        theta = geodesic(R, R_bar[None])
        w = w0 / (theta + eps)
        R_bar = chordal_mean(R, w)
        steps.append({"R": R_bar.copy(), "theta": theta.copy(), "w": w.copy()})
    return seed, steps


# ── data ────────────────────────────────────────────────────────────────────
def load_joint_rotations(scene_dir: Path, cams: list[str], pid: int, frame: int,
                         body_dir: str):
    """(K, 21, 3, 3) per-camera body-joint rotations at one frame, and the cams
    that actually had a detection there."""
    R, kept = [], []
    for cam in cams:
        f = scene_dir / cam / body_dir / f"person_{pid}.npz"
        if not f.exists():
            continue
        with np.load(str(f)) as d:
            if "smplx_body_pose" not in d.files:
                continue
            fi = d["frame_indices"].astype(int)
            hit = np.nonzero(fi == frame)[0]
            if hit.size == 0:
                continue
            aa = d["smplx_body_pose"][int(hit[0])].reshape(-1, 3)   # (21,3)
        R.append(aa_to_R(aa))
        kept.append(cam)
    if len(R) < 2:
        sys.exit(f"need >= 2 cameras with person {pid} at frame {frame}, got {len(R)}")
    return np.stack(R), kept


def spread_per_joint(R: np.ndarray) -> np.ndarray:
    """Mean pairwise geodesic angle per joint — how much the cameras disagree."""
    K, J = R.shape[0], R.shape[1]
    out = np.zeros(J)
    for j in range(J):
        d = [geodesic(R[a, j], R[b, j]) for a in range(K) for b in range(a + 1, K)]
        out[j] = float(np.mean(d)) if d else 0.0
    return out


def bone_axis(joint: int, smplx_model_path: Path) -> np.ndarray:
    """Rest-pose direction from this joint to its child, in the parent frame.

    The sphere panel shows where each rotation sends this axis; without the real
    bone the panel would be an arbitrary vector and would mean nothing.
    """
    import torch
    import smplx as smplx_lib
    p = Path(smplx_model_path)
    kw = {"model_path": str(p), "ext": p.suffix.lstrip(".")} if p.is_file() \
        else {"model_path": str(p), "ext": "pkl"}
    model = smplx_lib.create(**kw, model_type="smplx", gender="neutral",
                             use_pca=False, num_betas=10, batch_size=1)
    with torch.no_grad():
        J = model(return_verts=False).joints[0, :55].numpy()
    parents = model.parents.cpu().numpy()
    kids = np.nonzero(parents == joint + 1)[0]      # body_pose j is SMPL-X joint j+1
    if kids.size == 0:
        return np.array([0.0, -1.0, 0.0])
    v = J[kids[0]] - J[joint + 1]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, -1.0, 0.0])


# ── figure ──────────────────────────────────────────────────────────────────
JOINT_NAMES = [
    "left hip", "right hip", "spine 1", "left knee", "right knee", "spine 2",
    "left ankle", "right ankle", "spine 3", "left foot", "right foot", "neck",
    "left collar", "right collar", "head", "left shoulder", "right shoulder",
    "left elbow", "right elbow", "left wrist", "right wrist",
]


def style_3d(ax, label: str):
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
        pane.line.set_color((1, 1, 1, 0))
    ax.set_title(label, color=MUTED, fontsize=14, pad=0)


def basis_around(c: np.ndarray):
    """Two unit vectors spanning the plane orthogonal to c."""
    tmp = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(c, tmp)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(c, e1)


def project_cap(d: np.ndarray, c: np.ndarray, e1: np.ndarray, e2: np.ndarray):
    """Sphere -> disk, azimuthal equidistant about c: radius = angle in degrees.

    mplot3d fights this figure — it sorts by depth, so an opaque patch swallows
    the markers and the labels.  Projecting first gives the same geometry with
    none of that: a point at 10 degrees from the centre sits at radius 10.
    """
    d = np.atleast_2d(d)
    cosang = np.clip(d @ c, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    tang = d - cosang[:, None] * c
    n = np.linalg.norm(tang, axis=1, keepdims=True)
    unit = np.divide(tang, n, out=np.zeros_like(tang), where=n > 1e-12)
    return np.stack([ang * (unit @ e1), ang * (unit @ e2)], axis=1)


def draw_globe(ax, radius_deg: float, rings_deg: list[float]):
    """A disk shaded like a lit sphere, with a graticule, as the backdrop."""
    n = 420
    g = np.linspace(-radius_deg, radius_deg, n)
    X, Y = np.meshgrid(g, g)
    rr = np.sqrt(X ** 2 + Y ** 2) / radius_deg
    inside = rr <= 1.0
    z = np.sqrt(np.clip(1 - rr ** 2, 0, 1))            # the sphere's own normal
    shade = np.clip(0.55 * z + 0.45 * np.clip((-X / radius_deg + Y / radius_deg)
                                              * 0.5 + 0.7, 0, 1), 0, 1)
    rgba = np.zeros((n, n, 4))
    base = np.array([0.80, 0.855, 0.895])
    rgba[..., :3] = base[None, None] * (0.72 + 0.38 * shade)[..., None]
    rgba[..., 3] = inside.astype(float)
    ax.imshow(np.clip(rgba, 0, 1), origin="lower",
              extent=(-radius_deg, radius_deg, -radius_deg, radius_deg), zorder=0)
    t = np.linspace(0, 2 * np.pi, 200)
    for deg in rings_deg:
        if deg > radius_deg:
            continue
        ax.plot(deg * np.cos(t), deg * np.sin(t), color="#8fa0ad", linewidth=1.0,
                alpha=0.8, zorder=1)
        ax.text(deg * 0.72, deg * 0.72, f"{deg:g}°", color="#5f6c75", fontsize=11,
                zorder=2, ha="center", va="center")
    for k in range(8):        # meridians
        a = k * np.pi / 4
        ax.plot([0, radius_deg * np.cos(a)], [0, radius_deg * np.sin(a)],
                color="#9fb0bd", linewidth=0.7, alpha=0.7, zorder=1)
    ax.plot(radius_deg * np.cos(t), radius_deg * np.sin(t), color="#7d8d99",
            linewidth=1.4, zorder=2)
    ax.set_xlim(-radius_deg * 1.12, radius_deg * 1.12)
    ax.set_ylim(-radius_deg * 1.12, radius_deg * 1.12)
    ax.set_aspect("equal")
    ax.axis("off")


def sphere_cap(ax, c: np.ndarray, half_angle: float, rings_deg: list[float]):
    """Draw the piece of the unit sphere around c, with angular scale rings.

    A whole sphere is mostly empty at these scales — the cameras disagree by a
    few degrees — so only the patch holding them is drawn, and the rings say how
    many degrees wide that patch is.
    """
    e1, e2 = basis_around(c)
    aa = np.linspace(0, half_angle, 24)
    bb = np.linspace(0, 2 * np.pi, 64)
    A, B = np.meshgrid(aa, bb)
    P = (np.cos(A)[..., None] * c
         + np.sin(A)[..., None] * (np.cos(B)[..., None] * e1 + np.sin(B)[..., None] * e2))
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color="#d9e3ea",
                    edgecolor="none", alpha=1.0, shade=True, zorder=0,
                    rcount=48, ccount=48)
    for deg in rings_deg:
        r = np.radians(deg)
        if r > half_angle:
            continue
        ring = (np.cos(r) * c
                + np.sin(r) * (np.cos(bb)[:, None] * e1 + np.sin(bb)[:, None] * e2))
        ring *= 1.006                      # sit the line ON the patch, not in it
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#94a3ad",
                linewidth=1.3, zorder=1)
        lab = (np.cos(r) * c + np.sin(r) * e1) * 1.02
        ax.text(*lab, f" {deg:g}°", color="#6b7780", fontsize=11, zorder=2)
    for k in range(8):        # meridians, so the patch reads as curved
        b0 = 2 * np.pi * k / 8
        mer = (np.cos(aa)[:, None] * c
               + np.sin(aa)[:, None] * (np.cos(b0) * e1 + np.sin(b0) * e2))
        mer *= 1.004
        ax.plot(mer[:, 0], mer[:, 1], mer[:, 2], color="#b3bfc8",
                linewidth=0.9, zorder=1)
    ax.view_init(elev=np.degrees(np.arcsin(np.clip(c[2], -1, 1))) + 26,
                 azim=np.degrees(np.arctan2(c[1], c[0])) + 12)
    span = np.sin(half_angle) * 0.82
    ctr = np.cos(half_angle * 0.5) * c
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)


def scale_rings(ax, lim: float, rings_deg: list[float], plane):
    """Concentric circles at fixed angles, in one plane only.

    In the log chart the distance from the centre IS theta, so the rings are a
    protractor.  Drawing them in all three coordinate planes turns the panel
    into a ball of wire, so only the plane facing the viewer is drawn.
    """
    e1, e2 = plane
    t = np.linspace(0, 2 * np.pi, 96)[:, None]
    for deg in rings_deg:
        r = np.radians(deg)
        if r > lim:
            continue
        ring = r * (np.cos(t) * e1 + np.sin(t) * e2)
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#dfe4e8",
                linewidth=0.8, zorder=0)
        lab = r * (np.cos(np.pi / 4) * e1 + np.sin(np.pi / 4) * e2)
        ax.text(*lab, f"{deg:g}°", color="#b9c0c7", fontsize=10, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--cams", nargs="+", default=None)
    ap.add_argument("--body-dir", default="body_data_clean")
    ap.add_argument("--pid", type=int, default=1)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--joint", type=int, default=None,
                    help="body_pose index 0..20; default = the one the cameras "
                         "disagree on most")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--sub", type=int, default=14, help="frames per IRLS step")
    ap.add_argument("--smplx-model", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "body_models" / "SMPLX_NEUTRAL.pkl")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--pose-dir", type=Path, default=None,
                    help="dir of per-camera mesh PNGs (…/<cam>/**/<cam>_f<frame>.png) "
                         "from visualize_for_presentation --still --blank-bg --joint")
    ap.add_argument("--out", type=Path, default=Path("out_irls"))
    ap.add_argument("--mp4", type=Path, default=None)
    args = ap.parse_args()

    cams = args.cams or sorted(d.name for d in args.scene_dir.iterdir()
                               if d.is_dir() and (d / args.body_dir).is_dir())
    R_all, cams = load_joint_rotations(args.scene_dir, cams, args.pid, args.frame,
                                       args.body_dir)
    spread = spread_per_joint(R_all)
    joint = args.joint if args.joint is not None else int(np.argmax(spread))
    print(f"cameras {cams} | joint {joint} | cross-camera spread "
          f"{np.degrees(spread[joint]):.1f} deg (max of {np.degrees(spread.max()):.1f})")

    R = R_all[:, joint]                                   # (K,3,3)
    seed, steps = irls(R, args.iters)
    axis = bone_axis(joint, args.smplx_model)

    # chart at the seed: v_k = log(seed^T R_k), so the cluster is centred and the
    # estimate's motion is visible against a fixed frame
    v_cam = log_R(np.einsum("ij,kjl->kil", seed.T, R))
    v_path = [np.zeros(3)] + [log_R(seed.T @ s["R"]) for s in steps]
    lim = float(np.linalg.norm(v_cam, axis=1).max()) * 1.25
    rings = [d for d in (2, 5, 10, 20, 30, 45) if np.radians(d) < lim][-2:]
    # the ring plane faces the camera of the left panel (elev 20, azim -62)
    ev, az = np.radians(20.0), np.radians(-62.0)
    view = np.array([np.cos(ev) * np.cos(az), np.cos(ev) * np.sin(az), np.sin(ev)])
    ball_plane = basis_around(view)

    d_cam = R @ axis                                       # (K,3) on the sphere
    d_seed = seed @ axis
    d_path = [d_seed] + [s["R"] @ axis for s in steps]
    # the cap: centred on the cameras, wide enough to hold them with room to read
    cap_c = d_cam.mean(0) / np.linalg.norm(d_cam.mean(0))
    spread_ang = float(np.arccos(np.clip(d_cam @ cap_c, -1, 1)).max())
    cap_half = float(np.clip(spread_ang * 1.6, np.radians(3), np.radians(60)))
    _ev = np.arcsin(np.clip(cap_c[2], -1, 1)) + np.radians(26)
    _az = np.arctan2(cap_c[1], cap_c[0]) + np.radians(12)
    cap_view = np.array([np.cos(_ev) * np.cos(_az), np.cos(_ev) * np.sin(_az),
                         np.sin(_ev)])
    cap_e1, cap_e2 = basis_around(cap_c)
    rings = [d for d in (2, 5, 10, 20, 30, 45)
             if d < np.degrees(cap_half) * 0.95][-3:]

    # the poses being fused, one per camera, tinted with that camera's colour
    poses = {}
    if args.pose_dir:
        for cam in cams:
            hit = sorted(args.pose_dir.glob(f"**/{cam}_f{args.frame}.png"))
            if hit:
                poses[cam] = plt.imread(str(hit[0]))
    if not poses:
        sys.exit("--pose-dir given no images; render them with "
                 "visualize_for_presentation --still --blank-bg --joint")

    fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi)
    outer = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.35, 0.62],
                             left=0.015, right=0.975, top=0.83, bottom=0.07,
                             wspace=0.10)
    ncol = 2 if len(cams) > 2 else len(cams)
    nrow = int(np.ceil(len(cams) / ncol))
    pose_gs = outer[0, 0].subgridspec(nrow, ncol, wspace=0.02, hspace=0.18)
    ax_p = [fig.add_subplot(pose_gs[i // ncol, i % ncol]) for i in range(len(cams))]
    ax_s = fig.add_subplot(outer[0, 1])
    ax_w = fig.add_subplot(outer[0, 2])

    pose_im, pose_lbl = [], []
    for i, cam in enumerate(cams):
        ax = ax_p[i]
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        img = poses.get(cam)
        pose_im.append(ax.imshow(img) if img is not None else None)
        ax.set_title(cam, color=CAM_C[i % len(CAM_C)], fontsize=13,
                     fontweight="bold", pad=3)
        pose_lbl.append(ax.text(0.5, -0.02, "", transform=ax.transAxes,
                                ha="center", va="top", fontsize=12,
                                color=CAM_C[i % len(CAM_C)]))
    fig.text(0.13, 0.865, "the poses being fused", color=MUTED, fontsize=14)

    head = fig.text(0.5, 0.95, "", ha="center", fontsize=28, color=INK,
                    fontweight="bold")
    fig.text(0.5, 0.045, "■ chordal mean (seed)     ★ current estimate     "
             "● camera, size = weight", ha="center", fontsize=12, color=MUTED)
    fig.text(0.5, 0.012, "the patch shows the bone axis only — a twist about it "
             "does not move a dot", ha="center", fontsize=10, color="#b9c0c7")
    name = JOINT_NAMES[joint] if joint < len(JOINT_NAMES) else f"joint {joint}"
    fig.text(0.5, 0.905,
             f"geodesic median · {name} (SMPL-X joint {joint + 1}) · frame {args.frame}",
             ha="center", fontsize=13, color=MUTED)

    ax_w.set_xlim(0, 1.05)
    ax_w.set_ylim(-0.75, len(cams) - 0.25)
    ax_w.invert_yaxis()
    ax_w.set_yticks(range(len(cams)))
    ax_w.set_yticklabels(cams, color=MUTED, fontsize=12)
    ax_w.set_xticks([])
    ax_w.set_title("weight  1/(θ+ε)", color=MUTED, fontsize=12, pad=8)
    for side in ("top", "right", "left"):
        ax_w.spines[side].set_visible(False)
    ax_w.spines["bottom"].set_color("#c7ccd1")
    bars = ax_w.barh(range(len(cams)), [0] * len(cams),
                     color=[CAM_C[i % len(CAM_C)] for i in range(len(cams))],
                     height=0.22)
    texts = [ax_w.text(0.02, i + 0.30, "", color=MUTED, fontsize=11)
             for i in range(len(cams))]

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.png"):
        old.unlink()
    counter = 0

    view_b = np.array([np.cos(np.radians(20)) * np.cos(np.radians(-62)),
                       np.cos(np.radians(20)) * np.sin(np.radians(-62)),
                       np.sin(np.radians(20))])

    def draw(v_now, d_now, theta, w, trail_v, trail_d, title):
        nonlocal counter
        ax_s.clear()
        ax_s.set_title("the bone axis on the sphere", color=MUTED, fontsize=14,
                       pad=6)

        wn = w / max(w.max(), 1e-9)
        for i, im in enumerate(pose_im):
            if im is not None:
                im.set_alpha(0.25 + 0.75 * wn[i])       # weight, on the pose itself
            pose_lbl[i].set_text(f"θ = {np.degrees(theta[i]):.1f}°")

        radius_deg = np.degrees(cap_half)
        draw_globe(ax_s, radius_deg, rings)
        P_cam = project_cap(d_cam, cap_c, cap_e1, cap_e2)
        for i, xy in enumerate(P_cam):
            ax_s.scatter(*xy, s=150 + 330 * wn[i], color=CAM_C[i % len(CAM_C)],
                         edgecolor="white", linewidth=1.6, zorder=5)
            off = xy / max(np.linalg.norm(xy), 1e-9) * radius_deg * 0.17
            ax_s.text(*(xy + off), cams[i], color=CAM_C[i % len(CAM_C)],
                      fontsize=13, fontweight="bold", ha="center", va="center",
                      zorder=6)
        xy_seed = project_cap(d_seed, cap_c, cap_e1, cap_e2)[0]
        ax_s.scatter(*xy_seed, marker="s", s=130, color=SEED_C, edgecolor="white",
                     linewidth=1.6, zorder=6)
        if len(trail_d) > 1:
            tr = project_cap(np.stack(trail_d), cap_c, cap_e1, cap_e2)
            ax_s.plot(tr[:, 0], tr[:, 1], color=ACCENT, linewidth=2.6, zorder=7)
        xy_now = project_cap(d_now, cap_c, cap_e1, cap_e2)[0]
        ax_s.scatter(*xy_now, s=420, marker="*", color=ACCENT, edgecolor=INK,
                     linewidth=1.0, zorder=8)

        wn = w / max(w.max(), 1e-9)
        for i, b in enumerate(bars):
            b.set_width(wn[i])
            texts[i].set_text(f"θ = {np.degrees(theta[i]):.1f}°")
        head.set_text(title)
        fig.savefig(args.out / f"{counter:04d}.png")
        counter += 1

    def slerp_v(a, b, t):
        return a + (b - a) * t

    theta0 = geodesic(R, seed[None])
    w_flat = np.ones(len(cams))
    for _ in range(int(0.8 * args.fps)):
        draw(v_path[0], d_path[0], theta0, w_flat, [v_path[0]], [d_path[0]],
             "chordal mean  (every camera equal)")

    trail_v, trail_d = [v_path[0]], [d_path[0]]
    for s_i, st in enumerate(steps):
        for t in np.linspace(0, 1, args.sub, endpoint=True):
            v_now = slerp_v(v_path[s_i], v_path[s_i + 1], t)
            d_now = d_path[s_i] + (d_path[s_i + 1] - d_path[s_i]) * t
            d_now = d_now / max(np.linalg.norm(d_now), 1e-9)
            th = theta0 + (st["theta"] - theta0) * t if s_i == 0 else \
                steps[s_i - 1]["theta"] + (st["theta"] - steps[s_i - 1]["theta"]) * t
            w = w_flat + (st["w"] - w_flat) * t if s_i == 0 else \
                steps[s_i - 1]["w"] + (st["w"] - steps[s_i - 1]["w"]) * t
            draw(v_now, d_now, th, w, trail_v + [v_now], trail_d + [d_now],
                 f"IRLS step {s_i + 1}")
        trail_v.append(v_path[s_i + 1])
        trail_d.append(d_path[s_i + 1])

    last = steps[-1]
    moved = np.degrees(geodesic(seed[None], last["R"][None])[0])
    for _ in range(int(1.6 * args.fps)):
        draw(v_path[-1], d_path[-1], last["theta"], last["w"], trail_v, trail_d,
             f"median  ({moved:.1f}° from the mean)")

    print(f"{counter} frames -> {args.out}/%04d.png")
    if args.mp4:
        encode(args.out, args.mp4, args.fps)


def encode(png_dir: Path, mp4: Path, fps: int) -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print(f"no ffmpeg; run:\n  ffmpeg -framerate {fps} -i {png_dir}/%04d.png "
                  f"-c:v libx264 -pix_fmt yuv420p {mp4}")
            return
    mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "-y", "-framerate", str(fps), "-i", str(png_dir / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
                   check=True, capture_output=True)
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
