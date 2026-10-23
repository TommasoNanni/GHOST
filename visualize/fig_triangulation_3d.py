"""One frame of the reconstruction, in 3-D: cameras, rays, triangulated joints, body.

What the placer does for a single (person, frame), drawn instead of described:

  * every camera that saw the person contributes a 2-D keypoint per joint,
  * each of those is a ray from that camera centre through the image plane,
  * the rays for one joint are intersected by DLT — that is the black point,
  * the SMPL-X body is then fitted to those points (Procrustes on the fused
    pose), which is the mesh.

Geometry comes from BodyPlacer itself — the same intrinsics, extrinsics, image
rectification and `_triangulate_dlt` the pipeline runs — so the picture cannot
show a different reconstruction from the one that is measured.  The body pose is
the geodesic median over cameras, as shipped.

Everything is in metres: the VGGT extrinsics are scaled by the MapAnything
baseline scale, exactly as the placer scales them.

Usage:
    pixi run python visualize/fig_triangulation_3d.py \\
        --scene-dir /.../egohumans_new/02_lego/001_legoassemble \\
        --frame 140 --pid 1 --out tri3d.png
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from fusion.placer import BodyPlacer, _SMPLX_TO_MHR70                  # noqa: E402
from visualize.render_flyover import _clean_scene_view                 # noqa: E402
from visualize.visualize_for_presentation import body_faces_only       # noqa: E402

logger = logging.getLogger("tri3d")

BG = "#ffffff"
INK = "#1a1a1a"
MUTED = "#8a9098"
CAM_C = ["#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#c2185b", "#5d4037"]
BODY_C = "#3fae52"
POINT_C = "#c62828"


# ── SO(3) (the shipped estimator, in numpy) ─────────────────────────────────
def aa_to_R(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.divide(aa, theta, out=np.zeros_like(aa), where=theta > 1e-12)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    t = theta[..., None]
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def chordal_mean(R: np.ndarray, w: np.ndarray) -> np.ndarray:
    M = (R * w[:, None, None]).sum(0) / max(w.sum(), 1e-8)
    U, _, Vh = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vh))
    return U @ D @ Vh


def geodesic_median(R: np.ndarray, iters: int = 5, eps: float = 1e-3) -> np.ndarray:
    """median_fuse for one joint: chordal seed, then IRLS with w = 1/(theta+eps)."""
    w0 = np.ones(len(R))
    R_bar = chordal_mean(R, w0)
    for _ in range(iters):
        cos = np.clip((np.trace(R @ R_bar.T, axis1=-2, axis2=-1) - 1) * 0.5, -1, 1)
        R_bar = chordal_mean(R, w0 / (np.arccos(cos) + eps))
    return R_bar


def frustum(K: np.ndarray, ext: np.ndarray, w: float, h: float, depth: float):
    """(centre, 4 image-plane corners) in world metres for one camera."""
    R, t = ext[:3, :3], ext[:3, 3]
    C = -R.T @ t
    corners_px = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)
    rays = (np.linalg.inv(K) @ corners_px.T).T          # camera-space directions
    rays = rays / rays[:, 2:3] * depth
    return C, (R.T @ (rays - t[None]).T).T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--pid", type=int, default=1)
    ap.add_argument("--frame-start", type=int, default=1,
                    help="scene frame that VGGT index 0 corresponds to")
    ap.add_argument("--smplx-model", type=Path,
                    default=_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl")
    ap.add_argument("--body-dir", default="body_data_clean")
    ap.add_argument("--frustum-depth", type=float, default=0.22, metavar="M")
    ap.add_argument("--body-alpha", type=float, default=0.32,
                    help="the triangulated joints sit INSIDE the mesh; this is "
                         "what lets them show through")
    ap.add_argument("--margin", type=float, default=0.04,
                    help="fraction of the data extent left as blank border")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video", action="store_true",
                    help="animate every frame instead of drawing one")
    ap.add_argument("--frames", type=int, nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--step", type=int, default=1, help="frame stride for --video")
    ap.add_argument("--mesh-stride", type=int, default=1,
                    help="draw every Nth body triangle; 3 is ~3x faster and still "
                         "reads as a body under 32%% opacity")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    vggt = np.load(args.scene_dir / "vggt_cameras_centered.npz", allow_pickle=False)
    cam_names = [n.decode() if isinstance(n, bytes) else n for n in vggt["camera_names"]]
    view = _clean_scene_view(args.scene_dir, cam_names) \
        if args.body_dir != "body_data" else args.scene_dir
    try:
        placer = BodyPlacer(view, str(args.smplx_model), crop_meta_path=None)
        scale = placer.load_mapanything_scale(filename="mapanything_scale_baseline.npy")
        if scale is None:
            sys.exit("mapanything_scale_baseline.npy missing/mismatched")
        vggt_t = args.frame - args.frame_start
        s = float(scale[vggt_t]) if isinstance(scale, np.ndarray) else float(scale)
        logger.info(f"frame {args.frame} -> vggt index {vggt_t}, scale {s:.4f} m/unit")

        # every camera's whole track, read once
        tracks = {}
        for k, cam in enumerate(cam_names):
            f = args.scene_dir / cam / args.body_dir / f"person_{args.pid}.npz"
            if not f.exists():
                continue
            with np.load(str(f)) as d:
                tracks[k] = {"fi": {int(g): i for i, g in
                                    enumerate(d["frame_indices"].astype(int))},
                             "kp2d": d["pred_keypoints_2d"],
                             "pose": d["smplx_body_pose"],
                             "betas": d["smplx_betas"]}

        def observations(frame: int):
            vt = frame - args.frame_start
            out = {}
            if vt < 0 or vt >= placer.T:
                return out, vt
            for k, tr in tracks.items():
                if frame not in tr["fi"] or not placer.cam_valid[vt, k]:
                    continue
                lt = tr["fi"][frame]
                out[k] = {"kp2d": tr["kp2d"][lt],
                          "pose": tr["pose"][lt].reshape(-1, 3),
                          "betas": tr["betas"][lt]}
            return out, vt

        per_cam, vggt_t = observations(args.frame)
        if len(per_cam) < 2:
            sys.exit(f"person {args.pid} seen by {len(per_cam)} camera(s) at frame "
                     f"{args.frame}; triangulation needs at least 2")
        logger.info(f"cameras with a detection: {[cam_names[k] for k in per_cam]}")

        joints = sorted(j for j in _SMPLX_TO_MHR70
                        if _SMPLX_TO_MHR70[j] in placer._mapper._index)

        def triangulate(obs_by_cam: dict, vt: int, s_frame: float):
            """The placer's own DLT, joint by joint, plus the rays that fed it."""
            world_f, rays_f = {}, []
            for smplx_j in joints:
                jid = _SMPLX_TO_MHR70[smplx_j]
                obs, pmats, wts, used = [], [], [], []
                for k, e in obs_by_cam.items():
                    if jid >= e["kp2d"].shape[0]:
                        continue
                    oc = placer.original_coords[vt, k]
                    os_ = placer.original_size[vt, k]
                    off_x, off_y = placer._cam_offset(k)
                    u, v = placer._orig_to_vggt(e["kp2d"][jid], oc, float(os_[0]),
                                                float(os_[1]), off_x, off_y)
                    if not placer._in_bounds(u, v, oc[2], oc[3]):
                        continue
                    intr = placer.intrinsics[vt, k].astype(np.float64)
                    ext = placer.extrinsics[vt, k].astype(np.float64).copy()
                    ext[:3, 3] *= s_frame
                    pmats.append(intr @ ext)
                    obs.append((u, v))
                    used.append((k, -ext[:3, :3].T @ ext[:3, 3]))
                    cos_a = float(np.clip(ext[2, 2], -1.0, 1.0)) if k > 0 else 1.0
                    wts.append(1.0 if k == 0 else max(1.0 - cos_a ** 2, 0.0))
                if len(obs) >= 2:
                    X = placer._triangulate_dlt(obs, pmats, wts)
                    world_f[smplx_j] = X
                    rays_f += [(k, C, X) for k, C in used]
            return world_f, rays_f

        def fuse(obs_by_cam: dict):
            """median_fuse for one frame: geodesic median per joint over cameras."""
            R_cams = np.stack([aa_to_R(e["pose"]) for e in obs_by_cam.values()])
            fR = np.stack([geodesic_median(R_cams[:, j]) for j in range(R_cams.shape[1])])
            six = np.concatenate([fR[:, 0, :], fR[:, 1, :]], axis=-1)
            betas = np.mean([e["betas"] for e in obs_by_cam.values()], 0).astype(np.float32)
            return six, betas

        if args.video:
            if args.frames:
                lo, hi = args.frames
            else:
                seen = set().union(*(set(t["fi"]) for t in tracks.values()))
                lo, hi = min(seen), max(seen)
            wanted = list(range(lo, hi + 1, max(1, args.step)))
        else:
            wanted = [args.frame]

        # geometry per frame, then ONE placement solve for all of them: the
        # placer call is the expensive part and it accepts the whole set at once
        from fusion.placer import _6d_to_aa_batch
        geo, pose_arr, betas_acc = {}, np.zeros((placer.T, 54, 6), np.float32), []
        for fr in wanted:
            obs_by_cam, vt = observations(fr)
            if len(obs_by_cam) < 2:
                continue
            s_fr = float(scale[vt]) if isinstance(scale, np.ndarray) else float(scale)
            w_f, r_f = triangulate(obs_by_cam, vt, s_fr)
            if len(w_f) < 3:
                continue
            six, betas_f = fuse(obs_by_cam)
            pose_arr[vt, :21] = six
            betas_acc.append(betas_f)
            geo[fr] = {"world": w_f, "rays": r_f, "six": six, "vt": vt}
        if not geo:
            sys.exit("no frame had two cameras and three joints")
        betas = np.mean(betas_acc, 0).astype(np.float32)
        logger.info(f"{len(geo)}/{len(wanted)} frames reconstructable")

        trans_d, orient_d = placer.estimate_procrustes_dlt_mhr(
            scale=scale, all_pids={args.pid}, pred_betas_by_pid={args.pid: betas},
            fused_pose_by_pid={args.pid: pose_arr}, frame_start=args.frame_start,
            smooth_window=0, frames=set(geo))
        placed = [f for f in sorted(geo)
                  if trans_d.get(args.pid, {}).get(f) is not None
                  and orient_d.get(args.pid, {}).get(f) is not None]
        if not placed:
            sys.exit("the placer placed nothing")
        logger.info(f"{len(placed)}/{len(geo)} frames placed")

        bp_aa = np.stack([_6d_to_aa_batch(geo[f]["six"]).reshape(63) for f in placed])
        J55, V = placer._smplx_fk(np.repeat(betas[None], len(placed), 0),
                                  bp_aa.astype(np.float32),
                                  np.zeros((len(placed), 3), np.float32),
                                  return_verts=True)
        J55, V = np.asarray(J55).astype(np.float64), np.asarray(V).astype(np.float64)
        faces = body_faces_only(np.asarray(placer._smplx_model.faces, np.int32))
        faces = faces[::max(1, args.mesh_stride)]

        # Up, averaged over EVERY placed frame and then fixed.  Taking it from a
        # single frame puts the scene on a tilt whenever the person happened to
        # be bent over in that frame — in a lego-assembly clip that is most of
        # them, and the whole figure ends up looking top-down.  Averaging works
        # because people spend more time upright than folded, and the mean of
        # the spine directions is dominated by the upright ones.
        ups = []
        for i, f in enumerate(placed):
            u = orient_d[args.pid][f] @ (J55[i, 15] - J55[i, 0])
            n_u = np.linalg.norm(u)
            if n_u > 1e-9:
                ups.append(u / n_u)
        up = np.mean(ups, axis=0)
        up = up / max(np.linalg.norm(up), 1e-9)
        spread = np.degrees(np.arccos(np.clip(np.array(ups) @ up, -1, 1)))
        logger.info(f"up from {len(ups)} frames; spine tilts {spread.mean():.0f}° "
                    f"mean, {spread.max():.0f}° max from it")
        a = np.cross(up, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(a)
        if n < 1e-9:
            R_up = np.eye(3)
        else:
            a = a / n
            ang = np.arccos(np.clip(up @ np.array([0.0, 0.0, 1.0]), -1, 1))
            Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R_up = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * (Kx @ Kx)

        def W(x):
            return (R_up @ np.atleast_2d(x).T).T

        world_by_f, rays_by_f, verts_by_f = {}, {}, {}
        for i, f in enumerate(placed):
            pw, R_m = trans_d[args.pid][f], orient_d[args.pid][f]
            verts_by_f[f] = W((R_m @ (V[i] - J55[i, 0]).T).T + pw)
            world_by_f[f] = {j: W(x)[0] for j, x in geo[f]["world"].items()}
            rays_by_f[f] = [(k, W(C)[0], W(X)[0]) for k, C, X in geo[f]["rays"]]

        vt0 = geo[placed[0]]["vt"]
        s0 = float(scale[vt0]) if isinstance(scale, np.ndarray) else float(scale)
        frusta = {}
        for k in sorted({kk for f in placed for kk, _, _ in rays_by_f[f]}):
            ext = placer.extrinsics[vt0, k].astype(np.float64).copy()
            ext[:3, 3] *= s0
            oc = placer.original_coords[vt0, k]
            C, corners = frustum(placer.intrinsics[vt0, k].astype(np.float64), ext,
                                 float(oc[2]), float(oc[3]), args.frustum_depth)
            frusta[k] = (W(C)[0], W(corners))

        allpts = np.concatenate([np.concatenate([verts_by_f[f] for f in placed]),
                                 np.stack([c for c, _ in frusta.values()]),
                                 np.concatenate([np.stack(list(world_by_f[f].values()))
                                                 for f in placed])])
        lo_, hi_ = allpts.min(0), allpts.max(0)
        pad = (hi_ - lo_) * args.margin
        lo_, hi_ = lo_ - pad, hi_ + pad
        centre = allpts.mean(0)
        parents = placer._smplx_model.parents.cpu().numpy()
        light = np.array([0.4, -0.7, 0.6])
        light /= np.linalg.norm(light)
        base = np.array(matplotlib.colors.to_rgb(BODY_C))

        def draw(frame_id: int, path: Path):
            fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi)
            ax = fig.add_axes((-0.06, -0.10, 1.12, 1.12), projection="3d")
            ax.set_axis_off()
            ax.set_facecolor(BG)
            for k, (C, corners) in frusta.items():
                col = CAM_C[k % len(CAM_C)]
                for i in range(4):
                    ax.plot(*np.stack([C, corners[i]]).T, color=col, linewidth=1.0)
                    ax.plot(*np.stack([corners[i], corners[(i + 1) % 4]]).T,
                            color=col, linewidth=1.0)
                ax.add_collection3d(Poly3DCollection([corners], facecolor=col,
                                                     alpha=0.16, edgecolor="none"))
                ax.scatter(*C, s=26, color=col, depthshade=False)
                ax.text(*(C + (C - centre) * 0.05), cam_names[k], color=col,
                        fontsize=12, fontweight="bold")
            for k, C, X in rays_by_f[frame_id]:
                ax.plot(*np.stack([C, X]).T, color=CAM_C[k % len(CAM_C)],
                        linewidth=0.5, alpha=0.22)
            wf = world_by_f[frame_id]
            for j, X in wf.items():
                pj = int(parents[j]) if j < len(parents) else -1
                if pj in wf:
                    ax.plot(*np.stack([wf[pj], X]).T, color=POINT_C, linewidth=2.0,
                            alpha=0.85)
            P = np.stack(list(wf.values()))
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=95, color=POINT_C,
                       depthshade=False, edgecolor="white", linewidth=1.3)
            tri = verts_by_f[frame_id][faces]
            fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
            lam = np.clip(np.abs(fn @ light), 0.0, 1.0)
            cols = np.clip(base[None] * (0.45 + 0.65 * lam)[:, None], 0, 1)
            ax.add_collection3d(Poly3DCollection(tri, facecolors=cols,
                                                 edgecolor="none",
                                                 alpha=args.body_alpha))
            ax.set_xlim(lo_[0], hi_[0])
            ax.set_ylim(lo_[1], hi_[1])
            ax.set_zlim(lo_[2], hi_[2])
            ax.set_box_aspect(tuple(np.maximum(hi_ - lo_, 1e-6)))
            ax.view_init(elev=args.elev, azim=args.azim)
            fig.text(0.5, 0.95, f"one frame of the reconstruction · frame {frame_id}",
                     ha="center", fontsize=20, color=INK, fontweight="bold")
            fig.text(0.5, 0.915, f"{len(frusta)} cameras · {len(wf)} joints "
                     f"triangulated from {len(rays_by_f[frame_id])} rays · "
                     f"body fitted to those points",
                     ha="center", fontsize=13, color=MUTED)
            fig.savefig(path, facecolor=BG)
            plt.close(fig)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        if not args.video:
            draw(placed[0], args.out)
            print(f"wrote {args.out}")
        else:
            tmp = Path(tempfile.mkdtemp(prefix="tri3d_"))
            try:
                for i, f in enumerate(placed):
                    draw(f, tmp / f"{i:05d}.png")
                    if (i + 1) % 25 == 0:
                        logger.info(f"  {i + 1}/{len(placed)} frames")
                encode(tmp, args.out, args.fps)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    finally:
        if view != args.scene_dir:
            shutil.rmtree(view, ignore_errors=True)


def encode(png_dir: Path, mp4: Path, fps: int) -> None:
    """PNG sequence -> H.264."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe, "-y", "-framerate", str(fps), "-i", str(png_dir / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)],
                   check=True, capture_output=True)
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
