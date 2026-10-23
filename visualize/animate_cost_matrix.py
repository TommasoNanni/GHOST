"""Animate the DTW cost search as PNG frames, then an MP4.

One 16:9 slide, left to right:

  * the fixed mesh of camera A at one frame — the row being searched,
  * the cost matrix as a coarse grid, filling in as the search runs,
  * a thin bar repeating the current cost as a height,
  * the mesh of camera B at the column under test, redrawn every output frame.

The grid is deliberately coarser than the data: one cell per frame pair would be
a hairline at slide size, so the matrix is block-averaged to a readable grid and
cells light up as the sweep passes through them, with a soft leading edge so the
fill reads as a wave rather than a row of switches.

Phase 1 sweeps the row of camera A's fixed frame, opening on the column that
naive index alignment would pair with it, travelling to the true minimum, holding
through it and past, then running out.  Phase 2 holds and rings the minimum.
Phase 3 washes the rest of the matrix in as a diagonal wave from the top left.

Pacing is an explicit list of column indices — repeats are holds — so the timing
is data, and re-running gives the same movie.  Every mesh view is loaded once
into memory and drawn with set_data; nothing is re-rendered per output frame.

Inputs come from visualize/visualize_for_presentation.py:
    --dump-cost    writes the .npz read here (C plus its absolute frame axes)
    --still-range  writes the mesh PNG directory

Usage:
    pixi run python visualize/animate_cost_matrix.py \\
        --cost cost.npz --mesh-dir mesh_frames --sweep-prefix cam03 \\
        --fixed-mesh cam02_f140.png --row-frame 140 --out out --mp4 row.mp4
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

BG = "#ffffff"
INK = "#1a1a1a"
MUTED = "#8a9098"
ACCENT = "#c62828"                                # deep red, the cost hue
EMPTY = np.array([0.945, 0.937, 0.933, 1.0])     # unfilled cell, warm grey
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": BG,
})

_F_RE = re.compile(r"_f(\d+)\.png$")


# ── meshes ──────────────────────────────────────────────────────────────────
def load_meshes(mesh_dir: Path, prefix: str | None) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for p in sorted(mesh_dir.glob("*.png")):
        if prefix and not p.name.startswith(prefix):
            continue
        m = _F_RE.search(p.name)
        if m:
            out[int(m.group(1))] = plt.imread(str(p))
    if not out:
        sys.exit(f"no mesh PNGs matching prefix {prefix!r} in {mesh_dir}")
    return out


def _alpha_bbox(img: np.ndarray):
    alpha = img[:, :, 3] if img.shape[2] == 4 else np.ones(img.shape[:2])
    ys, xs = np.nonzero(alpha > 0.01)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def window_crop(images: list[np.ndarray], height_ref: float, zoom: float,
                aspect: float) -> list[np.ndarray]:
    """Crop each image so the body is exactly `win_h / zoom` pixels tall.

    The window size comes from a reference height, but each frame is cropped to
    its OWN body height and rescaled into that window, so the body is the same
    height in every frame — the person walking towards the camera no longer
    grows through the clip.  Two sets given the same zoom and aspect therefore
    match each other as well.
    """
    win_h = max(8, int(round(zoom * height_ref)))
    win_w = max(8, int(round(win_h * aspect)))
    out = []
    for img in images:
        box = _alpha_bbox(img)
        canvas = np.zeros((win_h, win_w, 4), img.dtype)
        if box is None:
            out.append(canvas)
            continue
        # the region is this frame's own body scaled by `zoom`, so resizing it
        # into the window puts every body at the same height
        body_h = max(1, box[3] - box[1])
        body_w = max(1, box[2] - box[0])
        k = max(zoom * body_h / win_h, zoom * body_w * 1.05 / win_w)
        reg_w, reg_h = max(2, int(round(win_w * k))), max(2, int(round(win_h * k)))
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        x0, y0 = cx - reg_w // 2, cy - reg_h // 2
        region = np.zeros((reg_h, reg_w, 4), img.dtype)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(img.shape[1], x0 + reg_w), min(img.shape[0], y0 + reg_h)
        patch = img[sy0:sy1, sx0:sx1]
        region[sy0 - y0:sy0 - y0 + patch.shape[0],
               sx0 - x0:sx0 - x0 + patch.shape[1]] = patch
        inter = cv2.INTER_AREA if k > 1.0 else cv2.INTER_CUBIC
        scaled = cv2.resize(region, (win_w, win_h), interpolation=inter)
        # cubic overshoots past the ends of the range; imshow would clip it and
        # warn once per frame
        out.append(np.clip(scaled, 0.0, 1.0) if scaled.dtype.kind == "f" else scaled)
    return out


def body_height(images: list[np.ndarray]) -> float:
    """Median body height of the set — the size the window is built around."""
    heights = [b[3] - b[1] for b in (_alpha_bbox(i) for i in images) if b is not None]
    return float(np.median(heights)) if heights else 1.0


# ── the coarse grid ─────────────────────────────────────────────────────────
def coarsen(C: np.ndarray, n_rows: int, n_cols: int):
    """Block-average C onto a readable grid; also the fine->coarse index maps.

    One cell per frame pair is a hairline on a slide.  Averaging keeps the shape
    of the cost surface — the valley stays a valley — while giving cells a size
    the eye can follow.
    """
    R, K = C.shape
    n_rows, n_cols = min(n_rows, R), min(n_cols, K)
    row_of = np.minimum((np.arange(R) * n_rows) // R, n_rows - 1)
    col_of = np.minimum((np.arange(K) * n_cols) // K, n_cols - 1)
    finite = np.isfinite(C)
    total = np.zeros((n_rows, n_cols))
    count = np.zeros((n_rows, n_cols))
    np.add.at(total, (row_of[:, None].repeat(K, 1), col_of[None, :].repeat(R, 0)),
              np.where(finite, C, 0.0))
    np.add.at(count, (row_of[:, None].repeat(K, 1), col_of[None, :].repeat(R, 0)),
              finite.astype(float))
    small = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return small, row_of, col_of


def grid_rgba(small: np.ndarray, alpha: np.ndarray, norm: Normalize, cmap) -> np.ndarray:
    """Cell colours with per-cell alpha: unfilled cells sit under filled ones."""
    filled = cmap(norm(np.nan_to_num(small, nan=norm.vmin)))
    filled[np.isnan(small)] = EMPTY
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    return filled * a + EMPTY[None, None] * (1.0 - a)


# ── pacing ──────────────────────────────────────────────────────────────────
def sweep_order(n_cols: int, start: int, argmin: int, slow_window: int,
                fast_step: int, tail_step: int) -> list[int]:
    """Columns to visit, in order; the stride is the speed.

    The sweep opens on `start` — the column naive index alignment would pair
    with this row — and runs toward the true minimum.  Within `slow_window` of
    it the stride is one column per output frame, which at 30 fps is the capture
    rate and reads as real motion; further out the stride opens up so the flat
    part of the row goes past quickly.  Speed is stride, never repetition: a
    repeated column freezes the pose, which reads as a stall.

    When that leg reaches the end, the untouched side of the row is swept from
    its outer end back inwards, faster still, so the row ends up complete.
    """
    def stride(n: int, fast: int) -> int:
        # slow near either landmark: the minimum, and the column the analysis is
        # centred on.  Both are places the viewer needs time to read, and the
        # return leg passes them too.
        near = min(abs(n - argmin), abs(n - start))
        return 1 if near <= slow_window else fast

    step = 1 if argmin >= start else -1
    end = n_cols - 1 if step > 0 else 0
    order: list[int] = []
    n = start
    while (n <= end if step > 0 else n >= end):
        order.append(n)
        n += step * stride(n, fast_step)
    order.append(end)

    # the untouched side, swept from its outer end back towards the start; the
    # direction is whichever way points at `start`, not the reverse of the first
    # leg — those differ, and getting it wrong leaves half the row empty
    far_end = 0 if step > 0 else n_cols - 1
    back = 1 if start > far_end else -1
    n = far_end
    while (n < start if back > 0 else n > start):
        order.append(n)
        n += back * stride(n, tail_step)
    order.append(start)
    return order


def norm_limits(values: np.ndarray) -> tuple[float, float]:
    """Robust limits: _compute_cost_matrix writes +inf where two frames share no
    confident joint, and one inf would flatten the scale."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


# ── figure ──────────────────────────────────────────────────────────────────
def build_figure(mesh_a: np.ndarray, mesh_b: np.ndarray, grid: np.ndarray,
                 dpi: int = 150):
    # 12.8 x 7.2 inches is 16:9; the dpi sets the pixel size (150 -> 1920x1080,
    # 300 -> 3840x2160).  Everything is placed in figure fractions, so the layout
    # is identical at any of them.
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    gs = fig.add_gridspec(1, 4, width_ratios=[3.0, 4.6, 0.4, 3.0],
                          left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.06)
    ax_a, ax_m, ax_bar, ax_b = (fig.add_subplot(gs[0, i]) for i in range(4))
    # the matrix and its bar keep their width but stand shorter than the poses,
    # sitting on the same baseline: shrinking about the centre lifts them off it
    for ax in (ax_m, ax_bar):
        box = ax.get_position()
        ax.set_position((box.x0, box.y0, box.width, box.height * MATRIX_HEIGHT))
    for ax in (ax_a, ax_b, ax_bar, ax_m):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    im_a = ax_a.imshow(mesh_a)
    im_b = ax_b.imshow(mesh_b)
    im_m = ax_m.imshow(grid, aspect="auto", interpolation="nearest", zorder=1)
    # crosshair: the row under test, and the column the right-hand pose is at.
    # Without it the two meshes and the matrix are three unrelated pictures.
    row_band = Rectangle((-0.5, 0), 1, 1, facecolor=ACCENT, alpha=0.10,
                         edgecolor="none", zorder=2)
    col_band = Rectangle((0, -0.5), 1, 1, facecolor=ACCENT, alpha=0.16,
                         edgecolor="none", zorder=3)
    ax_m.add_patch(row_band)
    ax_m.add_patch(col_band)
    for spine in ax_m.spines.values():
        spine.set_visible(True)
        spine.set_color("#c7ccd1")
        spine.set_linewidth(1.0)
    txt = fig.text(0.5, 0.945, "", ha="center", va="center", fontsize=34,
                   color=INK, fontweight="bold")
    fig.text(0.5, 0.905, "alignment cost", ha="center", va="center",
             fontsize=12, color=MUTED)
    cap_a = fig.text(0.13, 0.03, "", ha="center", fontsize=13, color=MUTED)
    cap_b = fig.text(0.87, 0.03, "", ha="center", fontsize=13, color=MUTED)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    # the bar carries the same red ramp as the cells, so its height and its
    # shade say the same thing the matrix says
    ramp = plt.get_cmap("Reds")(np.linspace(0.12, 1.0, 256))[:, None, :]
    ax_bar.imshow(ramp, extent=(0.1, 0.9, 0.0, 1.0), origin="lower",
                  aspect="auto", zorder=1)
    # a curtain over the unfilled part, rather than a coloured rectangle over an
    # empty one: what is left uncovered is exactly the cost
    bar = Rectangle((0.1, 0.0), 0.8, 1.0, facecolor=BG, edgecolor="none",
                    alpha=0.93, zorder=2)
    ax_bar.add_patch(bar)
    ax_bar.add_patch(Rectangle((0.1, 0.0), 0.8, 1.0, facecolor="none",
                               edgecolor="#e0d6d6", linewidth=1.0, zorder=3))
    return fig, im_a, im_b, im_m, txt, bar, row_band, col_band, cap_a, cap_b


# the matrix panel inside the animation: 4.6 of 11 width units across 96% of a
# 12.8in figure, by the 0.05..0.88 band of a 7.2in one.  The stills reuse it so a
# cell has the same shape on a slide as it does in the video.
MATRIX_HEIGHT = 2 / 3          # of the band the poses occupy
PANEL_ASPECT = ((12.8 * 0.96 * (4.6 / 11.0) * 0.94)
                / (7.2 * (0.88 - 0.05) * MATRIX_HEIGHT))


def save_matrix(path: Path, data: np.ndarray, norm: Normalize, cmap,
                aspect: float = PANEL_ASPECT) -> None:
    """Write the matrix on its own — no panels, no labels, nothing but cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9 * aspect, 9), dpi=220)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(f"wrote {path}")


def save_grid(path: Path, rgba: np.ndarray, aspect: float = PANEL_ASPECT) -> None:
    """Write a pre-coloured cell grid (already RGBA) at the animation's look."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9 * aspect, 9), dpi=220)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost", type=Path, required=True)
    ap.add_argument("--mesh-dir", type=Path, required=False, default=None,
                    help="not needed with --matrix-only")
    ap.add_argument("--fixed-mesh", type=Path, default=None)
    ap.add_argument("--fixed-dir", type=Path, default=None,
                    help="mesh PNG dir for the left camera (defaults to --mesh-dir)")
    ap.add_argument("--fixed-prefix", default=None)
    ap.add_argument("--sweep-prefix", default=None)
    ap.add_argument("--row-frame", type=int, required=True)
    ap.add_argument("--start-col", type=int, default=None,
                    help="column the sweep opens on; default = the row's own index")
    ap.add_argument("--grid-rows", type=int, default=26)
    ap.add_argument("--grid-cols", type=int, default=34)
    ap.add_argument("--slow-window", type=int, default=10,
                    help="columns either side of the minimum walked one per frame")
    ap.add_argument("--fast-step", type=int, default=4,
                    help="columns per frame away from the minimum")
    ap.add_argument("--tail-step", type=int, default=12,
                    help="columns per frame on the far side of the row")
    ap.add_argument("--ff-step", type=int, default=7,
                    help="frames advanced per output frame in the fast-forward")
    ap.add_argument("--wave", type=float, default=3.0,
                    help="output frames a cell takes to fade in once scored")
    ap.add_argument("--extra-rows", type=int, default=3)
    ap.add_argument("--zoom", type=float, default=1.30,
                    help="mesh window height as a multiple of body height")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=150,
                    help="150 = 1920x1080, 220 = 2816x1584, 300 = 3840x2160")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--mp4", type=Path, default=None)
    ap.add_argument("--matrix-only", action="store_true",
                    help="write the matrix stills and stop, no animation")
    ap.add_argument("--matrix-png", type=Path, default=None,
                    help="also save the finished matrix as a standalone image "
                         "(coarse grid, plus a _full.png at native resolution)")
    args = ap.parse_args()

    if args.mesh_dir is None and not args.matrix_only:
        ap.error("--mesh-dir is required unless --matrix-only is given")
    if args.matrix_only and args.matrix_png is None:
        ap.error("--matrix-only needs --matrix-png")

    d = np.load(args.cost, allow_pickle=False)
    C = d["C"].astype(np.float64)
    row_frames, col_frames = d["row_frames"].astype(int), d["col_frames"].astype(int)
    rows = np.nonzero(row_frames == args.row_frame)[0]
    if rows.size == 0:
        sys.exit(f"row frame {args.row_frame} not in {row_frames.min()}..{row_frames.max()}")
    m = int(rows[0])

    if args.matrix_only:
        small, row_of, col_of = coarsen(C, args.grid_rows, args.grid_cols)
        vmin, vmax = norm_limits(C)
        norm, cmap = Normalize(vmin=vmin, vmax=vmax), plt.get_cmap("Reds")
        start = int(np.clip(args.start_col if args.start_col is not None else m,
                            0, len(col_frames) - 1))
        stem = args.matrix_png.with_suffix("")
        blank = np.zeros_like(small)
        save_grid(Path(f"{stem}_empty.png"), grid_rgba(small, blank, norm, cmap))
        one = np.zeros_like(small)
        one[int(row_of[m]), int(col_of[start])] = 1.0
        save_grid(Path(f"{stem}_first.png"), grid_rgba(small, one, norm, cmap))
        save_matrix(args.matrix_png, small, norm, cmap)
        save_matrix(Path(f"{stem}_full.png"), C, norm, cmap)
        return

    sweep_raw = load_meshes(args.mesh_dir, args.sweep_prefix)
    keys = sorted(sweep_raw)
    sweep_imgs = [sweep_raw[k] for k in keys]
    # the left camera is a single frame for the sweep, but the fast-forward at
    # the end runs both cameras, so load its whole range when one is available
    fixed_dir = args.fixed_dir or args.mesh_dir
    fixed_all = {}
    if args.fixed_prefix:
        fixed_all = load_meshes(fixed_dir, args.fixed_prefix)
    if args.fixed_mesh is not None and args.row_frame not in fixed_all:
        fixed_all[args.row_frame] = plt.imread(str(args.fixed_mesh))
    if args.row_frame not in fixed_all:
        sys.exit(f"no mesh PNG for the fixed frame {args.row_frame}")
    fixed_img = fixed_all[args.row_frame]

    # same zoom and aspect for both sets -> the two bodies match in height even
    # though the cameras framed them differently
    aspect = 0.72
    sweep = dict(zip(keys, window_crop(sweep_imgs, body_height(sweep_imgs),
                                       args.zoom, aspect)))
    fixed_keys = sorted(fixed_all)
    fixed_crop = window_crop([fixed_all[k] for k in fixed_keys],
                             body_height([fixed_all[k] for k in fixed_keys]),
                             args.zoom, aspect)
    fixed = dict(zip(fixed_keys, fixed_crop))
    mesh_a = fixed[args.row_frame]

    usable = [j for j, f in enumerate(col_frames) if int(f) in sweep]
    if not usable:
        sys.exit("no column has a matching mesh PNG")
    row = np.where(np.isfinite(C[m]), C[m], np.inf)
    argmin = int(min(usable, key=lambda j: row[j]))
    start = int(np.clip(args.start_col if args.start_col is not None else m,
                        0, len(col_frames) - 1))
    if start not in set(usable):
        start = min(usable, key=lambda j: abs(j - start))

    small, row_of, col_of = coarsen(C, args.grid_rows, args.grid_cols)
    vmin, vmax = norm_limits(C)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Reds")
    alpha = np.zeros_like(small)
    visited = np.zeros(small.shape, bool)
    R, K = small.shape

    (fig, im_a, im_b, im_m, txt, bar,
     row_band, col_band, cap_a, cap_b) = build_figure(
        mesh_a, sweep[int(col_frames[start])], grid_rgba(small, alpha, norm, cmap),
        args.dpi)
    row_band.set_bounds(-0.5, int(row_of[m]) - 0.5, K, 1)
    cap_a.set_text(f"camera A · frame {args.row_frame}")

    if args.matrix_png:
        stem = args.matrix_png.with_suffix("")
        blank = np.zeros_like(small)
        save_grid(Path(f"{stem}_empty.png"), grid_rgba(small, blank, norm, cmap))
        one = np.zeros_like(small)
        one[int(row_of[m]), int(col_of[start])] = 1.0      # the cell the sweep opens on
        save_grid(Path(f"{stem}_first.png"), grid_rgba(small, one, norm, cmap))
        save_matrix(args.matrix_png, small, norm, cmap)
        save_matrix(Path(f"{stem}_full.png"), C, norm, cmap)

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.png"):
        old.unlink()
    counter = 0

    def emit() -> None:
        nonlocal counter
        fig.savefig(args.out / f"{counter:04d}.png")
        counter += 1

    def redraw_grid() -> None:
        im_m.set_data(grid_rgba(small, alpha, norm, cmap))

    def show_cost(value: float) -> None:
        txt.set_text("∞" if not np.isfinite(value) else f"{value:.2f}")
        frac = 0.0 if not np.isfinite(value) else \
            float(np.clip((value - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0))
        bar.set_y(frac)                 # curtain starts where the fill ends
        bar.set_height(1.0 - frac)

    # ── phase 1: sweep the row, the fill trailing the pose as a wave ────────
    rc = int(row_of[m])
    rate = float(np.clip(1.0 / max(args.wave, 1e-6), 0.05, 1.0))
    have = set(usable)
    last = start
    filled_lo = filled_hi = int(col_of[start])
    for n in sweep_order(len(col_frames), start, argmin, args.slow_window,
                         args.fast_step, args.tail_step):
        j = n if n in have else last
        last = j
        im_b.set_data(sweep[int(col_frames[j])])
        cap_b.set_text(f"camera B · frame {int(col_frames[j])}")
        col_band.set_bounds(int(col_of[j]) - 0.5, -0.5, 1, R)
        # cells the stride jumped over were scored too, so fill the whole span
        # since the last position: otherwise a fast leg leaves a dotted line
        c = int(col_of[j])
        if c > filled_hi:
            visited[rc, filled_hi:c + 1] = True
            filled_hi = c
        elif c < filled_lo:
            visited[rc, c:filled_lo + 1] = True
            filled_lo = c
        else:
            visited[rc, c] = True
        alpha[visited] += (1.0 - alpha[visited]) * rate
        redraw_grid()
        show_cost(C[m, j])
        emit()

    # ── phase 3: the rest washes in diagonally from the top left ───────────
    ramp = 4.0
    ii, jj = np.meshgrid(np.arange(R), np.arange(K), indexing="ij")
    front = ii + jj
    # the poses drop away and the matrix finishes on its own: with every pair
    # being scored at once there is no single frame pair left to show
    im_a.set_data(np.zeros_like(mesh_a))
    im_b.set_data(np.zeros_like(sweep[int(col_frames[start])]))
    cap_a.set_text("")
    cap_b.set_text("")
    txt.set_text("")
    bar.set_y(0.0)
    bar.set_height(1.0)
    row_band.set_visible(False)
    col_band.set_visible(False)
    for t in np.linspace(0, front.max() + ramp, int(2.5 * args.fps)):
        alpha[:] = np.maximum(alpha, np.clip((t - front) / ramp, 0.0, 1.0))
        redraw_grid()
        emit()
    for _ in range(args.fps):
        emit()

    if args.matrix_png:
        save_matrix(args.matrix_png, small, norm, cmap)
        save_matrix(args.matrix_png.with_name(args.matrix_png.stem + "_full.png"),
                    C, norm, cmap)

    print(f"{counter} frames -> {args.out}/%04d.png")
    if args.mp4:
        encode(args.out, args.mp4, args.fps)


def encode(png_dir: Path, mp4: Path, fps: int) -> None:
    """Assemble with ffmpeg — the static binary from imageio-ffmpeg when the
    environment has none of its own, as on this cluster."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print(f"no ffmpeg; run:\n  ffmpeg -framerate {fps} -i {png_dir}/%04d.png "
                  f"-c:v libx264 -pix_fmt yuv420p {mp4}")
            return
    subprocess.run([exe, "-y", "-framerate", str(fps), "-i", str(png_dir / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
                   check=True, capture_output=True)
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
