"""Animate the diagonal scan that turns a cost matrix into one offset.

The row animation shows how a single pair of frames is scored.  This shows what
the synchroniser does with the whole matrix: for every candidate offset k it
reads the diagonal cost[i, i+k], takes the median over the frames the two clips
share at that offset, and keeps the k where that median is lowest.  The diagonal
is the hypothesis "camera B is k frames ahead of camera A", so the scan is the
search over offsets, and its minimum is the answer the pair contributes.

Every diagonal stays on screen once scanned, so the picture builds into the full
family of hypotheses rather than a single sliding line, with the winner marked at
the end.  The curve beside the matrix is the same numbers as a function of k.

Aggregation mirrors Synchronizer.estimate_couple_offset: the k range is
[-2*max_shift, 2*max_shift] clipped to the matrix, diagonals shorter than
min_overlap are skipped, and +inf entries (frames sharing no confident joint)
are dropped before the median rather than poisoning it.

Usage:
    pixi run python visualize/animate_offset_scan.py \\
        --cost cost.npz --max-shift 75 --out out --mp4 scan.mp4
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
from matplotlib.colors import Normalize

BG = "#ffffff"
INK = "#1a1a1a"
MUTED = "#8a9098"
ACCENT = "#c62828"
SCANNED = np.array([0.06, 0.62, 0.84])       # #0f9ed5, the "already tried" tint
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": BG,
})


def coarsen_square(C: np.ndarray, cell: int):
    """Block-average C into cells that are `cell` frames on BOTH sides.

    The row animation's grid has a different number of frames per row than per
    column, which is fine for reading values off it but wrong here: a diagonal
    of that grid is not a constant-offset line.  With square cells, one step
    along a cell diagonal is one frame of offset times `cell`, so the scan over
    cell diagonals is the scan over offsets, just at a coarser grain.
    """
    if cell <= 1:
        return C
    n, m = C.shape
    rows, cols = int(np.ceil(n / cell)), int(np.ceil(m / cell))
    out = np.full((rows, cols), np.nan)
    for i in range(rows):
        for j in range(cols):
            block = C[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            finite = block[np.isfinite(block)]
            if finite.size:
                out[i, j] = finite.mean()
    return out


def diagonal_scores(C: np.ndarray, max_shift: int, min_overlap: int):
    """{k: median cost} over the candidate offsets, the synchroniser's own rule."""
    n, m = C.shape
    k_lo = max(-(n - 1), -2 * max_shift)
    k_hi = min(m, 2 * max_shift + 1)
    out: dict[int, float] = {}
    for k in range(k_lo, k_hi):
        i0, i1 = max(0, -k), min(n, m - k)
        if i1 - i0 < min_overlap:
            continue
        vals = C[np.arange(i0, i1), np.arange(i0, i1) + k]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[k] = float(np.median(vals))
    if not out:
        sys.exit("no candidate offset has enough overlap")
    return out


def scan_order(ks: list[int], best: int, slow_window: int, fast_step: int) -> list[int]:
    """Offsets in scanning order: quick over the flat range, one at a time near
    the minimum, which is the only part worth watching."""
    order, n = [], len(ks)
    i = 0
    while i < n:
        order.append(ks[i])
        i += 1 if abs(ks[i] - best) <= slow_window else fast_step
    if order[-1] != ks[-1]:
        order.append(ks[-1])
    return order


def norm_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost", type=Path, required=True)
    ap.add_argument("--max-shift", type=int, default=75)
    ap.add_argument("--min-overlap", type=int, default=100)
    ap.add_argument("--cell", type=int, default=1,
                    help="frames per grid cell; the scan runs on this coarse grid "
                         "so it is readable on a slide")
    ap.add_argument("--slow-window", type=int, default=12,
                    help="offsets either side of the best one scanned singly")
    ap.add_argument("--fast-step", type=int, default=3)
    ap.add_argument("--hold", type=int, default=1,
                    help="output frames per offset (the grid has few of them)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=150,
                    help="150 = 1920x1080, 300 = 3840x2160")
    ap.add_argument("--out", type=Path, default=Path("out_scan"))
    ap.add_argument("--mp4", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.cost, allow_pickle=False)
    C_full = d["C"].astype(np.float64)
    C = coarsen_square(C_full, args.cell)
    n, m = C.shape
    # every quantity below is in cells; only the labels convert back to frames
    scores = diagonal_scores(C, max(1, args.max_shift // args.cell),
                             max(1, args.min_overlap // args.cell))
    ks = sorted(scores)
    best = min(ks, key=lambda k: scores[k])
    vmin, vmax = norm_limits(C)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Reds")

    # ── figure: the matrix, and the same numbers as a curve over k ──────────
    fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.04, right=0.97,
                          top=0.86, bottom=0.12, wspace=0.16)
    ax_m, ax_k = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    for spine in ax_m.spines.values():
        spine.set_color("#c7ccd1")

    base = cmap(norm(np.nan_to_num(C, nan=vmin)))          # (n, m, 4)
    layer = np.zeros((n, m))                                # how scanned each cell is
    im = ax_m.imshow(base, aspect="auto", interpolation="nearest")

    sc_x = np.array(ks, dtype=float) * args.cell
    sc_y = np.array([scores[k] for k in ks])
    ax_k.set_xlim(sc_x.min(), sc_x.max())
    pad = 0.08 * (sc_y.max() - sc_y.min() + 1e-9)
    ax_k.set_ylim(sc_y.min() - pad, sc_y.max() + pad)
    ax_k.set_xlabel("offset k  (frames)", color=MUTED, fontsize=13)
    ax_k.set_ylabel("median cost along the diagonal", color=MUTED, fontsize=13)
    ax_k.tick_params(colors=MUTED, labelsize=11)
    for side in ("top", "right"):
        ax_k.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_k.spines[side].set_color("#c7ccd1")
    curve, = ax_k.plot([], [], color=ACCENT, linewidth=2.0)
    dot, = ax_k.plot([], [], marker="o", markersize=8, color=ACCENT, linestyle="none")
    win, = ax_k.plot([], [], marker="o", markersize=13, markerfacecolor="none",
                     markeredgecolor=INK, markeredgewidth=2.4, linestyle="none")
    win_txt = ax_k.text(0, 0, "", color=INK, fontsize=14, fontweight="bold",
                        ha="center", va="bottom")

    head = fig.text(0.5, 0.945, "", ha="center", va="center", fontsize=30,
                    color=INK, fontweight="bold")
    fig.text(0.5, 0.90, "scanning offsets", ha="center", va="center",
             fontsize=12, color=MUTED)

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.png"):
        old.unlink()
    counter = 0

    def emit() -> None:
        nonlocal counter
        fig.savefig(args.out / f"{counter:04d}.png")
        counter += 1

    def paint() -> None:
        a = np.clip(layer, 0.0, 1.0)[..., None]
        im.set_data(base * (1 - 0.55 * a) + np.r_[SCANNED, 1.0][None, None] * 0.55 * a)

    seen_x, seen_y = [], []
    last_k = None
    for k in scan_order(ks, best, args.slow_window, args.fast_step):
        # light every diagonal between the last position and this one: the stride
        # skips some, and a band with gaps in it looks like a rendering fault
        span = [k] if last_k is None else range(min(last_k, k), max(last_k, k) + 1)
        for kk in span:
            i0, i1 = max(0, -kk), min(n, m - kk)
            idx = np.arange(i0, i1)
            layer[idx, idx + kk] = 1.0     # scanned diagonals stay on screen
        last_k = k
        paint()
        seen_x.append(k * args.cell)
        seen_y.append(scores[k])
        order = np.argsort(seen_x)
        curve.set_data(np.array(seen_x)[order], np.array(seen_y)[order])
        dot.set_data([k * args.cell], [scores[k]])
        head.set_text(f"k = {k * args.cell:+d}      {scores[k]:.3f}")
        for _ in range(max(1, args.hold)):
            emit()

    # ── the winner ─────────────────────────────────────────────────────────
    idx = np.arange(max(0, -best), min(n, m - best))
    highlight = np.zeros((n, m))
    highlight[idx, idx + best] = 1.0
    dot.set_data([], [])
    win.set_data([best * args.cell], [scores[best]])
    win_txt.set_position((best * args.cell, scores[best] + pad * 0.6))
    win_txt.set_text(f"k = {best * args.cell:+d}")
    head.set_text(f"offset  k = {best * args.cell:+d}")
    for t in np.linspace(0, 1, int(0.6 * args.fps)):
        a = np.clip(layer, 0, 1)[..., None]
        img = base * (1 - 0.55 * a) + np.r_[SCANNED, 1.0][None, None] * 0.55 * a
        hl = highlight[..., None] * t
        im.set_data(img * (1 - hl) + np.r_[0.776, 0.157, 0.157, 1.0][None, None] * hl)
        emit()
    for _ in range(int(1.5 * args.fps)):
        emit()

    print(f"{counter} frames -> {args.out}/%04d.png   "
          f"(best k = {best * args.cell:+d} frames, {best:+d} cells)")
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
