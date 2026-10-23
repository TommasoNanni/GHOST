"""
Paper bar figure for the synchronization table (tab:syncres).

Eight panels, 2 rows x 4 columns:

    rows    = datasets            RICH, EgoHumans
    columns = metrics            MAE, <= 1 fr, AUC@100 ms, Coverage

Each panel puts the four ``delta_max`` settings (30, 45, 60, 75) on the x axis and
draws GHOST in blue against VisualSync in red -- as grouped vertical bars
(``--style bar``), horizontal bars with delta_max down the y axis
(``--style hbar``), or two lines with markers (``--style line``, the default).

The numbers are *not* recomputed here.  They are transcribed once from the LaTeX
table in the paper and kept in ``RESULTS`` below, so this file is a plotting
script and nothing else -- if the table changes, edit ``RESULTS``.

Y-axis scaling
--------------
Most of the metrics are bounded percentages or an AUC in [0, 1] and read fine on a
linear axis.  MAE is not: GHOST spans 0.08-0.63 frames while VisualSync spans
2.12-6.29, a ~80x range in which the GHOST bars would be invisible strips at the
baseline.  Panels therefore switch to a log y axis automatically when the ratio of
largest to smallest positive value exceeds ``--log-ratio`` (default 20).  Force the
decision either way with ``--log-metrics`` / ``--linear-metrics``.  In ``hbar`` the
metric lives on the x axis, so the log switch applies there instead.

Usage
-----
    pixi run python -m visualize.paper_sync_metrics                  # lines
    pixi run python -m visualize.paper_sync_metrics --style bar       # grouped bars
    pixi run python -m visualize.paper_sync_metrics --style hbar      # horizontal bars
    pixi run python -m visualize.paper_sync_metrics --linear-metrics mae
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _apply_typography(usetex: bool, base: float) -> None:
    """Computer Modern everywhere, so the figure matches the paper body text.

    ``usetex`` hands every string to a real LaTeX install; there is none on the
    CSCS nodes, so the default path picks CM-shaped TTFs already on the system.

    The conda matplotlib in this env ships **only** DejaVuSans.ttf -- no cm*, no
    STIX, no serif -- so ``mathtext.fontset="cm"`` silently substitutes per glyph
    and renders \\delta as +-, \\downarrow as # and \\leq as a bullet.  The fix is a
    ``custom`` fontset pointed at fonts fontconfig actually has: MathJax_Main /
    MathJax_Math are the TeX Computer Modern faces MathJax ships, with Liberation
    Serif and DejaVu Serif as the fallbacks if MathJax is absent elsewhere.
    """
    from matplotlib import font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    if "MathJax_Main" in have:
        rm, it, bf = "MathJax_Main", "MathJax_Math", "MathJax_Main:bold"
    elif "Liberation Serif" in have:
        rm, it, bf = "Liberation Serif", "Liberation Serif:italic", "Liberation Serif:bold"
    else:
        rm, it, bf = "DejaVu Serif", "DejaVu Serif:italic", "DejaVu Serif:bold"

    matplotlib.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": [rm.split(":")[0], "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": rm,
        "mathtext.it": it,
        "mathtext.bf": bf,
        "mathtext.fallback": "None",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "axes.labelsize": base,
        "axes.titlesize": base + 1,
        "legend.fontsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
    })
    if usetex:
        matplotlib.rcParams["text.latex.preamble"] = r"\\usepackage{amsmath}\\usepackage{amssymb}"


# --------------------------------------------------------------------------- #
# The table.  One entry per (dataset, method, delta_max).                       #
# Percentages are stored as percentages, AUC as a fraction, MAE in frames.      #
# --------------------------------------------------------------------------- #

DELTAS = [30, 45, 60, 75]
METHODS = ["GHOST", "VisualSync"]

RESULTS = {
    "RICH": {
        "VisualSync": {
            #        MAE   <=1fr    AUC@100  coverage
            30: dict(mae=2.12, le1=76.9, auc100=0.666, cov=63.9),
            45: dict(mae=3.03, le1=74.0, auc100=0.651, cov=58.9),
            60: dict(mae=2.16, le1=69.7, auc100=0.618, cov=50.6),
            75: dict(mae=3.39, le1=71.2, auc100=0.613, cov=48.3),
        },
        "GHOST": {
            30: dict(mae=0.08, le1=100.0, auc100=0.973, cov=100.0),
            45: dict(mae=0.10, le1=99.5, auc100=0.967, cov=99.4),
            60: dict(mae=0.37, le1=99.3, auc100=0.963, cov=95.0),
            75: dict(mae=0.63, le1=94.9, auc100=0.910, cov=87.8),
        },
    },
    "EgoHumans": {
        "VisualSync": {
            30: dict(mae=3.37, le1=66.5, auc100=0.607, cov=87.9),
            45: dict(mae=3.74, le1=69.9, auc100=0.634, cov=80.0),
            60: dict(mae=4.66, le1=67.8, auc100=0.614, cov=76.4),
            75: dict(mae=6.29, le1=66.1, auc100=0.598, cov=70.0),
        },
        "GHOST": {
            30: dict(mae=0.24, le1=97.1, auc100=0.882, cov=100.0),
            45: dict(mae=0.25, le1=97.1, auc100=0.877, cov=100.0),
            60: dict(mae=0.25, le1=96.4, auc100=0.875, cov=100.0),
            75: dict(mae=0.26, le1=97.0, auc100=0.871, cov=100.0),
        },
    },
}

DATASETS = ["RICH", "EgoHumans"]

# key, title template ({pct} -> a usetex-safe percent sign), formatter,
# upper bound for a linear axis (None = free)
METRICS = [
    ("mae", r"$\mathrm{{MAE}}$ (frames) $\downarrow$", "{:.2f}", None),
    ("le1", r"$\leq 1$ frame ({pct}) $\uparrow$", "{:.1f}", 100.0),
    ("auc100", r"$\mathrm{{AUC}}@100$ ms $\uparrow$", "{:.3f}", 1.0),
    ("cov", r"Coverage ({pct})", "{:.1f}", 100.0),
]

# muted paper palette: light fill, saturated edge of the same hue
FILL = {"GHOST": "#A8C4E0", "VisualSync": "#E5B7B2"}
EDGE = {"GHOST": "#3B6FB6", "VisualSync": "#A85D57"}
MARKERS = {"GHOST": "o", "VisualSync": "s"}
LINESTYLES = {"GHOST": "-", "VisualSync": "--"}
LABELS = {"GHOST": r"$\mathrm{GHOST}$ (ours)", "VisualSync": r"$\mathrm{VisualSync}$"}


def _use_log(values: list[float], ratio_threshold: float) -> bool:
    """A panel goes log when its bars span more than ``ratio_threshold``x."""
    positive = [v for v in values if v > 0]
    if len(positive) < 2:
        return False
    return max(positive) / min(positive) > ratio_threshold


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=("line", "bar", "hbar"), default="line",
                    help="vertical grouped bars, horizontal bars, or lines with markers")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: figures/sync_metrics{_line,_hbar,}.png, per --style")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--log-ratio", type=float, default=20.0,
                    help="max/min bar ratio above which a panel switches to a log y axis")
    ap.add_argument("--log-metrics", nargs="*", default=[],
                    help="metric keys forced to a log y axis (mae le1 auc100 cov)")
    ap.add_argument("--linear-metrics", nargs="*", default=[],
                    help="metric keys forced to a linear y axis")
    ap.add_argument("--fontsize", type=float, default=12.0,
                    help="base text size; ticks are -1, titles +1, bar values -3.5")
    ap.add_argument("--usetex", action="store_true",
                    help="render text through a real LaTeX install (needs pdflatex; absent on CSCS nodes)")
    ap.add_argument("--no-values", action="store_true", help="do not print the value at each point/bar")
    ap.add_argument("--figsize", type=float, nargs=2, default=None,
                    help="default: 14x6 (line/bar), 14x6.5 (hbar)")
    args = ap.parse_args()
    if args.out is None:
        stem = {"line": "sync_metrics_line", "hbar": "sync_metrics_hbar", "bar": "sync_metrics"}[args.style]
        args.out = _REPO_ROOT / "figures" / f"{stem}.png"

    _apply_typography(args.usetex, args.fontsize)
    val_fs = args.fontsize - 3.5           # value labels sit between narrow bars
    pct = r"\%" if args.usetex else "%"

    figsize = tuple(args.figsize) if args.figsize else ((15.0, 7.0) if args.style == "hbar" else (15.0, 6.4))
    fig, axes = plt.subplots(len(DATASETS), len(METRICS), figsize=figsize)
    axes = np.atleast_2d(axes)

    # bars sit on category slots, lines on the true delta_max values
    x = np.asarray(DELTAS, dtype=float) if args.style == "line" else np.arange(len(DELTAS), dtype=float)
    width = 0.38

    for r, dataset in enumerate(DATASETS):
        for c, (key, title_tpl, fmt, upper) in enumerate(METRICS):
            title = title_tpl.format(pct=pct)
            ax = axes[r, c]

            series = {m: [RESULTS[dataset][m][d][key] for d in DELTAS] for m in METHODS}
            flat = [v for vals in series.values() for v in vals]

            if key in args.log_metrics:
                log_y = True
            elif key in args.linear_metrics:
                log_y = False
            else:
                log_y = _use_log(flat, args.log_ratio)

            for i, method in enumerate(METHODS):
                label = LABELS[method] if (r == 0 and c == 0) else None
                if args.style in ("bar", "hbar"):
                    offset = (i - (len(METHODS) - 1) / 2) * width
                    # hbar mirrors bar: METHODS[0] on top of each delta_max group,
                    # which with the inverted y axis means the sign flips
                    draw = ax.bar if args.style == "bar" else ax.barh
                    pos = x + offset if args.style == "bar" else x - offset
                    bars = draw(pos, series[method], width,
                                color=FILL[method], label=label,
                                edgecolor=EDGE[method], linewidth=1.0, zorder=3)
                    if not args.no_values:
                        # horizontal in every panel, log axis included
                        ax.bar_label(bars, labels=[fmt.format(v) for v in series[method]],
                                     padding=2, fontsize=val_fs, rotation=0, color="black")
                else:
                    ax.plot(x, series[method], color=EDGE[method], label=label,
                            marker=MARKERS[method], markersize=5.5, linewidth=1.8,
                            linestyle=LINESTYLES[method], markerfacecolor=FILL[method],
                            markeredgecolor=EDGE[method], markeredgewidth=1.0, zorder=3)
                    if not args.no_values:
                        # GHOST above its line, VisualSync below, so the two never collide
                        va, dy = ("bottom", 6) if method == "GHOST" else ("top", -6)
                        for xi, v in zip(x, series[method]):
                            ax.annotate(fmt.format(v), (xi, v), textcoords="offset points",
                                        xytext=(0, dy), ha="center", va=va, fontsize=val_fs,
                                        color=EDGE[method])

            # in hbar the metric is on x and delta_max on y; everything else mirrors
            metric_scale = ax.set_xscale if args.style == "hbar" else ax.set_yscale
            metric_lim = ax.set_xlim if args.style == "hbar" else ax.set_ylim

            if log_y:
                metric_scale("log")
                lo = min(v for v in flat if v > 0)
                hi = max(flat)
                metric_lim(lo / (5.0 if args.style == "line" else 3.0),
                           hi * (2.5 if args.style == "hbar" else 4.0))
            else:
                hi = upper if upper is not None else max(flat)
                if args.style == "line":
                    # tight around the data, plus room for the value labels above/below
                    lo_lin = min(flat)
                    pad = 0.14 * max(hi - lo_lin, 1e-6)
                    metric_lim(max(0.0, lo_lin - pad), hi + pad)
                else:
                    metric_lim(0.0, hi * (1.13 if args.style == "hbar" else 1.18))

            if args.style == "hbar":
                ax.set_yticks(x)
                ax.set_yticklabels([str(d) for d in DELTAS])
                ax.invert_yaxis()          # delta_max = 30 at the top
                ax.set_ylabel(r"$\delta_{\max}$ (frames)", fontsize=args.fontsize)
                ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([str(d) for d in DELTAS])
                if args.style == "line":
                    ax.set_xlim(x[0] - 0.35 * (x[1] - x[0]), x[-1] + 0.35 * (x[1] - x[0]))
                ax.set_xlabel(r"$\delta_{\max}$ (frames)", fontsize=args.fontsize)
                ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=args.fontsize - 1)

            if r == 0:
                ax.set_title(title, fontsize=args.fontsize + 1)
            if c == 0:
                if args.style == "hbar":
                    ax.annotate(rf"$\mathbf{{{dataset}}}$", xy=(0, 0.5), xycoords="axes fraction",
                                xytext=(-42, 0), textcoords="offset points",
                                rotation=90, ha="center", va="center", fontsize=args.fontsize + 2)
                else:
                    ax.set_ylabel(rf"$\mathbf{{{dataset}}}$", fontsize=args.fontsize + 2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=args.fontsize + 1, bbox_to_anchor=(0.5, 1.005))

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
