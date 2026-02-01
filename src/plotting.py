import os
from typing import Optional
import typing
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================
# Global matplotlib styling
# =============================
plt.style.use("default")
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.figsize"] = (10, 0.8)
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titlesize"] = 9
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7

# =============================
# Color palette (paper-friendly)
# =============================
COLOR_TRUE = "#156dac"      # Time series (blue)
COLOR_SCORE = "#2ca02c"     # Anomaly score (green)
COLOR_GT_SHADE = "#1609a1"  # True anomaly (purple-ish)
COLOR_PRED_SHADE = "#c55409"  # Predicted anomaly (deeper orange)


def smooth(y: np.ndarray, box_pts: int = 1) -> np.ndarray:
    if box_pts is None or box_pts <= 1:
        return y
    y = np.asarray(y)
    box = np.ones(box_pts, dtype=float) / float(box_pts)
    return np.convolve(y, box, mode="same")


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _shade_runs(ax, mask: np.ndarray, *, min_run: int, color: str, alpha: float = 0.35) -> None:
    if mask is None:
        return
    m = np.asarray(mask).astype(bool).ravel()
    if m.size == 0:
        return
    diff = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) >= int(min_run):
            ax.axvspan(s - 0.5, e - 0.5, color=color, alpha=alpha)


def plotter(
    name: str,
    y_true,
    y_pred,
    ascore,
    labels,
    *,
    smooth_k: int = 1,
    clip_score_pct: float = 99.5,
    add_legend: bool = True,
    pred_labels: Optional[np.ndarray] = None,
    pred_min_run: int = 3,
    legend_on_separate_page: bool = True,
    dims_to_plot: Optional[typing.Sequence[int]] = None,
) -> None:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    ascore = _to_numpy(ascore)
    labels = _to_numpy(labels)

    if isinstance(name, str) and ("WGANGP" in name) and isinstance(y_true, np.ndarray):
        y_true = np.roll(y_true, shift=1, axis=0)

    if ascore.ndim == 2 and labels.ndim == 2 and ascore.shape[1] != labels.shape[1]:
        if ascore.shape[1] % labels.shape[1] == 0:
            feature_dim = labels.shape[1]
            window_size = ascore.shape[1] // feature_dim
            ascore = ascore.reshape(-1, window_size, feature_dim).mean(axis=1)
    elif ascore.ndim == 1:
        ascore = np.tile(ascore[:, None], (1, labels.shape[1]))

    T = int(min(y_true.shape[0], labels.shape[0], ascore.shape[0]))
    y_true = y_true[:T]
    labels = labels[:T]
    ascore = ascore[:T]

    F = int(min(y_true.shape[1], labels.shape[1], ascore.shape[1]))

    # Resolve which dimensions to plot
    if dims_to_plot is None:
        dims_index = list(range(F))
    else:
        # keep only valid, unique, in-range dims; preserve given order
        seen = set()
        dims_index = []
        for d in dims_to_plot:
            try:
                di = int(d)
            except Exception:
                continue
            if 0 <= di < F and di not in seen:
                dims_index.append(di)
                seen.add(di)
        if len(dims_index) == 0:
            dims_index = list(range(F)) # fallback to all if none valid

    if pred_labels is not None:
        pred_labels = np.asarray(pred_labels)
        if pred_labels.ndim == 1:
            pred_labels = pred_labels.reshape(-1, 1)
        if pred_labels.shape[0] < T:
            pad = np.zeros((T - pred_labels.shape[0], pred_labels.shape[1]))
            pred_labels = np.vstack([pred_labels, pad])
        elif pred_labels.shape[0] > T:
            pred_labels = pred_labels[:T]
        if pred_labels.shape[1] == 1 and F > 1:
            pred_labels = np.tile(pred_labels, (1, F))
        if pred_labels.shape[1] > F:
            pred_labels = pred_labels[:, :F]

    out_dir = os.path.join("plots", name)
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "output.pdf")
    pdf = PdfPages(pdf_path)

    if add_legend and legend_on_separate_page:
        fig_leg = plt.figure(figsize=(10, 0.8))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis("off")
        legend_elements = [
            Line2D([0], [0], color=COLOR_TRUE, lw=1.0, label="Time series"),
            Line2D([0], [0], color=COLOR_SCORE, lw=1.0, label="Anomaly Scores"),
            Patch(facecolor=COLOR_GT_SHADE, alpha=0.35, label="True Anomaly"),
            Patch(facecolor=COLOR_PRED_SHADE, alpha=0.35, label="Predicted Anomaly"),
        ]
        ax_leg.legend(
            handles=legend_elements,
            loc="center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 0.5),
        )
        legend_png = os.path.join(out_dir, "legend.png")
        fig_leg.savefig(legend_png, dpi=1000, bbox_inches="tight")
        plt.close(fig_leg)

    x = np.arange(T)
    for dim in dims_index:
        y_t = y_true[:, dim]
        l = labels[:, dim]
        a_s = ascore[:, dim]
        p = pred_labels[:, dim] if pred_labels is not None else None

        finite_vals = a_s[np.isfinite(a_s)]
        if finite_vals.size > 0:
            hi = np.percentile(finite_vals, clip_score_pct)
            lo = np.percentile(finite_vals, 100 - clip_score_pct)
            a_s = np.clip(a_s, a_min=lo, a_max=hi)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={"left": 0.12, "right": 0.98, "hspace": 0.12}
        )

        ax1.set_title(f"Dimension = {dim}")
        ax1.plot(smooth(y_t, smooth_k), linewidth=0.8, color=COLOR_TRUE, label="Time series")
        ax1.set_ylabel("Value", rotation=0, labelpad=8)
        # Keep the same x-position for both y-labels so they align perfectly
        ax1.yaxis.set_label_coords(-0.08, 0.5)

        ax3 = ax1.twinx()
        ax3.set_yticks([])
        ax3.plot(l, "--", linewidth=0.3, alpha=0.4, color=COLOR_GT_SHADE)
        ax3.fill_between(x, l, color=COLOR_GT_SHADE, alpha=0.35)

        ax2.plot(smooth(a_s, smooth_k), linewidth=0.8, color=COLOR_SCORE, label="Anomaly Scores")
        ax2.set_xlabel("Timestamp")
        ax2.set_ylabel("Anomaly Score", rotation=0, labelpad=8)
        ax2.yaxis.set_label_coords(-0.08, 0.5)

        if p is not None:
            ax4 = ax2.twinx()
            ax4.set_yticks([])
            _shade_runs(ax4, p, min_run=pred_min_run, color=COLOR_PRED_SHADE, alpha=0.35)

        if add_legend and not legend_on_separate_page and dim == 0:
            legend_elements = [
                Line2D([0], [0], color=COLOR_TRUE, lw=1.0, label="Time series"),
                Line2D([0], [0], color=COLOR_SCORE, lw=1.0, label="Anomaly Scores"),
                Patch(facecolor=COLOR_GT_SHADE, alpha=0.35, label="True Anomaly"),
                Patch(facecolor=COLOR_PRED_SHADE, alpha=0.35, label="Predicted Anomaly"),
            ]
            ax1.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.28),
                ncol=2,
                frameon=False,
            )

        png_path = os.path.join(out_dir, f"dim_{dim:02d}.png")
        fig.savefig(png_path, dpi=1000, bbox_inches="tight")
        plt.close(fig)

    #pdf.close()
    print(f"Saved PNGs to folder: {out_dir}")
def plotter_grid_3x2(
    name: str,
    y_true,
    y_pred,
    ascore,
    labels,
    *,
    smooth_k: int = 1,
    clip_score_pct: float = 99.5,
    add_legend: bool = True,
    pred_labels: Optional[np.ndarray] = None,
    pred_min_run: int = 3,
    legend_on_separate_page: bool = True,
    dims_to_plot: Optional[typing.Sequence[int]] = (0, 1, 5, 6, 17, 18),
) -> None:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    ascore = _to_numpy(ascore)
    labels = _to_numpy(labels)
    if isinstance(name, str) and ("WGANGP" in name) and isinstance(y_true, np.ndarray):
        y_true = np.roll(y_true, shift=1, axis=0)

    if ascore.ndim == 2 and labels.ndim == 2 and ascore.shape[1] != labels.shape[1]:
        if ascore.shape[1] % labels.shape[1] == 0:
            feature_dim = labels.shape[1]
            window_size = ascore.shape[1] // feature_dim
            ascore = ascore.reshape(-1, window_size, feature_dim).mean(axis=1)
    elif ascore.ndim == 1:
        ascore = np.tile(ascore[:, None], (1, labels.shape[1]))

    T = int(min(y_true.shape[0], labels.shape[0], ascore.shape[0]))
    y_true = y_true[:T]
    labels = labels[:T]
    ascore = ascore[:T]

    F = int(min(y_true.shape[1], labels.shape[1], ascore.shape[1]))

    if dims_to_plot is None:
        dims_index = list(range(F))[:6]
    else:
        seen = set()
        dims_index = []
        for d in dims_to_plot:
            try:
                di = int(d)
            except Exception:
                continue
            if 0 <= di < F and di not in seen:
                dims_index.append(di)
                seen.add(di)
        dims_index = dims_index[:6]

    if pred_labels is not None:
        pred_labels = np.asarray(pred_labels)
        if pred_labels.ndim == 1:
            pred_labels = pred_labels.reshape(-1, 1)
        if pred_labels.shape[0] < T:
            pad = np.zeros((T - pred_labels.shape[0], pred_labels.shape[1]))
            pred_labels = np.vstack([pred_labels, pad])
        elif pred_labels.shape[0] > T:
            pred_labels = pred_labels[:T]
        if pred_labels.shape[1] == 1 and F > 1:
            pred_labels = np.tile(pred_labels, (1, F))
        if pred_labels.shape[1] > F:
            pred_labels = pred_labels[:, :F]

    out_dir = os.path.join("plots", f"{name}_grid")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "output_3x2.pdf")
    pdf = PdfPages(pdf_path)

    if add_legend and legend_on_separate_page:
        fig_leg = plt.figure(figsize=(6.5, 1.3))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis("off")
        legend_elements = [
            Line2D([0], [0], color=COLOR_TRUE,  lw=1.0, label="Time series"),
            Line2D([0], [0], color=COLOR_SCORE, lw=1.0, label="Anomaly Scores"),
            Patch(facecolor=COLOR_GT_SHADE,   alpha=0.35, label="True Anomaly"),
            Patch(facecolor=COLOR_PRED_SHADE, alpha=0.35, label="Predicted Anomaly"),
        ]
        ax_leg.legend(handles=legend_elements, loc="center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.5))
        pdf.savefig(fig_leg)
        plt.close(fig_leg)

    fig, axes = plt.subplots(
        2, 3, figsize=(8.2, 4.2),
        gridspec_kw={"left": 0.08, "right": 0.98, "wspace": 0.22, "hspace": 0.28}
    )
    axes = axes.ravel()
    x = np.arange(T)

    for k, dim in enumerate(dims_index):
        ax = axes[k]
        y_t = y_true[:, dim]
        l   = labels[:, dim]
        a_s = ascore[:, dim]
        p   = pred_labels[:, dim] if pred_labels is not None else None

        finite_vals = a_s[np.isfinite(a_s)]
        if finite_vals.size > 0:
            hi = np.percentile(finite_vals, clip_score_pct)
            lo = np.percentile(finite_vals, 100 - clip_score_pct)
            a_s = np.clip(a_s, a_min=lo, a_max=hi)

        ax.plot(smooth(y_t, smooth_k), color=COLOR_TRUE, lw=0.8)
        ax.set_title(f"Dim {dim}", fontsize=9)
        ax.fill_between(x, l, color=COLOR_GT_SHADE, alpha=0.25)
        axr = ax.twinx()
        axr.plot(smooth(a_s, smooth_k), color=COLOR_SCORE, lw=0.8)
        if p is not None:
            _shade_runs(ax, p, min_run=pred_min_run, color=COLOR_PRED_SHADE, alpha=0.35)
        r, c = divmod(k, 3)
        if c != 0:
            ax.set_yticklabels([])
        if r != 1:
            ax.set_xticklabels([])
    for k in range(len(dims_index), 6):
        fig.delaxes(axes[k])

    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
    print(f"[3x2] Saved plots to: {pdf_path}")