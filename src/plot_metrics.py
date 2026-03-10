#!/usr/bin/env python3
"""
Plot per-image scatter plots comparing mean metrics vs ChatGPT mean metrics,
plus a scatter for pct_under_1.0_deg angular threshold.

Also writes dataset-level summary statistics comparing our method vs ChatGPT:
  retinex2_output/_plots/summary_stats_vs_chatgpt.json

Assumes directory structure:
  retinex2_output/<image_stem>/summary.json

Each summary.json contains:
  {
    "metrics": {
        "<metric_name>": {"mean": ... , ...},
        "threshold_metrics": {"pct_under_1.0_deg": ...}
    },
    "chatgpt_metrics": {
        "<metric_name>": {"mean": ... , ...},
        "threshold_metrics": {"pct_under_1.0_deg": ...}
    }
  }

Outputs:
  retinex2_output/_plots/means_scatter_<metric>.png
  retinex2_output/_plots/pct_under_1.0deg_scatter.png
  retinex2_output/_plots/summary_stats_vs_chatgpt.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_metrics_from_summary(
    summary_path: Path,
) -> Tuple[Dict[str, float], Dict[str, float], Optional[float], Optional[float], Dict[str, float], Dict[str, float]]:
    """
    Returns:
      means: {metric_name: mean}
      chat_means: {metric_name: mean}
      pct_under_1deg: ours (float or None)
      chat_pct_under_1deg: chatgpt (float or None)
      thr_metrics: {threshold_key: value}                (ours)
      chat_thr_metrics: {threshold_key: value}           (chatgpt)
    """
    with summary_path.open("r") as f:
        data = json.load(f)

    means: Dict[str, float] = {}
    chat_means: Dict[str, float] = {}

    metrics = data.get("metrics", {}) or {}
    chat_metrics = data.get("chatgpt_metrics", {}) or {}

    # collect "mean" metrics present in both
    for k, v in metrics.items():
        if (
            isinstance(v, dict)
            and ("mean" in v)
            and (k in chat_metrics)
            and isinstance(chat_metrics[k], dict)
            and ("mean" in chat_metrics[k])
        ):
            means[k] = float(v["mean"])
            chat_means[k] = float(chat_metrics[k]["mean"])

    # collect threshold metrics dicts (if present)
    thr = metrics.get("threshold_metrics", {})
    chat_thr = chat_metrics.get("threshold_metrics", {})
    thr_metrics = dict(thr) if isinstance(thr, dict) else {}
    chat_thr_metrics = dict(chat_thr) if isinstance(chat_thr, dict) else {}

    # convenience: pct_under_1.0_deg
    pct_under_1deg = _safe_float(thr_metrics.get("pct_under_1.0_deg"))
    chat_pct_under_1deg = _safe_float(chat_thr_metrics.get("pct_under_1.0_deg"))

    return means, chat_means, pct_under_1deg, chat_pct_under_1deg, thr_metrics, chat_thr_metrics


def collect_dataset(
    root: Path,
) -> Tuple[
    Dict[str, List[Tuple[str, float, float]]],
    List[Tuple[str, float, float]],
    Dict[str, List[Tuple[str, float, float]]],
]:
    """
    Returns:
      mean_points_by_metric:
        metric -> list of (label, ours_mean, chat_mean)

      pct_under_1deg_points:
        list of (label, ours_pct, chat_pct)

      threshold_points_by_key:
        threshold_key -> list of (label, ours_value, chat_value)
        (e.g., pct_under_0.25_deg, pct_under_1.0_deg, ...)
    """
    mean_points_by_metric: Dict[str, List[Tuple[str, float, float]]] = {}
    pct_under_1deg_points: List[Tuple[str, float, float]] = []
    threshold_points_by_key: Dict[str, List[Tuple[str, float, float]]] = {}

    summary_paths = sorted(root.glob("*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No summary.json files found under: {root}")

    for sp in summary_paths:
        label = sp.parent.name
        means, chat_means, pct1, chat_pct1, thr, chat_thr = load_metrics_from_summary(sp)

        # means
        for metric_name in set(means.keys()) & set(chat_means.keys()):
            mean_points_by_metric.setdefault(metric_name, []).append(
                (label, means[metric_name], chat_means[metric_name])
            )

        # thresholds (all keys that exist in both)
        for k in set(thr.keys()) & set(chat_thr.keys()):
            v1 = _safe_float(thr.get(k))
            v2 = _safe_float(chat_thr.get(k))
            if v1 is not None and v2 is not None:
                threshold_points_by_key.setdefault(k, []).append((label, v1, v2))

        # pct_under_1deg convenience list
        if pct1 is not None and chat_pct1 is not None:
            pct_under_1deg_points.append((label, pct1, chat_pct1))

    return mean_points_by_metric, pct_under_1deg_points, threshold_points_by_key


def scatter_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: Optional[List[str]] = None,
    annotate: bool = False,
    force_unit_square: bool = False,
) -> plt.Figure:
    """Scatter plot with y=x reference line."""
    fig = plt.figure()
    ax = plt.gca()

    ax.scatter(x, y)

    finite = np.isfinite(x) & np.isfinite(y)
    if np.any(finite):
        if force_unit_square:
            lo, hi = 0.0, 1.0
        else:
            lo = float(min(np.min(x[finite]), np.min(y[finite])))
            hi = float(max(np.max(x[finite]), np.max(y[finite])))
            if lo == hi:
                lo -= 1e-6
                hi += 1e-6

        ax.plot([lo, hi], [lo, hi])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if annotate and labels is not None:
        for xi, yi, lab in zip(x, y, labels):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(lab, (xi, yi), fontsize=7, alpha=0.8)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    return fig


def _summary_from_pairs(
    ours: np.ndarray,
    chat: np.ndarray,
    *,
    higher_is_better: bool,
) -> Dict[str, Any]:
    """
    Compute dataset-level stats for paired values (ours vs chatgpt).
    Returns JSON-serializable dict.
    """
    finite = np.isfinite(ours) & np.isfinite(chat)
    ours = ours[finite]
    chat = chat[finite]

    if ours.size == 0:
        return {"count": 0}

    diff = ours - chat
    abs_diff = np.abs(diff)

    # win = better according to direction
    if higher_is_better:
        win = ours > chat
    else:
        win = ours < chat

    # correlation (guard against constant arrays)
    corr = None
    if ours.size >= 2 and (np.std(ours) > 0) and (np.std(chat) > 0):
        corr = float(np.corrcoef(ours, chat)[0, 1])

    return {
        "count": int(ours.size),
        "ours": {
            "mean": float(np.mean(ours)),
            "median": float(np.median(ours)),
            "std": float(np.std(ours)),
            "min": float(np.min(ours)),
            "max": float(np.max(ours)),
        },
        "chatgpt": {
            "mean": float(np.mean(chat)),
            "median": float(np.median(chat)),
            "std": float(np.std(chat)),
            "min": float(np.min(chat)),
            "max": float(np.max(chat)),
        },
        "diff_ours_minus_chatgpt": {
            "mean": float(np.mean(diff)),
            "median": float(np.median(diff)),
            "std": float(np.std(diff)),
        },
        "abs_diff": {
            "mean": float(np.mean(abs_diff)),
            "median": float(np.median(abs_diff)),
            "p90": float(np.percentile(abs_diff, 90)),
            "p95": float(np.percentile(abs_diff, 95)),
            "p99": float(np.percentile(abs_diff, 99)),
            "max": float(np.max(abs_diff)),
        },
        "win_rate": float(np.mean(win)),
        "pearson_corr": corr,
    }


def compute_summary_statistics(
    mean_points_by_metric: Dict[str, List[Tuple[str, float, float]]],
    threshold_points_by_key: Dict[str, List[Tuple[str, float, float]]],
) -> Dict[str, Any]:
    """
    Build summary statistics across the whole dataset for:
      - all mean metrics (assumed LOWER is better)
      - all threshold metrics (assumed HIGHER is better if key starts with 'pct_under_')
    """
    out: Dict[str, Any] = {
        "means": {},
        "thresholds": {},
    }

    # Mean metrics: typically error metrics => lower is better
    for metric, pts in sorted(mean_points_by_metric.items()):
        ours = np.array([p[1] for p in pts], dtype=np.float64)
        chat = np.array([p[2] for p in pts], dtype=np.float64)
        out["means"][metric] = _summary_from_pairs(ours, chat, higher_is_better=False)

    # Threshold metrics: "pct_under_*" => higher is better (usually)
    for key, pts in sorted(threshold_points_by_key.items()):
        ours = np.array([p[1] for p in pts], dtype=np.float64)
        chat = np.array([p[2] for p in pts], dtype=np.float64)

        higher_is_better = key.startswith("pct_under_")
        out["thresholds"][key] = _summary_from_pairs(ours, chat, higher_is_better=higher_is_better)

    return out


def save_summary_statistics(stats: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)


def main() -> None:
    root = Path("retinex2_output")
    out_dir = root / "_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_points_by_metric, pct_under_1deg_points, threshold_points_by_key = collect_dataset(root)

    annotate = False  # flip to True if you want labels on points

    # 1) Mean scatters (one per metric)
    for metric in sorted(mean_points_by_metric.keys()):
        pts = mean_points_by_metric[metric]
        labels = [p[0] for p in pts]
        x = np.array([p[1] for p in pts], dtype=np.float64)  # ours
        y = np.array([p[2] for p in pts], dtype=np.float64)  # chatgpt

        fig = scatter_xy(
            x,
            y,
            title=f"Mean {metric}: ours vs ChatGPT",
            xlabel="Ours (mean)",
            ylabel="ChatGPT (mean)",
            labels=labels,
            annotate=annotate,
            force_unit_square=False,
        )
        out_path = out_dir / f"means_scatter_{metric}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    # 2) pct_under_1.0_deg scatter
    if pct_under_1deg_points:
        labels = [p[0] for p in pct_under_1deg_points]
        x = np.array([p[1] for p in pct_under_1deg_points], dtype=np.float64)  # ours
        y = np.array([p[2] for p in pct_under_1deg_points], dtype=np.float64)  # chatgpt

        fig = scatter_xy(
            x,
            y,
            title="pct_under_1.0_deg (angular): ours vs ChatGPT",
            xlabel="Ours (pct under 1°)",
            ylabel="ChatGPT (pct under 1°)",
            labels=labels,
            annotate=annotate,
            force_unit_square=True,  # pct is in [0,1]
        )
        out_path = out_dir / "pct_under_1.0deg_scatter.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        print("No pct_under_1.0_deg values found in summaries; skipping that plot.")

    # 3) Save dataset-level summary stats (ours vs ChatGPT)
    stats = compute_summary_statistics(mean_points_by_metric, threshold_points_by_key)
    stats_path = out_dir / "summary_stats_vs_chatgpt.json"
    save_summary_statistics(stats, stats_path)

    print(f"Wrote plots to: {out_dir}")
    print(f"Wrote summary stats to: {stats_path}")


if __name__ == "__main__":
    main()
