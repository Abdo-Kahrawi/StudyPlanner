"""Shared plotting helpers for experiments and run_demo.py. Owner: Person B."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def grouped_bars(summary: pd.DataFrame, group: str, series: str, metric: str, path,
                 title: str, ylabel: str, log: bool = False, series_labels: dict | None = None):
    """Bars of metric_mean with metric_std error bars: one cluster per `group`, one bar per `series`."""
    groups = list(dict.fromkeys(summary[group]))
    names = list(dict.fromkeys(summary[series]))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(groups))
    for i, s in enumerate(names):
        sub = summary[summary[series] == s].set_index(group).reindex(groups)
        ax.bar(x + i * width - 0.4 + width / 2, sub[f"{metric}_mean"], width,
               yerr=sub[f"{metric}_std"].fillna(0), capsize=3, color=PALETTE[i % len(PALETTE)],
               label=(series_labels or {}).get(s, s))
    ax.set_xticks(x, [str(g) for g in groups])
    ax.set_xlabel(group.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("symlog")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
