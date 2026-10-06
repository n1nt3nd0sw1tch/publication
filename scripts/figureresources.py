"""External-resource figures for the thesis.

Exact visual grammar matched to scripts/figureread.py and scripts/figuresafe.py.

This version outputs:
- Numbers / Calls
- Support resources (Helplines + Organisations / public resources), top 15

python scripts/figureresources.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis
from settings import ROOT

FIGURES = ROOT / "figures" / "resources"
INVENTORY = ROOT / "tables" / "supplement" / "resource_inventory_taxonomy.csv"

TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 10.2
PANEL_FILL = "#F5F5F5"
MUTED = analysis.MUTED
TOTAL_E1_REPLIES = 46_640

GROUP_MAP = {
    "Emergency number": "Numbers / Calls",
    "Named helpline": "Support resources",
    "Organisation / charity": "Support resources",
    "Public / institutional resource": "Support resources",
}

RESOURCE_COLOUR = {
    "Numbers / Calls": "#D97706",
    "Support resources": "#2E8B57",
}

DISPLAY = {
    "988 Suicide and Crisis Lifeline": "988 Suicide & Crisis Lifeline",
    "Papyrus HOPELINE247": "PAPYRUS HOPELINE247",
    "World Health Organization": "World Health Organization (WHO)",
    "headspace": "headspace",
    "love is respect": "love is respect",
}

def display_name(value):
    return DISPLAY.get(value, value)

def styled(display, width_inches=7.4, label_points=None):
    scale = display * TEXT_WIDTH_CM / (width_inches * 2.54)
    points = (LABEL_POINTS if label_points is None else label_points) / scale
    plt.rcParams.update({
        "font.size": points,
        "axes.labelsize": points,
        "axes.titlesize": points * 1.05,
        "xtick.labelsize": points * 0.95,
        "ytick.labelsize": points * 0.95,
        "legend.fontsize": points * 0.95,
        "axes.edgecolor": analysis.MUTED,
        "axes.labelcolor": "black",
        "axes.linewidth": 0.7,
        "text.color": "black",
        "xtick.color": analysis.MUTED,
        "ytick.color": analysis.MUTED,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
    })
    return points

def panel(ax, title=None, points=9):
    ax.set_facecolor(PANEL_FILL)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25, color=MUTED)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, pad=points * 0.5, color="black")

def save(fig, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")
    return path

def nice_step(maximum):
    if maximum > 8:
        return 2.0
    if maximum > 3:
        return 1.0
    if maximum > 1:
        return 0.5
    return 0.2

def figure_height(rows):
    return max(3.0, 1.55 + 0.36 * rows)

def draw(frame, group, filename, display, top_n=None):
    part = frame[frame["Figure Group"].eq(group)].copy()
    part = part.sort_values(["Rate", "Display"], ascending=[False, True])
    if top_n is not None:
        part = part.head(top_n).copy()
    part = part.sort_values(["Rate", "Display"], ascending=[True, False])

    width = 9.0
    points = styled(display, width_inches=width)

    fig, ax = plt.subplots(
        figsize=(width, figure_height(len(part))),
        constrained_layout=True,
    )

    colour = RESOURCE_COLOUR[group]
    bars = ax.barh(
        part["Display"],
        part["Rate"],
        height=0.64,
        facecolor=mpl.colors.to_rgba(colour, 0.28),
        edgecolor=colour,
        linewidth=1.10,
        zorder=3,
    )

    panel(ax, None, points)
    ax.set_xlabel("Rate (%)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0)

    maximum = float(part["Rate"].max())
    step = nice_step(maximum)
    ceiling = (int(maximum / step) + 2) * step
    ax.set_xlim(0, ceiling)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))

    pad = max(0.025, maximum * 0.012)
    for bar, value in zip(bars, part["Rate"]):
        ax.text(
            value + pad,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=points * 0.68,
            color="black",
            zorder=5,
            clip_on=False,
        )

    return save(fig, filename)

def main(display=1.0):
    mpl.rcParams.update(analysis.STYLE)

    frame = pd.read_csv(INVENTORY)
    frame = frame[frame["Resource Class"].isin(GROUP_MAP)].copy()
    frame["Figure Group"] = frame["Resource Class"].map(GROUP_MAP)
    frame["Rate"] = 100.0 * frame["E1 Replies"] / TOTAL_E1_REPLIES
    frame["Display"] = frame["Resource"].map(display_name)

    print("Resource figures\n")
    draw(frame, "Numbers / Calls", "resources_numbers_calls.pdf", display, top_n=None)
    draw(frame, "Support resources", "resources_support_top15.pdf", display, top_n=15)

    print(f"\nWritten to {FIGURES.relative_to(ROOT)}/")

if __name__ == "__main__":
    main()
