"""Generate one supplementary Top 10 support-resource figure.

Notes
-----
Create a single supplementary figure showing the most frequently
mentioned support resources in Experiment 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis
from settings import ROOT


FIGURES = ROOT / "figures" / "resources"
INVENTORY = ROOT / "tables" / "supplement" / "resource_inventory_taxonomy.csv"

TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 10.2
PANEL_FILL = "white"
GRID = "#D9D9D9"
MUTED = analysis.MUTED
TOTAL_E1_REPLIES = 46_640
TOP_N = 10

GROUP_MAP = {
    "Named helpline": "Support Resources",
    "Organisation / charity": "Support Resources",
    "Public / institutional resource": "Support Resources",
}

RESOURCE_COLOUR = "#2E8B57"

DISPLAY = {
    "988 Suicide and Crisis Lifeline": "988 Suicide & Crisis Lifeline",
    "Papyrus HOPELINE247": "PAPYRUS HOPELINE247",
    "World Health Organization": "World Health Organization (WHO)",
    "headspace": "headspace",
    "love is respect": "love is respect",
}


def display_name(value: str) -> str:
    """Return standardised display names."""
    return DISPLAY.get(value, value)



def styled(display: float, width_inches: float = 7.4, label_points: float | None = None) -> float:
    """Apply publication plotting defaults and return scaled point size."""
    scale = display * TEXT_WIDTH_CM / (width_inches * 2.54)
    points = (LABEL_POINTS if label_points is None else label_points) / scale
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": points,
            "axes.labelsize": points,
            "axes.titlesize": points * 1.05,
            "xtick.labelsize": points * 0.95,
            "ytick.labelsize": points * 0.95,
            "legend.fontsize": points * 0.95,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": "black",
            "axes.linewidth": 0.7,
            "text.color": "black",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": "black",
            "ytick.labelcolor": "black",
        }
    )
    return points



def panel(ax) -> None:
    """Apply consistent axis styling."""
    ax.set_facecolor(PANEL_FILL)
    ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.45, color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def save(fig: plt.Figure, filename: str) -> Path:
    """Write a figure and print the filename."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {path.name}")
    return path



def nice_step(maximum: float) -> float:
    """Choose a readable major-tick spacing."""
    if maximum > 8:
        return 2.0
    if maximum > 4:
        return 1.0
    if maximum > 2:
        return 0.5
    return 0.2



def figure_height(rows: int) -> float:
    """Choose figure height from row count."""
    return max(3.4, 1.6 + 0.42 * rows)



def load_frame() -> pd.DataFrame:
    """Load and prepare the resource inventory."""
    frame = pd.read_csv(INVENTORY)
    frame = frame[frame["Resource Class"].isin(GROUP_MAP)].copy()
    frame["Figure Group"] = frame["Resource Class"].map(GROUP_MAP)
    frame["Rate"] = 100.0 * frame["E1 Replies"] / TOTAL_E1_REPLIES
    frame["Display"] = frame["Resource"].map(display_name)
    return frame



def top_support_resources(frame: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Return the top support resources by reply rate."""
    part = frame[frame["Figure Group"].eq("Support Resources")].copy()
    part = part.sort_values(["Rate", "Display"], ascending=[False, True]).head(top_n).copy()
    part = part.sort_values(["Rate", "Display"], ascending=[True, False]).reset_index(drop=True)
    return part



def draw_support_resources(frame: pd.DataFrame, display: float = 1.0) -> Path:
    """Draw the single publication resource figure."""
    part = top_support_resources(frame)

    width = 8.4
    points = styled(display, width_inches=width)
    fig, ax = plt.subplots(
        figsize=(width, figure_height(len(part))),
        constrained_layout=True,
    )

    bars = ax.barh(
        part["Display"],
        part["Rate"],
        height=0.64,
        facecolor=mpl.colors.to_rgba(RESOURCE_COLOUR, 0.22),
        edgecolor=RESOURCE_COLOUR,
        linewidth=1.10,
        zorder=3,
    )

    panel(ax)
    ax.set_xlabel("Mentions (%)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0)

    maximum = float(part["Rate"].max())
    step = nice_step(maximum)
    ceiling = (int(maximum / step) + 2) * step
    ax.set_xlim(0, ceiling)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))

    pad = max(0.03, maximum * 0.012)
    for bar, value in zip(bars, part["Rate"]):
        ax.text(
            value + pad,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=points * 0.78,
            color="black",
            zorder=5,
            clip_on=False,
        )

    return save(fig, "resources_support_top10.pdf")



def main(display: float = 1.0) -> None:
    """Generate the publication resource figure."""
    mpl.rcParams.update(analysis.STYLE)
    frame = load_frame()

    print("Resource figures\n")
    draw_support_resources(frame, display)
    print(f"\nWritten to {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
