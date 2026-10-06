"""Readability figures for the thesis.

python scripts/figureread.py
python scripts/figureread.py --set main
python scripts/figureread.py --only ladder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import language
from analysis import (
    COLOUR, INK, MARKER, MUTED, NAME, ORDER, PALE, STYLE,
    SCENARIO_COLOUR, SCENARIO_ORDER,
    bootstrap_paired, macro_average,
)
from settings import ROOT


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FIGURES = ROOT / "figures/readability"

# The primary contrast is drawn from the table the notebook froze, not
# recomputed here, so the figure and Table 4.12 cannot come apart. The script
# says so and stops if the file is absent, on the same rule figuresafe.py uses
# for the safety register.
CONDITIONING_PATH = ROOT / "tables/main/readability_02_conditioning.csv"
MACRO = "Macro-average"  # machine/data key; do not rename

def display_label(value):
    """Canonical display spelling for figure labels."""
    return "Macro-Average" if value == MACRO else value

TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 11
GRID_SIZE = (13.5, 7.8)

# A single axis carrying seven rows. Its width is passed to styled() as the real
# canvas width, so the type arrives at LABEL_POINTS once LaTeX has scaled the
# file to \textwidth. The grid figures above keep their existing call for now,
# since they are already tuned and re-scaling them would move every readability
# figure in the thesis.
FOREST_SIZE = (9.0, 5.4)

LADDER = [7, 9, 11, 13, 15, 17, 18, 21]
MINOR = [7, 9, 11, 13, 15, 17]
ADULT = [18, 21]
TYPES = list(SCENARIO_ORDER)
SLUG = {x: x.lower().replace(" ", "_") for x in TYPES}
TARGET = {age: min(age - 5, 12) for age in MINOR}

SIGNALS = [
    ("neutral", "Neutral", ""),
    ("adult_cue", "Adult", "(Cue)"),
    ("adult_age", "Adult", "(Age)"),
    ("minor_cue", "Minor", "(Cue)"),
    ("minor_age", "Minor", "(Age)"),
]
SIGNAL_TRACKS = [
    ("fkgl", "FKGL", "-", "o", 1.00),
    ("mean_aoa", "Mean AoA", "--", "^", 0.85),
    ("p90_aoa", "P90 AoA", ":", "s", 0.70),
]
SECONDARY_CONDITIONING = [
    ("mean_aoa", "Difference in Mean AoA", "readability_conditioning_mean_aoa.pdf"),
    ("p90_aoa", "Difference in P90 AoA", "readability_conditioning_p90_aoa.pdf"),
    ("response_length", "Difference in Response Length",
     "readability_conditioning_response_length.pdf"),
]
# Scenario line grammar. Colour comes from SCENARIO_COLOUR; line style and
# marker provide a redundant cue for print and colour-vision accessibility.
TYPE_STYLE = {
    "Benign": ("-.", "D"),
    "Rights": (":", "^"),
    "Age Restricted": ("--", "s"),
    "Harmful": ("-", "o"),
}

MEASURE_GROUPS = [
    ("Readability", [("fkgl", "FKGL"), ("fre", "FRE"), ("gunning_fog", "Gunning Fog"),
                     ("ari", "ARI"), ("smog", "SMOG")]),
    ("Vocabulary", [("mean_aoa", "Mean AoA"), ("p90_aoa", "P90 AoA"),
                    ("max_aoa", "Max AoA"), ("difficult_share", "Difficult Share"),
                    ("aoa_coverage", "AoA Coverage")]),
    ("Structure", [("response_length", "Response Length"),
                   ("sentence_length", "Sentence Length"), ("word_length", "Word Length"),
                   ("ttr", "TTR"), ("mtld", "MTLD")]),
]
MEASURES = [x for _, group in MEASURE_GROUPS for x in group]

PANEL_FILL = "#F5F5F5"
MINOR_BAND = "#E8EDF2"
ADULT_BAND = "#F2EDE8"

FORMULAE = ["fkgl", "fre", "gunning_fog", "ari", "smog"]
LEGEND = dict(
    loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.075),
    handlelength=2.2, handletextpad=0.7, columnspacing=1.5
)


# ---------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------

def by_scenario(df, measure, keys):
    cell = df.groupby(keys + ["scenario_id", "condition"], observed=True)[measure].mean()
    scenario = cell.groupby(keys + ["scenario_id"], observed=True).mean()
    return scenario.groupby(keys, observed=True).mean()


def blocks(df, column="fkgl"):
    wide = (df.pivot_table(index="scenario_id", columns="age", values=column, aggfunc="mean")
              .reindex(columns=MINOR + ADULT).dropna())
    if wide.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    return wide[MINOR].mean(axis=1), wide[ADULT].mean(axis=1)


def contrast(df, column="fkgl"):
    minor, adult = blocks(df, column)
    if minor.empty:
        return np.nan, np.nan, np.nan, 0
    diff = minor - adult
    point, low, high = bootstrap_paired(diff)
    return point, low, high, len(diff)


# ---------------------------------------------------------------------
# Shared figure grammar
# ---------------------------------------------------------------------

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
        "axes.edgecolor": MUTED,
        "axes.labelcolor": "black",
        "axes.linewidth": 0.7,
        "text.color": "black",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
    })
    return points


def grid(figsize=GRID_SIZE, **kwargs):
    return plt.subplots(2, 3, figsize=figsize, **kwargs)


def panel(ax, title=None, points=9):
    ax.set_facecolor(PANEL_FILL)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25, color=MUTED)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, pad=points * 0.5, color="black")


def map_panel(ax, title=None, points=9):
    """Heatmap/map axes: no grey panel fill and no plot grid."""
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_axisbelow(False)
    if title:
        ax.set_title(title, pad=points * 0.5, color="black")


def outer_labels(fig, axes, xlabel, ylabel, points):
    for ax in axes.flat:
        ax.label_outer()
    if xlabel:
        fig.supxlabel(xlabel, color="black", fontsize=points * 1.08)
    if ylabel:
        fig.supylabel(ylabel, color="black", fontsize=points * 1.15)


def legend(fig, handles, labels, points, ncol):
    fig.legend(handles, labels, ncol=ncol, fontsize=points * 1.05, **LEGEND)


def filename(prefix, kind):
    return f"{prefix}.pdf" if kind is None else f"{prefix}_{SLUG[kind]}.pdf"


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")
    return path


def annotate_contrast(ax, point, low, high, location="right"):
    """Draw a compact effect badge in unused panel space."""
    if not np.isfinite(point):
        return

    text = f"Δ {point:+.2f}\n[{low:+.2f}, {high:+.2f}]"
    x, ha = (0.035, "left") if location == "left" else (0.965, "right")
    ax.text(
        x, 0.955, text, transform=ax.transAxes, ha=ha, va="top",
        fontsize=plt.rcParams["font.size"] * 0.84,
        fontweight="semibold", linespacing=1.18, color=INK, zorder=7,
        bbox=dict(
            boxstyle="round,pad=0.24", facecolor="white", alpha=0.92,
            edgecolor="#D0D0D0", linewidth=0.55
        )
    )


def age_bands(ax):
    ax.axvspan(6.4, 17.5, color=MINOR_BAND, zorder=0)
    ax.axvspan(17.5, 21.6, color=ADULT_BAND, zorder=0)


def readable_on(rgba):
    r, g, b = rgba[:3]
    return "white" if 0.2126*r + 0.7152*g + 0.0722*b < 0.42 else "black"


def spread_labels(values, minimum_gap, lower, upper):
    """Return non-overlapping y positions while preserving value order."""
    values = np.asarray(values, dtype=float)
    positions = values.copy()
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < 2:
        return positions

    order = valid[np.argsort(values[valid])]
    placed = values[order].copy()

    # Forward pass: enforce the minimum vertical separation.
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + minimum_gap)

    # Shift the group back inside the plotting range if needed.
    if placed[-1] > upper:
        placed -= placed[-1] - upper
    if placed[0] < lower:
        placed += lower - placed[0]

    # A second backward pass keeps separation if clamping moved the group.
    for i in range(len(placed) - 2, -1, -1):
        placed[i] = min(placed[i], placed[i + 1] - minimum_gap)

    positions[order] = placed
    return positions


# ---------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------

def ladder_panel(ax, df, model, points, title=None, small=False):
    values = by_scenario(df, "fkgl", ["age"]).reindex(LADDER)
    age_bands(ax)

    ax.axvline(17.5, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.65, zorder=1)
    ax.plot(list(TARGET), list(TARGET.values()), ":", color=MUTED,
            linewidth=1.2 if small else 1.4, alpha=0.75, zorder=2)
    ax.plot(
        LADDER, values.values, marker=MARKER[model],
        markersize=3.5 if small else 4.8, markerfacecolor="white",
        markeredgecolor=COLOUR[model], markeredgewidth=1.1,
        linewidth=1.5 if small else 1.9, color=COLOUR[model], zorder=3
    )

    # Representative ages: child, adolescent, pre-threshold and adult threshold.
    # Labels are staggered, and high points are moved below the line so they do
    # not collide with panel titles or the top edge.
    base_specs = {
        7:  ((0, 11), "bottom"),
        13: ((0, 11), "bottom"),
        17: ((-12, 12), "bottom"),
        18: ((12, -14), "top"),
    }
    ymax = ax.get_ylim()[1]
    for age, (offset, va) in base_specs.items():
        value = values.loc[age]
        if not np.isfinite(value):
            continue

        dx, dy = offset
        if value > ymax - 0.65:
            dy, va = -14, "top"
            if age == 17:
                dx = -12
            elif age == 18:
                dx = 12

        note = ax.annotate(
            f"{value:.1f}", (age, value), xytext=(dx, dy),
            textcoords="offset points", ha="center", va=va,
            fontsize=points * 0.84, fontweight="bold",
            color=COLOUR[model], zorder=6
        )
        note.set_path_effects([
            pe.withStroke(linewidth=2.2, foreground="white"),
            pe.Normal(),
        ])

    panel(ax, title, points)
    ax.set_xticks(LADDER)
    ax.set_xlim(6.4, 21.6)
    ax.yaxis.set_major_locator(MultipleLocator(1))


def draw_ladder(frame, display, kind=None):
    points = styled(display)
    fig, axes = grid(figsize=(13.5, 7.6), sharex=True, sharey=True,
                     constrained_layout=True)

    for ax, model in zip(axes.flat, ORDER):
        part = frame[frame["label"] == model]
        ladder_panel(ax, part, model, points, model)
        annotate_contrast(ax, *contrast(part)[:3], location="left")

    outer_labels(fig, axes, "Age", "Flesch-Kincaid Grade Level", points)
    legend(
        fig,
        [Patch(facecolor=MINOR_BAND), Patch(facecolor=ADULT_BAND),
         Line2D([0], [0], color=MUTED, linestyle=":", linewidth=1.5)],
        ["Minor Ages", "Adult Ages", "Target"], points, 3
    )
    return save(fig, filename("readability_ladder", kind))


# ---------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------

def overlap(minor, adult, edges):
    if minor.empty or adult.empty:
        return np.nan
    a, _ = np.histogram(minor, bins=edges, density=True)
    b, _ = np.histogram(adult, bins=edges, density=True)
    return float(np.minimum(a, b).sum() * np.diff(edges)[0])


def draw_distribution(frame, display, kind=None):
    points = styled(display)
    edges = np.linspace(2, 14, 19)
    fig, axes = grid(figsize=(13.5, 7.6), sharex=True, sharey=True,
                     constrained_layout=True)

    suffix = "" if kind is None else f", {kind}"
    print(f"  overlap, minor against adult{suffix}:")

    for ax, model in zip(axes.flat, ORDER):
        part = frame[frame["label"] == model]
        minor, adult = blocks(part)

        ax.hist(minor, bins=edges, density=True, alpha=0.40,
                facecolor=COLOUR[model], edgecolor=COLOUR[model], linewidth=1.2)
        ax.hist(adult, bins=edges, density=True, histtype="step",
                linestyle="--", linewidth=2.0, color=COLOUR[model])
        panel(ax, model, points)
        annotate_contrast(ax, *contrast(part)[:3], location="right")
        print(f"    {model:<24}{overlap(minor, adult, edges):.0%}")

    outer_labels(fig, axes, "Flesch-Kincaid Grade Level", "Density", points)
    legend(
        fig,
        [Patch(facecolor=PALE, alpha=0.5, edgecolor=MUTED, linewidth=1.1),
         Line2D([0], [0], color=MUTED, linewidth=2.0, linestyle="--")],
        ["Minor Ages", "Adult Ages"], points, 2
    )
    return save(fig, filename("readability_distribution", kind))


# ---------------------------------------------------------------------
# Signal strength
# ---------------------------------------------------------------------

def signal_level(frame):
    level = pd.Series("", index=frame.index, dtype=object)
    cue = frame["signal"].eq("cue")
    condition = frame["condition"].astype(str)

    level[frame["condition"].eq("neutral")] = "neutral"
    level[cue & condition.str.contains("adult", na=False)] = "adult_cue"
    level[cue & condition.str.contains("minor", na=False)] = "minor_cue"
    level[frame["age"].isin(ADULT)] = "adult_age"
    level[frame["age"].isin(MINOR)] = "minor_age"
    return level


def draw_signals(frame, display, kind=None):
    points = styled(display)
    frame = frame.assign(level=signal_level(frame))
    frame = frame[frame["level"].ne("")]

    keys = [key for key, _, _ in SIGNALS]
    labels = [top if not sub else f"{top}\n{sub}" for _, top, sub in SIGNALS]
    fig, axes = grid(sharex=True, sharey=True, constrained_layout=True)

    for index, (ax, model) in enumerate(zip(axes.flat, ORDER)):
        part = frame[frame["label"] == model]
        ax.axhline(0, color=INK, linewidth=1.0, alpha=0.75, zorder=2)

        moved_tracks = []
        for measure, name, style, marker, weight in SIGNAL_TRACKS:
            levels = by_scenario(part, measure, ["level"]).reindex(keys)
            moved = (levels - levels.loc["neutral"]) / part[measure].std()
            moved_tracks.append(moved)

            ax.plot(
                range(len(keys)), moved.values, style, marker=marker,
                markersize=4.8, markerfacecolor="white",
                markeredgecolor=COLOUR[model], markeredgewidth=1.1,
                linewidth=1.9 * weight, color=COLOUR[model],
                alpha=max(weight, 0.78), zorder=3,
                label=name if index == 0 else None
            )

        panel(ax, model, points)
        ax.set_xticks(range(len(keys)), labels)
        ax.set_xlim(-0.5, 4.72)
        ax.margins(y=0.26)
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

        # Show exact values for Minor Cue and Minor Age in dedicated label
        # columns rather than directly on top of the trajectories.
        ymin, ymax = ax.get_ylim()
        pad = 0.07 * (ymax - ymin)

        for x, text_x in ((3, 3.18), (4, 4.18)):
            raw = np.array([series.iloc[x] for series in moved_tracks], dtype=float)

            label_y = spread_labels(
                raw,
                minimum_gap=0.17,
                lower=ymin + pad,
                upper=ymax - pad,
            )

            for value, y_text in zip(raw, label_y):
                if not np.isfinite(value):
                    continue

                note = ax.annotate(
                    f"{value:+.2f}",
                    xy=(x, value),
                    xytext=(text_x, y_text),
                    textcoords="data",
                    ha="left",
                    va="center",
                    fontsize=points * 0.82,
                    fontweight="bold",
                    color=COLOUR[model],
                    zorder=6,
                    arrowprops=dict(
                        arrowstyle="-",
                        color=COLOUR[model],
                        linewidth=0.50,
                        alpha=0.45,
                        shrinkA=2,
                        shrinkB=3,
                    ),
                )
                note.set_path_effects([
                    pe.withStroke(linewidth=2.0, foreground="white"),
                    pe.Normal(),
                ])

    outer_labels(fig, axes, "", "Change from Neutral (SD)", points)
    handles, names = axes.flat[0].get_legend_handles_labels()
    legend(fig, handles, names, points, 3)
    return save(fig, filename("readability_signals", kind))


# ---------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------

def draw_coverage(frame, floor, display):
    points = styled(display)
    frame = frame.assign(short=frame["response_length"] < floor)
    fig, axes = grid(sharex=True, sharey=True, constrained_layout=True)

    # Enough labels to make the plot quantitative without turning every line
    # segment into text. These capture the youngest point, mid-adolescence,
    # the minor/adult boundary, and the oldest point.
    label_ages = [7, 13, 17, 18, 21]

    for index, (ax, model) in enumerate(zip(axes.flat, ORDER)):
        part = frame[frame["label"] == model]
        series = {}

        for kind in TYPES:
            style, marker = TYPE_STYLE[kind]
            loss = (
                part[part["scenario_type"] == kind]
                .groupby("age")["short"].mean().reindex(LADDER) * 100
            )
            series[kind] = loss

            ax.plot(
                LADDER, loss.values, style, marker=marker,
                markersize=4.0, linewidth=1.6,
                color=SCENARIO_COLOUR[kind],
                markerfacecolor="white",
                markeredgecolor=SCENARIO_COLOUR[kind],
                markeredgewidth=0.9,
                label=kind if index == 0 else None,
            )

        panel(ax, model, points)
        ax.set_xticks(LADDER)
        ax.set_xlim(6.4, 21.6)
        ax.margins(y=0.12)

        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin

        for age in label_ages:
            values, colours = [], []

            for kind in TYPES:
                value = series[kind].get(age, np.nan)

                # Suppress tiny values: visually they are already zero and
                # printing them adds clutter without information.
                if np.isfinite(value) and value >= 1.5:
                    values.append(float(value))
                    colours.append(SCENARIO_COLOUR[kind])

            if not values:
                continue

            label_y = spread_labels(
                values,
                minimum_gap=max(2.2, 0.045 * yrange),
                lower=ymin + 0.04 * yrange,
                upper=ymax - 0.04 * yrange,
            )

            # Alternate left/right around the crowded 17/18 boundary.
            if age == 17:
                text_x, ha = age - 0.25, "right"
            elif age == 18:
                text_x, ha = age + 0.25, "left"
            elif age == 21:
                text_x, ha = age - 0.28, "right"
            else:
                text_x, ha = age + 0.22, "left"

            for value, y_text, colour in zip(values, label_y, colours):
                note = ax.annotate(
                    f"{value:.0f}",
                    xy=(age, value),
                    xytext=(text_x, y_text),
                    textcoords="data",
                    ha=ha,
                    va="center",
                    fontsize=points * 0.76,
                    fontweight="bold",
                    color=colour,
                    zorder=6,
                    arrowprops=dict(
                        arrowstyle="-",
                        color=colour,
                        linewidth=0.45,
                        alpha=0.40,
                        shrinkA=2,
                        shrinkB=3,
                    ),
                )
                note.set_path_effects([
                    pe.withStroke(linewidth=1.8, foreground="white"),
                    pe.Normal(),
                ])

    outer_labels(fig, axes, "Age", f"Below the {floor} Word Floor (\\%)", points)
    handles, names = axes.flat[0].get_legend_handles_labels()
    legend(fig, handles, names, points, 4)
    return save(fig, "readability_coverage.pdf")


def draw_coverage_grid(frame, floor, display):
    points = styled(display, 9.6, label_points=9.0)
    frame = frame.assign(short=frame["response_length"] < floor)
    ceiling = float(
        frame.groupby(["label", "scenario_type", "age"])["short"].mean().max() * 100
    )

    fig, axes = grid(figsize=(13.0, 8.0))
    fig.subplots_adjust(hspace=0.42, wspace=0.10, right=0.88, top=0.90, bottom=0.14)

    for index, (ax, model) in enumerate(zip(axes.flat, ORDER)):
        loss = (frame[frame["label"] == model]
                .pivot_table(index="scenario_type", columns="age",
                             values="short", aggfunc="mean")
                .reindex(index=TYPES, columns=LADDER) * 100)

        map_panel(ax, model, points)
        image = ax.imshow(loss.values, cmap="Blues", vmin=0, vmax=ceiling, aspect="auto")
        ax.set_xticks(range(len(LADDER)),
                      [str(x) for x in LADDER] if index // 3 == 1 else [])
        ax.set_yticks(range(len(TYPES)), TYPES if index % 3 == 0 else [])
        ax.tick_params(length=0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for row, col in np.ndindex(loss.shape):
            value = loss.iat[row, col]
            if np.isfinite(value):
                ax.text(
                    col, row, f"{value:.0f}", ha="center", va="center",
                    fontsize=points * 0.6,
                    color="white" if value > 0.55 * ceiling else "black"
                )

    fig.supxlabel("Age", color="black", fontsize=points, y=0.04)
    top, bottom = axes[0, -1].get_position(), axes[-1, -1].get_position()
    cax = fig.add_axes([top.x1 + 0.02, bottom.y0, 0.014, top.y1 - bottom.y0])
    bar = fig.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=points * 0.85, length=0, labelcolor="black")
    return save(fig, "readability_coverage_grid.pdf")


# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

def draw_correlations(frame, display):
    points = styled(display, 10.5, label_points=9.0)
    columns = [key for key, _ in MEASURES if key in frame.columns]
    labels = [label for key, label in MEASURES if key in frame.columns]
    matrix = frame[columns].corr()
    blues = colormaps["Blues"]

    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    map_panel(ax, None, points)
    image = ax.imshow(np.abs(matrix.values), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    edge = 0
    for _, group in MEASURE_GROUPS[:-1]:
        edge += len(group)
        ax.axhline(edge - 0.5, color="white", linewidth=2.2)
        ax.axvline(edge - 0.5, color="white", linewidth=2.2)

    for row, col in np.ndindex(matrix.shape):
        value = matrix.iat[row, col]
        text = f"{value:.2f}".replace("0.", ".").replace("-.", "$-$.")
        ax.text(
            col, row, text, ha="center", va="center",
            fontsize=points * 0.58, color=readable_on(blues(abs(value)))
        )

    cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.18)
    bar = fig.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=points * 0.85, length=0, labelcolor="black")
    fig.tight_layout()
    return save(fig, "readability_correlations.pdf")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

# Define function to draw the primary readability contrast as a forest.
#
# The distributions elsewhere in this chapter show what the grade levels look
# like; none of them shows the paired estimand itself, which is one number a
# model with an interval on it. A forest is the plainest way to put the six
# effects, their intervals and the panel figure on one axis, and it is the same
# grammar the safety chapter uses for its primary contrasts.
def draw_conditioning(display):
    """Primary paired FKGL contrast shown with the standard readability panel."""
    if not CONDITIONING_PATH.exists():
        raise SystemExit(
            f"{CONDITIONING_PATH.relative_to(ROOT)} is missing. Run "
            f"notebooks/16_readability.ipynb first: the effects drawn here are "
            f"the ones it published, and recomputing them separately would let "
            f"a figure and Table 4.12 disagree."
        )

    table = pd.read_csv(CONDITIONING_PATH).set_index("Model")
    rows = [MACRO] + ORDER[::-1]
    positions = np.arange(len(rows))

    points = styled(display, FOREST_SIZE[0])
    fig, ax = plt.subplots(figsize=FOREST_SIZE, layout="constrained")

    for position, model in zip(positions, rows):
        if model not in table.index:
            continue

        effect = float(table.at[model, "Effect (Grades)"])
        low = float(table.at[model, "95% CI Lower"])
        high = float(table.at[model, "95% CI Upper"])
        colour = INK if model == MACRO else COLOUR[model]

        ax.plot(
            [low, high], [position, position],
            color=colour,
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            effect, position,
            marker="D" if model == MACRO else MARKER[model],
            color=colour,
            markersize=7.0,
            linestyle="none",
            zorder=3,
        )

        note = ax.annotate(
            f"{effect:.2f} [{low:.2f}, {high:.2f}]",
            xy=(effect, position),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=points * 0.72,
            fontweight="semibold",
            color=colour,
            zorder=5,
            annotation_clip=False,
        )
        note.set_path_effects([
            pe.withStroke(linewidth=2.0, foreground=PANEL_FILL),
            pe.Normal(),
        ])

    ax.axvline(
        0,
        color=MUTED,
        linewidth=1.0,
        linestyle="--",
        alpha=0.70,
        zorder=1,
    )
    ax.axhline(0.5, color="0.80", linewidth=0.8)

    ax.set_yticks(positions)
    ax.set_yticklabels([display_label(row) for row in rows])
    ax.set_ylim(-0.8, len(rows) - 0.15)

    low_limit = float(table["95% CI Lower"].min()) - 0.28
    ax.set_xlim(low_limit, 0.18)
    ax.set_xlabel("Difference in FKGL")
    panel(ax, "Minor Ages vs Adult Ages", points)

    path = save(fig, filename("readability_conditioning", None))
    print(
        f"      {FOREST_SIZE[0]:.1f} in canvas, {points:.1f} pt set, "
        f"arriving at {LABEL_POINTS} pt"
    )
    return path


def conditioning_panel(ax, frame, measure, xlabel, rows, positions, points):
    """Draw one model-level forest panel for a secondary conditioning metric."""
    # The macro row is the joint bootstrap of Section 3.6.3 over the six
    # per-model differences, not a contrast over the pooled replies of all six.
    # Pooling weights a model by how many replies it returned, which is a
    # property of provider blocking rather than of behaviour.
    per_model = {}
    for model in ORDER:
        minor, adult = blocks(frame[frame["label"] == model], measure)
        per_model[model] = minor - adult

    summary = {}
    for model in rows:
        if model == MACRO:
            summary[model] = macro_average(per_model)
            continue
        part = frame[frame["label"] == model]
        point, low, high, _ = contrast(part, measure)
        summary[model] = (point, low, high)

    finite = [
        value
        for triple in summary.values()
        for value in triple
        if np.isfinite(value)
    ]
    if not finite:
        finite = [-1.0, 1.0]

    pad = 0.08 * (max(finite) - min(finite) if max(finite) > min(finite) else 1.0)
    x_low = min(finite) - pad
    x_high = max(finite) + pad

    for position, model in zip(positions, rows):
        point, low, high = summary[model]
        if not np.isfinite(point):
            continue

        colour = INK if model == MACRO else COLOUR[model]
        ax.plot(
            [low, high], [position, position],
            color=colour,
            linewidth=2.0,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            point, position,
            marker="D" if model == MACRO else MARKER[model],
            color=colour,
            markersize=6.6,
            linestyle="none",
            zorder=3,
        )

        note = ax.annotate(
            f"{point:.2f} [{low:.2f}, {high:.2f}]",
            xy=(point, position),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=points * 0.67,
            fontweight="semibold",
            color=colour,
            zorder=5,
            annotation_clip=False,
        )
        note.set_path_effects([
            pe.withStroke(linewidth=2.0, foreground=PANEL_FILL),
            pe.Normal(),
        ])

    ax.axvline(
        0,
        color=MUTED,
        linewidth=1.0,
        linestyle="--",
        alpha=0.70,
        zorder=1,
    )
    ax.axhline(0.5, color="0.80", linewidth=0.8)

    ax.set_xlim(x_low, x_high)
    ax.set_xlabel(xlabel)
    panel(ax, "Minor Ages vs Adult Ages", points)


def draw_conditioning_measure(frame, display, measure, xlabel, filename):
    """Draw one standalone conditioning forest for a single secondary metric."""
    rows = [MACRO] + ORDER[::-1]
    positions = np.arange(len(rows))

    points = styled(display, FOREST_SIZE[0])
    fig, ax = plt.subplots(figsize=FOREST_SIZE, layout="constrained")

    conditioning_panel(ax, frame, measure, xlabel, rows, positions, points)
    ax.set_yticks(positions)
    ax.set_yticklabels([display_label(row) for row in rows])
    ax.set_ylim(-0.8, len(rows) - 0.15)

    path = save(fig, filename)
    print(
        f"      {FOREST_SIZE[0]:.1f} in canvas, {points:.1f} pt set, "
        f"arriving at {LABEL_POINTS} pt"
    )
    return path

def registry(raw, frame, age_conditioned, floor, display):
    jobs = {
        "conditioning": (lambda: draw_conditioning(display), "main"),
        "conditioning_fkgl": (lambda: draw_conditioning(display), "supplement"),
        "conditioning_mean_aoa": (
            lambda: draw_conditioning_measure(
                age_conditioned, display, "mean_aoa", "Difference in Mean AoA",
                "readability_conditioning_mean_aoa.pdf"
            ),
            "supplement",
        ),
        "conditioning_p90_aoa": (
            lambda: draw_conditioning_measure(
                age_conditioned, display, "p90_aoa", "Difference in P90 AoA",
                "readability_conditioning_p90_aoa.pdf"
            ),
            "supplement",
        ),
        "conditioning_response_length": (
            lambda: draw_conditioning_measure(
                age_conditioned, display, "response_length",
                "Difference in Response Length",
                "readability_conditioning_response_length.pdf"
            ),
            "supplement",
        ),
        "ladder": (lambda: draw_ladder(age_conditioned, display), "main"),
        "distribution": (lambda: draw_distribution(age_conditioned, display), "main"),
        "signals": (lambda: draw_signals(frame, display), "main"),
        "coverage": (lambda: draw_coverage(raw, floor, display), "main"),
        "correlations": (lambda: draw_correlations(frame, display), "main"),
    }

    for kind in TYPES:
        slug = SLUG[kind]
        s = age_conditioned[age_conditioned["scenario_type"] == kind]
        f = frame[frame["scenario_type"] == kind]
        jobs[f"ladder_{slug}"] = (lambda data=s, k=kind: draw_ladder(data, display, k),
                                  "supplement")
        jobs[f"distribution_{slug}"] = (
            lambda data=s, k=kind: draw_distribution(data, display, k), "supplement"
        )
        jobs[f"signals_{slug}"] = (
            lambda data=f, k=kind: draw_signals(data, display, k), "supplement"
        )

    jobs["coverage_grid"] = (
        lambda: draw_coverage_grid(raw, floor, 1.0), "supplement"
    )
    return jobs


def main(args):
    # Start from exactly the same global Matplotlib grammar as figuresafe.py.
    # STYLE enables the shared faint x/y grid; panel() then applies the common
    # horizontal-grid styling on top. Heatmaps explicitly opt out via map_panel().
    plt.rcParams.update(STYLE)

    raw = language.load()
    raw["label"] = raw["model"].map(NAME)

    frame = raw.copy()
    short = frame["response_length"] < args.floor
    frame.loc[short, FORMULAE] = np.nan
    age_conditioned = frame[frame["signal"].eq("stated")]

    print(f"{len(raw):,} replies, {int(short.sum()):,} below the {args.floor} word floor\n")

    for name, (build, tier) in registry(raw, frame, age_conditioned, args.floor, args.display).items():
        if args.only not in {"all", name} or args.set not in {"both", tier}:
            continue
        try:
            build()
        except Exception as exc:
            print(f"  {name} FAILED, {type(exc).__name__}: {exc}")

    print(f"\nWritten to {FIGURES.relative_to(ROOT)}")


def parser():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--floor", type=int, default=50)
    cli.add_argument("--display", type=float, default=1.0)
    cli.add_argument("--only", default="all")
    cli.add_argument("--set", default="both", choices=["both", "main", "supplement"])
    return cli


if __name__ == "__main__":
    main(parser().parse_args())