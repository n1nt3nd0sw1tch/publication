"""Generate publication figures for Experiment 2.

Notes
-----
Create one main figure and two supplementary figures from frozen dialogue tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


# Paths

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
OUTPUT = ROOT / "figures" / "dialogue"


# Study Order

MODELS = [
    "GPT-5.6 Luna",
    "Claude Haiku 4.5",
    "Gemini 3.5 Flash Lite",
    "DeepSeek-V4 Flash",
    "Mistral Small 4",
    "Gemma 4 31B",
]
MACRO = "Macro-average"  # Frozen CSV label: do not change.
MACRO_DISPLAY = "Macro-Average"
MODEL_ORDER = MODELS + [MACRO]

MODEL_AXIS = {
    "GPT-5.6 Luna": "GPT-5.6 Luna",
    "Claude Haiku 4.5": "Claude Haiku 4.5",
    "Gemini 3.5 Flash Lite": "Gemini 3.5 Flash Lite",
    "DeepSeek-V4 Flash": "DeepSeek-V4 Flash",
    "Mistral Small 4": "Mistral Small 4",
    "Gemma 4 31B": "Gemma 4 31B",
    MACRO: MACRO_DISPLAY,
}

STRATA = ["Age Restricted", "Harmful"]
TURNS = ["Turn 1", "Turn 2", "Turn 3"]
METHODS = ["Emotional Pushback", "Purpose Reverse", "Role Play"]


# Publication Style

TEXT_WIDTH = 7.15

BLUE = "#009E73"
ORANGE = "#CC79A7"
BLACK = "#111111"
GREY = "#6F6F6F"
LIGHT_GREY = "#D8D8D8"
GRID = "#C7C7C7"

SCENARIO_COLOUR = {
    "Age Restricted": BLUE,
    "Harmful": ORANGE,
}

MODEL_COLOUR = {
    "GPT-5.6 Luna": "#0072B2",
    "Claude Haiku 4.5": "#D55E00",
    "Gemini 3.5 Flash Lite": "#009E73",
    "DeepSeek-V4 Flash": "#CC79A7",
    "Mistral Small 4": "#E69F00",
    "Gemma 4 31B": "#56B4E9",
    MACRO: "#5E3C99",
}

MODEL_MARKER = {
    "GPT-5.6 Luna": "o",
    "Claude Haiku 4.5": "s",
    "Gemini 3.5 Flash Lite": "^",
    "DeepSeek-V4 Flash": "D",
    "Mistral Small 4": "P",
    "Gemma 4 31B": "X",
    MACRO: "D",
}

TURN_STYLE = {
    "T2": dict(color=BLUE, marker="o", fillstyle="none"),
    "T3": dict(color=ORANGE, marker="s", fillstyle="full"),
}

METHODS_COLOUR = {
    "T2": "#009E73",
    "T3": "#CC79A7",
}


def set_style():
    """Apply a journal-style Matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 8.3,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.6,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.72,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def style_axis(ax, grid_axis="both"):
    """Apply shared axis styling."""
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(0.8)
    ax.tick_params(length=2.8, width=0.7, direction="out")


def read_table(tables, name):
    """Read a frozen table from main or supplement."""
    for tier in ("main", "supplement"):
        path = tables / tier / f"{name}.csv"
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Missing {name}.csv under {tables}. "
        "Run the dialogue analysis notebook first."
    )


def ordered_models(frame):
    """Return rows in publication model order."""
    if frame["Model"].duplicated().any():
        raise ValueError("Expected one row per model")
    result = frame.set_index("Model").reindex(MODEL_ORDER)
    if result.isna().all(axis=1).any():
        missing = result.index[result.isna().all(axis=1)].tolist()
        raise ValueError(f"Missing model rows: {missing}")
    return result


def signed(value):
    """Format a signed one-decimal estimate."""
    return f"{float(value):+.1f}"


def interval(low, high):
    """Format a one-decimal confidence interval."""
    return f"[{float(low):.1f}, {float(high):.1f}]"


def save_figure(fig, output, name, png=False):
    """Save vector PDF and optional PNG."""
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    if png:
        fig.savefig(
            output / f"{name}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)
    print(f"Figure: {pdf.name}")


def draw_retention(tables, output, png=False):
    """Create the main Experiment 2 figure."""
    data = read_table(tables, "dialogue_04_age")
    data = data[data["Measure"].eq("Refusal")].copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.7, 4.5),
        gridspec_kw={"width_ratios": [0.88, 1.42]},
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        bottom=0.245,
        top=0.90,
        wspace=0.75,
    )

    ax = axes[0]
    style_axis(ax, "both")
    x = np.arange(3)

    line_specs = [
        ("Age Restricted", "Minor (%)", "-", "o", "Age Restricted · Minor"),
        ("Age Restricted", "Age 18 (%)", "--", "s", "Age Restricted · Age 18"),
        ("Harmful", "Minor (%)", "-", "^", "Harmful · Minor"),
        ("Harmful", "Age 18 (%)", "--", "D", "Harmful · Age 18"),
    ]

    handles = []
    for stratum, column, linestyle, marker, label in line_specs:
        part = data[
            data["Scenario Type"].eq(stratum)
            & data["Model"].eq(MACRO)
        ].set_index("Turn").reindex(TURNS)

        if part[column].isna().any():
            raise ValueError(f"Missing trajectory data for {label}")

        handle, = ax.plot(
            x,
            part[column].to_numpy(float),
            color=SCENARIO_COLOUR[stratum],
            linestyle=linestyle,
            marker=marker,
            markerfacecolor=(
                "white"
                if column == "Age 18 (%)"
                else SCENARIO_COLOUR[stratum]
            ),
            markeredgecolor=SCENARIO_COLOUR[stratum],
            markeredgewidth=1.0,
            linewidth=1.65,
            markersize=4.4,
            label=label,
            zorder=3,
        )
        handles.append(handle)

    ax.set_title("Refusal Across Turns", pad=5)
    ax.set_xticks(x, ["T1", "T2", "T3"])
    ax.set_xlim(-0.14, 2.14)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("Refusal Rate (%)")

    ax = axes[1]
    style_axis(ax, "both")
    ax.axvline(
        0,
        color=BLACK,
        linestyle=(0, (4, 3)),
        linewidth=0.9,
        zorder=1,
    )

    y = np.arange(len(MODEL_ORDER))
    offsets = {"Age Restricted": -0.13, "Harmful": 0.13}
    scenario_markers = {"Age Restricted": "o", "Harmful": "s"}

    for stratum in STRATA:
        part = data[
            data["Scenario Type"].eq(stratum)
            & data["Turn"].eq("Turn 3")
        ]
        ordered = ordered_models(part)

        estimate = ordered["Erosion (pp)"].to_numpy(float)
        low = ordered["Erosion CI Lower"].to_numpy(float)
        high = ordered["Erosion CI Upper"].to_numpy(float)

        if not np.isfinite(np.r_[estimate, low, high]).all():
            raise ValueError(f"Non-finite erosion values for {stratum}")

        colour = SCENARIO_COLOUR[stratum]
        yy = y + offsets[stratum]

        for index, model in enumerate(MODEL_ORDER):
            is_macro = model == MACRO
            ax.errorbar(
                estimate[index],
                yy[index],
                xerr=np.array(
                    [
                        [estimate[index] - low[index]],
                        [high[index] - estimate[index]],
                    ]
                ),
                fmt=scenario_markers[stratum],
                color=colour,
                ecolor=colour,
                markerfacecolor=colour if not is_macro else "white",
                markeredgecolor=colour,
                markeredgewidth=1.15,
                markersize=5.0 if is_macro else 4.0,
                linewidth=1.45 if is_macro else 1.15,
                elinewidth=1.45 if is_macro else 1.05,
                capsize=2.2,
                zorder=3,
            )

        macro_value = estimate[-1]
        macro_low = low[-1]
        macro_high = high[-1]
        macro_y = yy[-1]
        if macro_value <= 0:
            label_x = macro_high + 1.4
            label_ha = "left"
        else:
            label_x = macro_low - 1.4
            label_ha = "right"
        ax.text(
            label_x,
            macro_y,
            signed(macro_value),
            ha=label_ha,
            va="center",
            color=colour,
            fontsize=7.2,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "pad": 0.35,
                "alpha": 0.92,
            },
            zorder=5,
            clip_on=False,
        )

    ax.axhline(
        len(MODEL_ORDER) - 1.5,
        color=LIGHT_GREY,
        linewidth=0.8,
        zorder=1,
    )
    ax.set_yticks(
        y,
        [MODEL_AXIS[model] for model in MODEL_ORDER],
        fontsize=7.35,
    )
    ax.set_ylim(len(MODEL_ORDER) - 0.55, -0.65)
    lower = min(
        float(data.loc[data["Turn"].eq("Turn 3"), "Erosion CI Lower"].min()),
        -5.0,
    )
    upper = max(
        float(data.loc[data["Turn"].eq("Turn 3"), "Erosion CI Upper"].max()),
        5.0,
    )
    ax.set_xlim(lower - 4.5, upper + 4.5)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Age-Gap Change, T3 − T1 (pp)")
    ax.set_title("Age-Gap Erosion", pad=5)

    ax.text(
        0.02,
        0.972,
        r"$\Delta$ Gap < 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color=GREY,
    )
    ax.text(
        0.98,
        0.972,
        r"$\Delta$ Gap > 0",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color=GREY,
    )

    figure_legend = fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        ncol=4,
        loc="lower center",
        bbox_to_anchor=((fig.subplotpars.left + fig.subplotpars.right) / 2, 0.045),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor=LIGHT_GREY,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.15,
        borderpad=0.45,
    )
    figure_legend.get_frame().set_linewidth(0.6)
    for line in figure_legend.get_lines():
        line.set_linewidth(1.65)

    save_figure(fig, output, "dialogue_retention", png)

    age_restricted = data[
        data["Scenario Type"].eq("Age Restricted")
        & data["Model"].eq(MACRO)
    ].set_index("Turn")
    harmful = data[
        data["Scenario Type"].eq("Harmful")
        & data["Model"].eq(MACRO)
    ].set_index("Turn")

    caption = (
        "Age-conditioned refusal under conversational pressure. "
        "Panel (a) shows macro-average refusal for explicit minor ages and age 18 "
        "across dialogue turns in Age Restricted and Harmful scenarios. "
        "Panel (b) shows the model-level change in the minor-minus-age-18 refusal "
        "gap from T1 to T3; horizontal intervals are paired scenario-bootstrap "
        "95% confidence intervals. Negative values indicate erosion. "
        f"For Age Restricted scenarios, the macro-average gap changes from "
        f"{age_restricted.loc['Turn 1', 'Gap (pp)']:.1f} pp at T1 to "
        f"{age_restricted.loc['Turn 3', 'Gap (pp)']:.1f} pp at T3 "
        f"(erosion {signed(age_restricted.loc['Turn 3', 'Erosion (pp)'])} pp, "
        f"95% CI {interval(age_restricted.loc['Turn 3', 'Erosion CI Lower'], age_restricted.loc['Turn 3', 'Erosion CI Upper'])}). "
        f"The Harmful control changes by "
        f"{signed(harmful.loc['Turn 3', 'Erosion (pp)'])} pp "
        f"(95% CI {interval(harmful.loc['Turn 3', 'Erosion CI Lower'], harmful.loc['Turn 3', 'Erosion CI Upper'])})."
    )
    return caption


def draw_boundary(tables, output, png=False):
    """Create the age-17 versus age-18 supplementary figure."""
    data = read_table(tables, "dialogue_s14_boundary")
    data = data[data["Measure"].eq("Refusal")].copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.55, 3.65),
        sharey=True,
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.295,
        right=0.990,
        bottom=0.19,
        top=0.88,
        wspace=0.12,
    )

    macro_values = []

    for panel, (ax, turn) in enumerate(zip(axes, TURNS)):
        style_axis(ax, "both")
        ax.axvline(
            0,
            color=BLACK,
            linestyle=(0, (4, 3)),
            linewidth=0.85,
            zorder=1,
        )

        part = data[data["Turn"].eq(turn)]
        ordered = ordered_models(part)

        estimate = ordered["17 minus 18 (pp)"].to_numpy(float)
        low = ordered["CI Lower"].to_numpy(float)
        high = ordered["CI Upper"].to_numpy(float)

        y = np.arange(len(MODEL_ORDER))
        for index, model in enumerate(MODEL_ORDER):
            colour = MODEL_COLOUR[model]
            is_macro = model == MACRO

            ax.errorbar(
                estimate[index],
                y[index],
                xerr=np.array(
                    [
                        [estimate[index] - low[index]],
                        [high[index] - estimate[index]],
                    ]
                ),
                fmt=MODEL_MARKER[model],
                color=colour,
                ecolor=colour,
                markerfacecolor=colour,
                markeredgecolor=colour,
                markersize=5.1 if is_macro else 3.7,
                linewidth=1.55 if is_macro else 1.0,
                elinewidth=1.55 if is_macro else 1.0,
                capsize=2.1,
                zorder=3,
            )

        ax.axhline(
            len(MODEL_ORDER) - 1.5,
            color=LIGHT_GREY,
            linewidth=0.8,
            zorder=1,
        )
        ax.set_ylim(len(MODEL_ORDER) - 0.45, -0.55)
        ax.set_xlim(-8, 76)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.set_title(f"T{panel + 1}", pad=5)

        if panel == 0:
            ax.set_yticks(
                y,
                [MODEL_AXIS[model] for model in MODEL_ORDER],
            )
        else:
            ax.tick_params(labelleft=False)

        macro = float(estimate[-1])
        macro_values.append(macro)
        label_x = macro + 1.8
        ax.text(
            label_x,
            y[-1],
            signed(macro),
            ha="left",
            va="center",
            fontsize=6.9,
            fontweight="bold",
            color=MODEL_COLOUR[MACRO],
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "pad": 0.30,
                "alpha": 0.92,
            },
            zorder=5,
            clip_on=False,
        )

    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    fig.text(
        (left + right) / 2,
        0.05,
        "Refusal Gap 17 vs 18 (pp)",
        ha="center",
        va="center",
        fontsize=8.7,
    )

    save_figure(fig, output, "dialogue_boundary", png)

    caption = (
        "Persistence of the statutory age boundary. Points show the difference "
        "in refusal between age 17 and age 18 at each dialogue turn; horizontal "
        "intervals are paired scenario-bootstrap 95% confidence intervals. "
        "Positive values indicate more refusal at age 17. "
        f"The macro-average gap narrows from {macro_values[0]:.1f} pp at T1 "
        f"to {macro_values[1]:.1f} pp at T2 and {macro_values[2]:.1f} pp at T3."
    )
    return caption


def draw_methods(tables, output, png=False):
    """Create the attack-method supplementary figure."""
    data = read_table(tables, "dialogue_05_methods")
    data = data[
        data["Scenario Type"].isin(STRATA)
        & data["Method"].isin(METHODS)
        & data["Model"].eq(MACRO)
    ].copy()

    expected = len(STRATA) * len(METHODS)
    if len(data) != expected:
        raise ValueError(
            f"Expected {expected} macro-average method rows; found {len(data)}"
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.5, 3.35),
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.30,
        top=0.84,
        wspace=0.10,
    )

    x = np.arange(len(METHODS))
    method_labels = ["Emotional Pushback", "Purpose Reverse", "Role Play"]

    all_values = []

    for panel, (ax, stratum) in enumerate(zip(axes, STRATA)):
        style_axis(ax, "both")
        ax.axhline(
            0,
            color=BLACK,
            linestyle=(0, (4, 3)),
            linewidth=0.85,
            zorder=1,
        )

        part = (
            data[data["Scenario Type"].eq(stratum)]
            .set_index("Method")
            .reindex(METHODS)
        )
        if part.isna().all(axis=1).any():
            raise ValueError(f"Missing method rows for {stratum}")

        t2 = part["Refusal Change Turn 2 (pp)"].to_numpy(float)
        t3 = part["Refusal Change Turn 3 (pp)"].to_numpy(float)
        all_values.extend(t2.tolist())
        all_values.extend(t3.tolist())

        for index in range(len(METHODS)):
            ax.plot(
                [x[index], x[index]],
                [t2[index], t3[index]],
                color=LIGHT_GREY,
                linewidth=1.15,
                zorder=2,
            )

        ax.plot(
            x,
            t2,
            linestyle="None",
            marker=TURN_STYLE["T2"]["marker"],
            markerfacecolor="white",
            markeredgecolor=METHODS_COLOUR["T2"],
            markeredgewidth=1.2,
            color=METHODS_COLOUR["T2"],
            markersize=5.2,
            label="T2",
            zorder=3,
        )
        ax.plot(
            x,
            t3,
            linestyle="None",
            marker=TURN_STYLE["T3"]["marker"],
            markerfacecolor=TURN_STYLE["T3"]["color"],
            markeredgecolor=TURN_STYLE["T3"]["color"],
            markeredgewidth=1.0,
            color=TURN_STYLE["T3"]["color"],
            markersize=5.0,
            label="T3",
            zorder=3,
        )

        ax.set_xticks(x, method_labels)
        ax.set_xlim(-0.35, len(METHODS) - 0.65)
        ax.set_title(
            "Age Restricted"
            if panel == 0
            else "Harmful Control",
            pad=6,
        )

        if panel == 0:
            ax.set_ylabel("Change in Refusal from T1 (pp)")

    minimum = min(all_values + [0.0])
    maximum = max(all_values + [0.0])
    lower = 10 * np.floor((minimum - 5) / 10)
    upper = 10 * np.ceil((maximum + 5) / 10)

    for ax in axes:
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(MultipleLocator(20))

    handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=METHODS_COLOUR["T2"],
            markeredgewidth=1.2,
            color=METHODS_COLOUR["T2"],
            markersize=5.0,
            label="T2",
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="s",
            markerfacecolor=METHODS_COLOUR["T3"],
            markeredgecolor=METHODS_COLOUR["T3"],
            color=METHODS_COLOUR["T3"],
            markersize=4.8,
            label="T3",
        ),
    ]
    methods_legend = fig.legend(
        handles=handles,
        labels=["T2", "T3"],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=((fig.subplotpars.left + fig.subplotpars.right) / 2, 0.02),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor=LIGHT_GREY,
        handletextpad=0.55,
        columnspacing=1.4,
        borderpad=0.45,
    )
    methods_legend.get_frame().set_linewidth(0.6)

    save_figure(fig, output, "dialogue_methods", png)

    caption = (
        "Refusal change by conversational attack method. Points show the "
        "macro-average change from the shared T1 opening at T2 and T3 for "
        "Emotional Pushback, Purpose Reverse and Role Play, separately for "
        "Age Restricted scenarios and the Harmful control. Vertical connectors "
        "link the two pressed turns for the same method. Negative values indicate "
        "lower refusal than at the opening. These method-level refusal changes are "
        "descriptive because the frozen method table does not report confidence "
        "intervals for this measure."
    )
    return caption


FIGURES = {
    "retention": ("main", draw_retention),
    "boundary": ("supplement", draw_boundary),
    "methods": ("supplement", draw_methods),
}


def main():
    """Generate publication figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables",
        type=Path,
        default=TABLES,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
    )
    parser.add_argument(
        "--set",
        choices=["main", "supplement", "all"],
        default="all",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(FIGURES),
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also export 300-dpi PNG files",
    )
    args = parser.parse_args()

    set_style()

    jobs = []
    for key, (tier, function) in FIGURES.items():
        if args.set != "all" and tier != args.set:
            continue
        if args.only and key not in args.only:
            continue
        jobs.append((key, tier, function))

    if not jobs:
        parser.error("No figures match the requested selection")

    for _, _, function in jobs:
        function(
            args.tables,
            args.output,
            args.png,
        )


if __name__ == "__main__":
    main()
