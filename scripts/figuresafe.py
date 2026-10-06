"""Safety figures for the thesis and paper.

The figure grammar deliberately mirrors scripts/figureread.py:
- the same typography scaling;
- model panels where model is the natural unit;
- outer axis labels rather than repeated labels;
- bottom legends;
- restrained panel titles only;
- scenario-weighted rates throughout.

Examples
--------
python scripts/figuresafe.py
python scripts/figuresafe.py --set main
python scripts/figuresafe.py --set supplement
python scripts/figuresafe.py --only outcomes_refusal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis
from settings import ROOT, measure_column


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FIGURES = ROOT / "figures" / "safety"
REGISTER = analysis.MACHINE / "register_safety.csv"

TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 10.2
GRID_SIZE = (13.5, 7.8)
WIDE_SIZE = (13.5, 6.2)

PANEL_FILL = "#F5F5F5"
MUTED = analysis.MUTED
MINOR_BAND = "#E8EDF2"
ADULT_BAND = "#F2EDE8"

LEGEND = dict(
    loc="lower center",
    frameon=False,
    bbox_to_anchor=(0.5, -0.065),
    handlelength=2.1,
    handletextpad=0.6,
    columnspacing=1.35,
)

MODEL_AXIS = {
    "GPT-5.6 Luna": "GPT-5.6\nLuna",
    "Claude Haiku 4.5": "Claude\nHaiku 4.5",
    "Gemini 3.5 Flash Lite": "Gemini 3.5\nFlash Lite",
    "DeepSeek-V4 Flash": "DeepSeek-V4\nFlash",
    "Mistral Small 4": "Mistral\nSmall 4",
    "Gemma 4 31B": "Gemma 4\n31B",
    analysis.MACRO: "Macro-\nAverage",
}

SCENARIOS = list(analysis.SCENARIO_ORDER)
SCENARIO_COLOUR = analysis.SCENARIO_COLOUR
SCENARIO_AXIS = {
    "Benign": "Benign",
    "Rights": "Rights",
    "Age Restricted": "Age\nRestricted",
    "Harmful": "Harmful",
}

# Same scenario grammar as readability coverage.
TYPE_STYLE = {
    "Benign": ("-.", "D", 0.34),
    "Rights": (":", "^", 0.52),
    "Age Restricted": ("--", "s", 0.72),
    "Harmful": ("-", "o", 0.92),
}

AGE_MAP = {
    condition: int(condition.replace("age", ""))
    for condition in analysis.CONDITION_ORDER
    if condition.startswith("age")
}
AGE_CONDITIONS = sorted(AGE_MAP, key=AGE_MAP.get)
AGES = [AGE_MAP[condition] for condition in AGE_CONDITIONS]
AGE_CONDITION = {AGE_MAP[condition]: condition for condition in AGE_CONDITIONS}
MINOR_AGE_CONDITIONS = [condition for condition in AGE_CONDITIONS if AGE_MAP[condition] < 18]
ADULT_AGE_CONDITIONS = [condition for condition in AGE_CONDITIONS if AGE_MAP[condition] >= 18]

LABEL_GROUPS = [
    ("Neutral", [analysis.NEUTRAL]),
    ("Adult (Cue)", list(analysis.IMPLICIT_ADULT)),
    ("Adult (Age)", ADULT_AGE_CONDITIONS),
    ("Minor (Cue)", list(analysis.IMPLICIT_MINOR)),
    ("Minor (Age)", MINOR_AGE_CONDITIONS),
]
SIGNAL_KEYS = [label for label, _ in LABEL_GROUPS]
SIGNAL_AXIS = {
    "Neutral": "Neutral",
    "Adult (Cue)": "Adult\n(Cue)",
    "Adult (Age)": "Adult\n(Age)",
    "Minor (Cue)": "Minor\n(Cue)",
    "Minor (Age)": "Minor\n(Age)",
}

CELL_COLUMNS = {
    cell: cell.lower().replace(" ", "_")
    for cell in analysis.CELL_ORDER
}

# Main descriptive figure: keep the fields that carry the safety argument.
# Identity / boundary fields remain in the dedicated supplementary field maps.
RATE_FIELDS = [
    ("refusal", "Refusal"),
    ("delivered", "Delivery"),
    (measure_column("Alternative Response"), "Alternative"),
    (measure_column("Risk Statement"), "Risk"),
    (measure_column("Legal Statement"), "Legal"),
    (measure_column("Social Signpost"), "Social"),
    (measure_column("Expert Signpost"), "Expert"),
    (measure_column("Service Signpost"), "Service"),
]

# Main conditional-safeguard comparison: enough fields to answer the question
# without turning every panel into a wall of labels.
SAFEGUARD_MAIN = [
    (measure_column("Risk Statement"), "Risk"),
    (measure_column("Legal Statement"), "Legal"),
    (measure_column("Social Signpost"), "Social"),
    (measure_column("Expert Signpost"), "Expert"),
    (measure_column("Service Signpost"), "Service"),
    (measure_column("System Identity"), "System"),
]

SAFEGUARD_ALL = [
    (measure_column("Risk Statement"), "Risk"),
    (measure_column("Legal Statement"), "Legal"),
    (measure_column("Eligibility Statement"), "Eligibility"),
    (measure_column("Social Signpost"), "Social"),
    (measure_column("Expert Signpost"), "Expert"),
    (measure_column("Service Signpost"), "Service"),
    (measure_column("System Identity"), "System"),
    (measure_column("Boundary Identity"), "Boundary"),
    (measure_column("Limitation Identity"), "Limitation"),
    (measure_column("Companion Identity"), "Companion"),
]

PRIMARY = [
    analysis.TRAJECTORY,
    analysis.THRESHOLD_CONTRAST,
    analysis.SIGNAL,
]
PRIMARY_LABEL = {
    analysis.TRAJECTORY: "Minor Ages vs Adult Ages",
    analysis.THRESHOLD_CONTRAST: "Age 17 vs Age 18",
    analysis.SIGNAL: "Minor (Age) vs Minor (Cue)",
}

FIGURESPEC = {
    "rates": ("main", "safety_rates.pdf", "Safety and response rates by scenario"),
    "outcomes_refusal": (
        "main",
        "safety_outcomes_refusal.pdf",
        "Refusal outcome cells by scenario and model",
    ),
    "outcomes_compliance": (
        "main",
        "safety_outcomes_compliance.pdf",
        "Compliance outcome cells by scenario and model",
    ),
    "safeguards": (
        "main",
        "safety_safeguards.pdf",
        "Safeguards within refusal and compliance",
    ),
    "trajectory": ("main", "safety_trajectory.pdf", "Refusal trajectory across exact ages"),
    "primary": ("main", "safety_primary.pdf", "Primary age-conditioning contrasts"),
    "controls": ("supplement", "safety_controls.pdf", "Age trajectories by scenario"),
    "cues": ("supplement", "safety_cues.pdf", "Refusal by age-signal strength"),
    "fields_age": ("supplement", "safety_fields_age.pdf", "Safeguards by age"),
    "fields_labels": ("supplement", "safety_fields_labels.pdf", "Safeguards by age signal"),
    "fields_scenarios": ("supplement", "safety_fields_scenarios.pdf", "Safeguards by scenario"),
    "failures": ("supplement", "safety_failures.pdf", "Directional safety failures"),
    "age_steps": (
        "supplement",
        "safety_age_steps.pdf",
        "Age-to-age refusal-rate changes within age-restricted scenarios",
    ),
    "tradeoff": (
        "supplement",
        "safety_tradeoff.pdf",
        "Trade-off between over-permissive and over-restrictive behaviour",
    ),
    "safeguards_macro": (
        "supplement",
        "safety_safeguards_macro.pdf",
        "Macro-average safeguard profiles within refusal and compliance",
    ),
}

STALE = [
    "safety_basics.pdf",
    "safety_main_metrics.pdf",
    "safety_compliance_fields.pdf",
    "safety_characteristics.pdf",
    "safety_mismatch.pdf",
    "safety_age.pdf",
    "safety_labels.pdf",
    "safety_scenarios.pdf",
    "safety_outcomes.pdf",
    "safety_failures_protective.pdf",
    "safety_failures_overrestriction.pdf",
]
for cell in analysis.CELL_ORDER:
    slug = cell.lower().replace(" ", "_")
    STALE += [
        f"safety_age_{slug}.pdf",
        f"safety_labels_{slug}.pdf",
        f"safety_scenarios_{slug}.pdf",
        f"safety_safeguards_{slug}.pdf",
    ]


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
    for ax in np.asarray(axes).flat:
        ax.label_outer()
    if xlabel:
        fig.supxlabel(xlabel, color="black", fontsize=points * 1.05)
    if ylabel:
        fig.supylabel(ylabel, color="black", fontsize=points * 1.10)


def legend(fig, handles, labels, points, ncol):
    fig.legend(handles, labels, ncol=ncol, fontsize=points * 1.05, **LEGEND)


def save(fig, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")
    return path


def clean_stale():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for filename in STALE:
        path = FIGURES / filename
        if path.exists():
            path.unlink()


def add_grid_colorbar(fig, axes, image, points):
    top = axes[0, -1].get_position()
    bottom = axes[-1, -1].get_position()
    cax = fig.add_axes([
        top.x1 + 0.020,
        bottom.y0,
        0.014,
        top.y1 - bottom.y0,
    ])
    bar = fig.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(
        labelsize=points * 0.85,
        length=0,
        labelcolor="black",
    )
    return bar


def add_single_colorbar(fig, ax, image, points):
    cax = make_axes_locatable(ax).append_axes(
        "right",
        size="3.5%",
        pad=0.18,
    )
    bar = fig.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(
        labelsize=points * 0.85,
        length=0,
        labelcolor="black",
    )
    return bar


def readable_on(rgba):
    rgb = np.asarray(rgba[:3])
    luminance = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
    return "white" if luminance < 0.48 else "black"


def nice_ceiling(maximum):
    if not np.isfinite(maximum) or maximum <= 0:
        return 1.0
    if maximum <= 1:
        return max(0.5, np.ceil(maximum * 10) / 10)
    if maximum <= 3:
        return np.ceil(maximum * 2) / 2
    if maximum <= 10:
        return np.ceil(maximum)
    if maximum <= 25:
        return np.ceil(maximum / 2) * 2
    if maximum <= 60:
        return np.ceil(maximum / 5) * 5
    return 100.0


def effect_note(ax, x, y, text, colour, points, dy=10):
    note = ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=points * 0.68,
        fontweight="bold",
        color=colour,
        zorder=6,
        annotation_clip=False,
    )
    note.set_path_effects([
        pe.withStroke(linewidth=1.6, foreground="white"),
        pe.Normal(),
    ])


# ---------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------

def load_register():
    if not REGISTER.exists():
        raise SystemExit(
            f"{REGISTER.relative_to(ROOT)} is missing. "
            "Run notebooks/15_safety.ipynb first."
        )
    raw = pd.read_csv(REGISTER)
    return analysis.adjust(raw) if hasattr(analysis, "adjust") else raw


def by_scenario(frame, measure, conditions=None):
    return analysis.by_scenario(frame, measure, conditions=conditions)


def rate_by_model(frame, measure, conditions=None):
    return analysis.rate_by_model(frame, measure, conditions=conditions).reindex(analysis.ORDER)


def available_fields(frame, candidates):
    return [(measure, label) for measure, label in candidates if measure in frame.columns]


def age_table(frame, intervals=False):
    rows = []
    for age in AGES:
        values = by_scenario(frame, "refusal", [AGE_CONDITION[age]])
        if values.empty:
            rows.append((age, np.nan, np.nan, np.nan))
            continue

        point = float(values.mean()) * 100
        if intervals:
            _, low, high = analysis.bootstrap_paired(values)
            low, high = 100 * low, 100 * high
        else:
            low = high = np.nan
        rows.append((age, point, low, high))

    return pd.DataFrame(rows, columns=["age", "point", "low", "high"]).set_index("age")


def register_rows(register, contrast):
    rows = analysis.ORDER[::-1] + [analysis.MACRO]
    part = register[register["contrast"].eq(contrast)].copy()
    part["order"] = pd.Categorical(part["model"], categories=rows, ordered=True)
    return part.sort_values("order")


def groups_for(kind, frame):
    if kind == "age":
        return [
            (f"Age {age}", frame[frame["condition"].eq(AGE_CONDITION[age])])
            for age in AGES
        ]
    if kind == "labels":
        return [
            (label, frame[frame["condition"].isin(conditions)])
            for label, conditions in LABEL_GROUPS
        ]
    if kind == "scenarios":
        return [
            (scenario, frame[frame["scenario_type"].eq(scenario)])
            for scenario in SCENARIOS
        ]
    raise ValueError(kind)


# ---------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------

def draw_heatmap(ax, table, points, vmax, annotate=True):
    map_panel(ax, None, points)
    blues = colormaps["Blues"]
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelsize=points * 0.80)
    ax.tick_params(axis="y", labelsize=points * 0.80)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        for row, col in np.ndindex(values.shape):
            value = values[row, col]
            if np.isfinite(value):
                rgba = blues(value / vmax if vmax else 0)
                ax.text(
                    col, row, f"{value:.1f}",
                    ha="center", va="center",
                    fontsize=points * 0.70,
                    fontweight="bold",
                    color=readable_on(rgba),
                )
    return image


# ---------------------------------------------------------------------
# Main 1: rates by scenario
# ---------------------------------------------------------------------

def scenario_rate_table(part, fields):
    return pd.DataFrame(
        {
            SCENARIO_AXIS[scenario]: [
                float(
                    by_scenario(
                        part[part["scenario_type"].eq(scenario)],
                        measure,
                    ).mean()
                ) * 100
                for measure, _ in fields
            ]
            for scenario in SCENARIOS
        },
        index=[label for _, label in fields],
    )


def draw_rates(returned, display):
    """Scenario-specific safety rates, one panel per model."""
    points = styled(display, 9.6, label_points=9.0)
    fields = available_fields(returned, RATE_FIELDS)

    tables = {
        model: scenario_rate_table(
            returned[returned["label"].eq(model)],
            fields,
        )
        for model in analysis.ORDER
    }
    vmax = nice_ceiling(
        max(
            np.nanmax(table.to_numpy(dtype=float))
            for table in tables.values()
        )
    )

    fig, axes = grid(figsize=(13.0, 8.0))
    fig.subplots_adjust(
        hspace=0.40,
        wspace=0.10,
        right=0.88,
        top=0.90,
        bottom=0.13,
    )

    image = None
    for index, (ax, model) in enumerate(
        zip(axes.flat, analysis.ORDER)
    ):
        table = tables[model]
        values = table.to_numpy(dtype=float)

        image = ax.imshow(
            values,
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )

        ax.set_xticks(
            range(len(table.columns)),
            table.columns if index // 3 == 1 else [],
        )
        ax.set_yticks(
            range(len(table.index)),
            table.index if index % 3 == 0 else [],
        )
        map_panel(ax, model, points)
        ax.tick_params(length=0)
        ax.tick_params(axis="x", labelsize=points * 0.80)
        ax.tick_params(axis="y", labelsize=points * 0.80)

        for spine in ax.spines.values():
            spine.set_visible(False)

        blues = colormaps["Blues"]
        for row, col in np.ndindex(values.shape):
            value = values[row, col]
            if not np.isfinite(value):
                continue
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=points * 0.62,
                fontweight="bold",
                color=readable_on(
                    blues(value / vmax if vmax else 0)
                ),
            )

    add_grid_colorbar(fig, axes, image, points)
    return save(fig, FIGURESPEC["rates"][1])


# ---------------------------------------------------------------------
# Main 2: outcomes by scenario and model
# ---------------------------------------------------------------------

def draw_outcome_pair(returned, display, cells, filename):
    """Draw one wide 1x2 outcome figure for a pair of outcome cells."""
    points = styled(display, width_inches=11.0, label_points=9.6)

    # Same scenario palette as every other figure.
    colour = SCENARIO_COLOUR

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.3, 8.9),
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.18,
        right=0.985,
        top=0.86,
        bottom=0.225,
        wspace=0.12,
    )

    # More vertical breathing room between model groups.
    group_gap = 2.02
    y = np.arange(len(analysis.ORDER))[::-1] * group_gap

    # Four clearly separated bars per model.
    height = 0.14
    offsets = np.array([0.45, 0.15, -0.15, -0.45])

    # Small-value panels benefit from a light minimum text offset.
    tiny_columns = [0.20, 0.42, 0.64, 0.86]

    for index, (ax, cell) in enumerate(zip(axes, cells)):
        panel(ax, cell, points)
        all_values = []

        for s_idx, (offset, scenario) in enumerate(zip(offsets, SCENARIOS)):
            part = returned[returned["scenario_type"].eq(scenario)]
            values = rate_by_model(
                part,
                CELL_COLUMNS[cell],
            ).to_numpy(dtype=float)
            all_values.extend(values.tolist())

            bars = ax.barh(
                y + offset,
                values,
                height=height,
                facecolor=mpl.colors.to_rgba(colour[scenario], 0.28),
                edgecolor=colour[scenario],
                linewidth=1.10,
                label=scenario if index == 0 else None,
                zorder=3,
            )

            maximum = max(values) if len(values) else 1.0

            for bar, value in zip(bars, values):
                if not np.isfinite(value) or value <= 0.05:
                    continue

                y_text = bar.get_y() + bar.get_height() / 2

                if cell == "Weak Refusal":
                    # Place numbers close to each bar end rather than in distant columns.
                    x_text = value + 0.07
                    if value < 0.12:
                        x_text = max(x_text, 0.12)
                elif cell == "Total Compliance" and value >= 95:
                    # Keep near-100 labels inside the panel and separated from the border.
                    x_text = min(value + 0.45, 101.0)
                elif value < 1.5:
                    # Tiny values in the left panel still need a little room from the axis.
                    x_text = max(value + 0.08, tiny_columns[s_idx])
                else:
                    x_text = value + max(0.10, maximum * 0.016)

                label_size = points * (0.62 if cell == "Weak Refusal" else 0.68)

                note = ax.text(
                    x_text,
                    y_text,
                    f"{value:.1f}",
                    ha="left",
                    va="center",
                    fontsize=label_size,
                    color="black",
                    zorder=5,
                    clip_on=False,
                )
                note.set_path_effects([
                    pe.withStroke(linewidth=1.4, foreground="white"),
                    pe.Normal(),
                ])

        ax.set_yticks(y, [MODEL_AXIS[m] for m in analysis.ORDER])
        ax.tick_params(axis="x", labelsize=points * 0.90)
        ax.tick_params(axis="y", labelsize=points * 0.90)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

        maximum = max(all_values) if all_values else 1.0
        if cell == "Weak Refusal":
            ax.set_xlim(0, max(4.4, nice_ceiling(maximum * 1.55)))
            ax.set_xticks([0, 1, 2, 3, 4])
        elif cell == "Minimal Compliance":
            ax.set_xlim(0, nice_ceiling(maximum * 1.45))
        elif cell == "Strong Refusal":
            ax.set_xlim(0, nice_ceiling(maximum * 1.18))
        else:  # Total Compliance
            ax.set_xlim(0, 104)
            ax.set_xticks([0, 20, 40, 60, 80, 100])

        ax.set_ylim(y.min() - 1.00, y.max() + 1.00)

        if index == 1:
            ax.tick_params(labelleft=False)

    # Centre the shared x-axis label over the actual two-panel plotting area,
    # not over the full page. The large left margin for model names means
    # figure coordinate x=0.5 is visibly too far left.
    fig.canvas.draw()
    left_edge = axes[0].get_position().x0
    right_edge = axes[1].get_position().x1
    panels_centre = (left_edge + right_edge) / 2.0

    fig.text(
        panels_centre,
        0.125,
        "Rate (%)",
        ha="center",
        va="center",
        color="black",
        fontsize=points * 1.02,
    )

    handles = [
        Patch(
            facecolor=mpl.colors.to_rgba(colour[scenario], 0.28),
            edgecolor=colour[scenario],
            linewidth=1.15,
            label=scenario,
        )
        for scenario in SCENARIOS
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        ncol=4,
        frameon=False,
        bbox_to_anchor=(panels_centre, 0.02),
        loc="lower center",
        fontsize=points * 0.95,
        handlelength=1.9,
        handletextpad=0.6,
        columnspacing=1.35,
    )

    return save(fig, filename)


def draw_outcomes_refusal(returned, display):
    return draw_outcome_pair(
        returned,
        display,
        ["Strong Refusal", "Weak Refusal"],
        FIGURESPEC["outcomes_refusal"][1],
    )


def draw_outcomes_compliance(returned, display):
    return draw_outcome_pair(
        returned,
        display,
        ["Minimal Compliance", "Total Compliance"],
        FIGURESPEC["outcomes_compliance"][1],
    )


# ---------------------------------------------------------------------
# Main 3: safeguards within Strong Refusal vs Total Compliance
# ---------------------------------------------------------------------

def conditional_field_rates(part, model, fields):
    model_part = part[part["label"].eq(model)]
    return pd.Series(
        {
            label: float(by_scenario(model_part, measure).mean()) * 100
            for measure, label in fields
        },
        dtype=float,
    )


def draw_safeguards(returned, display):
    """Safeguard fields within Strong Refusal and Total Compliance."""
    points = styled(display)
    fields = available_fields(returned, SAFEGUARD_MAIN)
    labels = [label for _, label in fields]
    x = np.arange(len(labels))
    width = 0.34

    strong = returned[
        returned[CELL_COLUMNS["Strong Refusal"]].eq(1.0)
    ]
    total = returned[
        returned[CELL_COLUMNS["Total Compliance"]].eq(1.0)
    ]

    fig, axes = grid(
        figsize=(13.5, 7.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for index, (ax, model) in enumerate(
        zip(axes.flat, analysis.ORDER)
    ):
        a = conditional_field_rates(
            strong,
            model,
            fields,
        ).reindex(labels)
        b = conditional_field_rates(
            total,
            model,
            fields,
        ).reindex(labels)

        ax.bar(
            x - width / 2,
            a.values,
            width=width,
            facecolor=analysis.COLOUR[model],
            edgecolor=analysis.COLOUR[model],
            linewidth=1.1,
            alpha=0.40,
            label="Strong Refusal" if index == 0 else None,
            zorder=3,
        )

        outline = ax.bar(
            x + width / 2,
            b.values,
            width=width,
            facecolor="none",
            edgecolor=analysis.COLOUR[model],
            linewidth=1.5,
            label="Total Compliance" if index == 0 else None,
            zorder=3,
        )
        for patch in outline:
            patch.set_linestyle("--")

        panel(ax, model, points)
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.55, len(labels) - 0.45)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))

        if index < 3:
            ax.tick_params(labelbottom=False)
        else:
            ax.tick_params(axis="x", labelsize=points * 0.78)
            ax.tick_params(axis="x", labelsize=points * 0.76)
            plt.setp(
                ax.get_xticklabels(),
                rotation=0,
                ha="center",
            )

    outer_labels(
        fig,
        axes,
        "",
        "Conditional Rate (%)",
        points,
    )
    handles, names = axes.flat[0].get_legend_handles_labels()
    legend(fig, handles, names, points, 2)
    return save(fig, FIGURESPEC["safeguards"][1])


# ---------------------------------------------------------------------
# Main 4: exact-age trajectory
# ---------------------------------------------------------------------

def age_bands(ax):
    ax.axvspan(6.4, 17.5, color=MINOR_BAND, zorder=0)
    ax.axvspan(17.5, 21.6, color=ADULT_BAND, zorder=0)
    ax.axvline(
        analysis.THRESHOLD,
        color=analysis.MUTED,
        linestyle=":",
        linewidth=1.0,
        zorder=1,
    )


def value_note(ax, x, y, text, colour, points, dx, dy, va):
    note = ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=points * 0.78,
        fontweight="bold",
        color=colour,
        zorder=6,
    )
    note.set_path_effects([
        pe.withStroke(linewidth=2.0, foreground="white"),
        pe.Normal(),
    ])


def draw_trajectory(focus, display):
    points = styled(display)
    fig, axes = grid(
        figsize=(13.5, 7.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, model in zip(axes.flat, analysis.ORDER):
        part = focus[focus["label"].eq(model)]
        estimates = age_table(part, intervals=True)

        age_bands(ax)
        errors = np.vstack([
            estimates["point"] - estimates["low"],
            estimates["high"] - estimates["point"],
        ])

        ax.errorbar(
            AGES,
            estimates["point"],
            yerr=errors,
            marker=analysis.MARKER[model],
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=analysis.COLOUR[model],
            markeredgewidth=1.1,
            linewidth=1.9,
            elinewidth=0.75,
            capsize=1.8,
            color=analysis.COLOUR[model],
            zorder=3,
        )

        panel(ax, model, points)
        ax.set_xticks(AGES)
        ax.set_xlim(6.4, 21.6)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))

        base_specs = {
            7: ((0, 11), "bottom"),
            11: ((0, -14), "top"),
            13: ((0, 11), "bottom"),
            15: ((0, -14), "top"),
            17: ((-12, 12), "bottom"),
            18: ((12, -14), "top"),
            21: ((0, -14), "top"),
        }
        for age, (offset, va) in base_specs.items():
            value = estimates.at[age, "point"]
            if not np.isfinite(value):
                continue

            dx, dy = offset
            if value > 94:
                dy, va = -14, "top"

            value_note(
                ax,
                age,
                value,
                f"{value:.1f}",
                analysis.COLOUR[model],
                points,
                dx,
                dy,
                va,
            )

    outer_labels(
        fig,
        axes,
        "Age",
        "Refusal Rate (%)",
        points,
    )
    legend(
        fig,
        [
            Patch(facecolor=MINOR_BAND),
            Patch(facecolor=ADULT_BAND),
            Line2D(
                [0],
                [0],
                color=analysis.MUTED,
                linestyle=":",
                linewidth=1.5,
            ),
            Line2D(
                [0],
                [0],
                color=analysis.MUTED,
                linewidth=1.2,
            ),
        ],
        ["Minor Ages", "Adult Ages", "Age 18 Threshold", "95% CI"],
        points,
        4,
    )
    return save(fig, FIGURESPEC["trajectory"][1])


# ---------------------------------------------------------------------
# Main 5: primary contrasts
# ---------------------------------------------------------------------

def forest(ax, table, points):
    y = np.arange(len(table))

    for position, (_, row) in enumerate(table.iterrows()):
        model = row["model"]
        colour = analysis.INK if model == analysis.MACRO else analysis.COLOUR[model]
        marker = "D" if model == analysis.MACRO else analysis.MARKER[model]

        ax.plot(
            [row["low"], row["high"]],
            [position, position],
            color=colour,
            linewidth=analysis.LINEWIDTH,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            row["effect"],
            position,
            marker=marker,
            color=colour,
            markersize=analysis.MARKERSIZE + 0.5,
            linestyle="none",
            zorder=3,
        )
        effect_note(
            ax,
            row["effect"],
            position,
            f'{row["effect"]:.1f} [{row["low"]:.1f}, {row["high"]:.1f}]',
            colour,
            points,
        )

    ax.axvline(
        0,
        color=analysis.MUTED,
        linewidth=1.0,
        linestyle="--",
        alpha=0.70,
        zorder=1,
    )
    models = table["model"].tolist()
    if analysis.MACRO in models:
        split = models.index(analysis.MACRO) + 0.5
        ax.axhline(split, color="0.80", linewidth=0.8, zorder=1)

    ax.set_yticks(y, [MODEL_AXIS[model] for model in table["model"]])
    panel(ax, None, points)


def draw_primary(register, display):
    points = styled(display)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=WIDE_SIZE,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.87, bottom=0.16, wspace=0.03)

    extent = register[register["contrast"].isin(PRIMARY)]
    low = min(0.0, float(extent["low"].min()))
    high = max(0.0, float(extent["high"].max()))
    span = max(high - low, 1.0)

    for index, (ax, contrast) in enumerate(zip(axes, PRIMARY)):
        part = register_rows(register, contrast)
        forest(ax, part, points)
        panel(ax, PRIMARY_LABEL[contrast], points)
        ax.set_xlim(low - 0.08 * span, high + 0.08 * span)

        if index > 0:
            ax.tick_params(labelleft=False)

    fig.supxlabel(
        "Difference in Refusal Rate (pp)",
        color="black",
        fontsize=points * 1.08,
    )
    return save(fig, FIGURESPEC["primary"][1])


# ---------------------------------------------------------------------
# Supplement 1: age trajectories for all scenario types
# ---------------------------------------------------------------------

def dynamic_ylim(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0, 1

    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 1.0)

    if high <= 10:
        return 0, high + max(1.0, 0.18 * span)

    return max(0, low - 0.14 * span), min(100, high + 0.14 * span)


def draw_controls(returned, display):
    """
    Four scenario panels, with separate y-scales so the small Benign/Rights
    movements remain visible without compressing Age Restricted/Harmful.
    """
    points = styled(display)

    # Keep the original compact 2x2 geometry, but reserve explicit space
    # between rows and below the panels for the shared x-label and legend.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 8.15),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        top=0.955,
        bottom=0.205,
        wspace=0.045,
        hspace=0.145,
    )

    for index, (ax, scenario) in enumerate(zip(axes.flat, SCENARIOS)):
        all_values = []

        for model in analysis.ORDER:
            part = returned[
                returned["label"].eq(model)
                & returned["scenario_type"].eq(scenario)
            ]
            estimates = age_table(part, intervals=False)
            all_values.extend(estimates["point"].dropna().tolist())

            ax.plot(
                AGES,
                estimates["point"],
                marker=analysis.MARKER[model],
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=0.9,
                linewidth=1.45,
                color=analysis.COLOUR[model],
                label=model if index == 0 else None,
            )

        panel(ax, scenario, points)
        ax.set_xticks(AGES)
        ax.set_xlim(6.4, 21.6)

        lower, upper = dynamic_ylim(all_values)
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        if scenario == analysis.FOCUS:
            ax.axvline(
                analysis.THRESHOLD,
                color=analysis.MUTED,
                linestyle=":",
                linewidth=1.0,
                zorder=1,
            )

    # Only the bottom row needs x tick labels; only the left column needs y labels.
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)

    # Use fixed figure coordinates rather than supxlabel/supylabel so
    # bbox_inches="tight" cannot pull the labels into the panels.
    fig.text(
        0.535,
        0.132,
        "Age",
        ha="center",
        va="center",
        fontsize=points * 1.08,
        color="black",
    )
    fig.text(
        0.022,
        0.535,
        "Refusal Rate (%)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=points * 1.12,
        color="black",
    )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.535, 0.018),
        fontsize=points * 1.00,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.45,
    )

    return save(fig, FIGURESPEC["controls"][1])


# ---------------------------------------------------------------------
# Supplement 2: signal ladder
# ---------------------------------------------------------------------

def draw_cues(returned, display):
    """Refusal across the five age-signal levels, one model per panel."""
    points = styled(display)
    x = np.arange(len(LABEL_GROUPS))

    fig, axes = grid(
        figsize=(13.5, 7.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for index, (ax, model) in enumerate(
        zip(axes.flat, analysis.ORDER)
    ):
        part = returned[
            returned["label"].eq(model)
            & returned["scenario_type"].eq(analysis.FOCUS)
        ]

        values = pd.Series(
            [
                float(
                    by_scenario(
                        part,
                        "refusal",
                        conditions,
                    ).mean()
                ) * 100
                for _, conditions in LABEL_GROUPS
            ],
            index=[label for label, _ in LABEL_GROUPS],
            dtype=float,
        )

        ax.plot(
            x,
            values.values,
            "-",
            marker=analysis.MARKER[model],
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=analysis.COLOUR[model],
            markeredgewidth=1.1,
            linewidth=1.9,
            color=analysis.COLOUR[model],
            zorder=3,
        )

        panel(ax, model, points)
        ax.set_xticks(
            x,
            [SIGNAL_AXIS[label] for label in values.index],
        )
        ax.set_xlim(-0.45, 4.45)
        ax.set_ylim(0, 85)
        ax.yaxis.set_major_locator(MultipleLocator(20))

        # Label the explicit adult baseline and the two minor conditions.
        for xpos in (2, 3, 4):
            value = values.iloc[xpos]
            if not np.isfinite(value):
                continue
            note = ax.annotate(
                f"{value:.1f}",
                xy=(xpos, value),
                xytext=(0, 10 if value < 74 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if value < 74 else "top",
                fontsize=points * 0.72,
                fontweight="bold",
                color=analysis.COLOUR[model],
                zorder=6,
            )
            note.set_path_effects([
                pe.withStroke(
                    linewidth=2.0,
                    foreground="white",
                ),
                pe.Normal(),
            ])

        if index < 3:
            ax.tick_params(labelbottom=False)
        else:
            plt.setp(
                ax.get_xticklabels(),
                rotation=0,
                ha="center",
            )

    outer_labels(
        fig,
        axes,
        "",
        "Refusal Rate (%)",
        points,
    )
    return save(fig, FIGURESPEC["cues"][1])


# ---------------------------------------------------------------------
# Supplement 3-5: safeguard field maps
# ---------------------------------------------------------------------

def draw_field_slice(returned, display, kind):
    points = styled(
        display,
        width_inches=12.0,
        label_points=8.1,
    )
    fields = available_fields(returned, SAFEGUARD_ALL)
    groups = groups_for(kind, returned)

    table = pd.DataFrame(
        {
            label: {
                group_label: float(
                    analysis.rate_by_model(
                        group_frame,
                        measure,
                    ).mean()
                )
                for group_label, group_frame in groups
            }
            for measure, label in fields
        }
    )

    values = table.to_numpy(dtype=float)
    vmax = nice_ceiling(np.nanmax(values))

    height = {
        "age": 6.2,
        "labels": 5.0,
        "scenarios": 4.6,
    }[kind]

    fig, ax = plt.subplots(
        figsize=(12.0, height),
        constrained_layout=True,
    )

    map_panel(ax, None, points)
    image = ax.imshow(
        values,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_xticks(
        range(len(table.columns)),
        table.columns,
    )
    ax.set_yticks(
        range(len(table.index)),
        table.index,
    )
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelsize=points * 0.80)
    ax.tick_params(axis="y", labelsize=points * 0.80)

    # Explicitly keep every field label horizontal.
    plt.setp(
        ax.get_xticklabels(),
        rotation=0,
        ha="center",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    blues = colormaps["Blues"]
    for row, col in np.ndindex(values.shape):
        value = values[row, col]
        if not np.isfinite(value):
            continue
        ax.text(
            col,
            row,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=points * 0.56,
            fontweight="bold",
            color=readable_on(
                blues(value / vmax if vmax else 0)
            ),
        )

    add_single_colorbar(
        fig,
        ax,
        image,
        points,
    )
    return save(fig, FIGURESPEC[f"fields_{kind}"][1])


# ---------------------------------------------------------------------
# Supplement 6: directional failures
# ---------------------------------------------------------------------

def failure_series(returned):
    harmful = returned[returned["scenario_type"].eq("Harmful")]
    age_restricted = returned[returned["scenario_type"].eq("Age Restricted")]
    benign = returned[returned["scenario_type"].eq("Benign")]
    rights = returned[returned["scenario_type"].eq("Rights")]

    return {
        "Harmful Delivery": rate_by_model(harmful, "delivered"),
        "Age Restricted Delivery": rate_by_model(
            age_restricted,
            "delivered",
            MINOR_AGE_CONDITIONS,
        ),
        "Age Restricted Refusal": rate_by_model(
            age_restricted,
            "refusal",
            ADULT_AGE_CONDITIONS,
        ),
        "Benign Refusal": rate_by_model(
            benign,
            "refusal",
        ),
        "Rights Refusal": rate_by_model(
            rights,
            "refusal",
        ),
    }


def grouped_failure_panel(ax, series_items, points, title):
    """Grouped horizontal bars using the same layout grammar as safety_outcomes."""
    # Match safety_outcomes exactly for model spacing.
    group_gap = 2.02
    y = np.arange(len(analysis.ORDER))[::-1] * group_gap
    n = len(series_items)

    # Same narrow-bin style as safety_outcomes.
    height = 0.14

    # Preserve the requested visual order from top to bottom.
    if n == 2:
        offsets = np.array([0.15, -0.15])
    elif n == 3:
        offsets = np.array([0.30, 0.00, -0.30])
    else:
        offsets = np.linspace(0.45, -0.45, n)

    panel(ax, title, points)
    ax.set_yticks(
        y,
        [MODEL_AXIS[m] for m in analysis.ORDER],
    )
    ax.tick_params(axis="x", labelsize=points * 0.90)
    ax.tick_params(axis="y", labelsize=points * 0.90)
    all_arrays = [series.to_numpy(dtype=float) for _, series, _ in series_items]
    maximum = max(float(np.nanmax(arr)) for arr in all_arrays) if all_arrays else 1.0
    ax.set_xlim(0, nice_ceiling(maximum * 1.18))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylim(y.min() - 1.00, y.max() + 1.00)

    legend_handles = []

    for offset, (label, series, colour) in zip(offsets, series_items):
        values = series.to_numpy(dtype=float)
        bars = ax.barh(
            y + offset,
            values,
            height=height,
            facecolor=mpl.colors.to_rgba(colour, 0.28),
            edgecolor=colour,
            linewidth=1.10,
            label=label,
            zorder=3,
        )

        legend_handles.append(
            Patch(
                facecolor=mpl.colors.to_rgba(colour, 0.28),
                edgecolor=colour,
                linewidth=1.10,
                label=label,
            )
        )

        for bar in bars:
            value = bar.get_width()
            if not np.isfinite(value):
                continue

            x_text = value + max(0.08, maximum * 0.014)
            y_text = bar.get_y() + bar.get_height() / 2

            note = ax.text(
                x_text,
                y_text,
                f"{value:.1f}",
                ha="left",
                va="center",
                fontsize=points * 0.68,
                color="black",
                zorder=5,
                clip_on=False,
            )
            note.set_path_effects([
                pe.withStroke(linewidth=1.4, foreground="white"),
                pe.Normal(),
            ])

    return legend_handles


def draw_failures(returned, display):
    """Directional failures using the same 1x2 layout as safety_outcomes."""
    points = styled(
        display,
        width_inches=11.0,
        label_points=9.6,
    )
    series = failure_series(returned)

    # Match safety_outcomes canvas, margins, and panel spacing.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.3, 8.9),
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.18,
        right=0.985,
        top=0.86,
        bottom=0.225,
        wspace=0.12,
    )

    # Global visual order:
    # Benign -> Rights -> Age Restricted -> Harmful.
    # Each panel shows only the categories relevant to that failure direction.
    grouped_failure_panel(
        axes[0],
        [
            ("Age Restricted", series["Age Restricted Delivery"], SCENARIO_COLOUR["Age Restricted"]),
            ("Harmful", series["Harmful Delivery"], SCENARIO_COLOUR["Harmful"]),
        ],
        points,
        "Over-Permissive",
    )
    grouped_failure_panel(
        axes[1],
        [
            ("Benign", series["Benign Refusal"], SCENARIO_COLOUR["Benign"]),
            ("Rights", series["Rights Refusal"], SCENARIO_COLOUR["Rights"]),
            ("Age Restricted", series["Age Restricted Refusal"], SCENARIO_COLOUR["Age Restricted"]),
        ],
        points,
        "Over-Restrictive",
    )

    axes[1].tick_params(labelleft=False)

    # Same centering logic as safety_outcomes.
    fig.canvas.draw()
    left_edge = axes[0].get_position().x0
    right_edge = axes[1].get_position().x1
    panels_centre = (left_edge + right_edge) / 2.0

    fig.text(
        panels_centre,
        0.125,
        "Rate (%)",
        ha="center",
        va="center",
        color="black",
        fontsize=points * 1.02,
    )

    legend_order = [
        (scenario, SCENARIO_COLOUR[scenario])
        for scenario in SCENARIOS
    ]
    legend_handles = [
        Patch(
            facecolor=mpl.colors.to_rgba(colour, 0.28),
            edgecolor=colour,
            linewidth=1.15,
            label=label,
        )
        for label, colour in legend_order
    ]

    fig.legend(
        legend_handles,
        [label for label, _ in legend_order],
        ncol=4,
        frameon=False,
        bbox_to_anchor=(panels_centre, 0.02),
        loc="lower center",
        fontsize=points * 0.95,
        handlelength=1.9,
        handletextpad=0.6,
        columnspacing=1.35,
    )

    return save(fig, FIGURESPEC["failures"][1])


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------



def draw_age_steps(focus, display):
    """Heatmap of age-to-age changes in refusal rate within age-restricted scenarios."""
    points = styled(display, width_inches=9.8, label_points=9.4)
    transitions = list(zip(AGES[:-1], AGES[1:]))
    labels = [f"{a}→{b}" for a, b in transitions]
    rows = []

    for model in [analysis.MACRO] + analysis.ORDER:
        part = focus if model == analysis.MACRO else focus[focus["label"].eq(model)]
        estimates = age_table(part, intervals=False)["point"]
        diffs = [float(estimates.loc[b] - estimates.loc[a]) for a, b in transitions]
        rows.append(pd.Series(diffs, index=labels, name="Macro-Average" if model == analysis.MACRO else model))

    table = pd.DataFrame(rows)
    values = table.to_numpy(dtype=float)
    magnitude = np.abs(values)
    vmax = max(5.0, float(np.nanmax(magnitude)))
    y_labels = [MODEL_AXIS.get(idx, idx).replace("Macro-\nAverage", "Macro-Average") for idx in table.index]

    fig, ax = plt.subplots(figsize=(10.7, 4.9), constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.93, top=0.87, bottom=0.18)
    map_panel(ax, None, points)
    image = ax.imshow(magnitude, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(table.index)), y_labels)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelsize=points * 0.84)
    ax.tick_params(axis="y", labelsize=points * 0.84)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for r, c in np.ndindex(values.shape):
        value = values[r, c]
        display_value = abs(value)
        text_colour = "white" if display_value >= 0.62 * vmax else "black"
        ax.text(
            c,
            r,
            f"{value:+.1f}",
            ha="center",
            va="center",
            fontsize=points * 0.72,
            fontweight="bold",
            color=text_colour,
        )

    map_panel(ax, "Age-to-Age Changes in Refusal Rate", points)
    ax.set_xlabel("Adjacent Exact-Age Transitions")
    bar = add_single_colorbar(fig, ax, image, points)
    bar.set_label("Absolute Change in Refusal Rate (pp)", fontsize=points * 0.86)
    return save(fig, FIGURESPEC["age_steps"][1])


def draw_tradeoff(returned, display):
    """Model-level trade-off between over-permissive and over-restrictive failure."""
    points = styled(display, width_inches=8.8, label_points=9.4)

    harmful = rate_by_model(
        returned[returned["scenario_type"].eq("Harmful")],
        "delivered",
    )
    restrictive = rate_by_model(
        returned[returned["scenario_type"].isin(["Benign", "Rights"])],
        "refusal",
    )

    # Name offsets: move GPT label to the LEFT, keep others to the right.
    name_offsets = {
        "GPT-5.6 Luna": (-10, -5),
        "Claude Haiku 4.5": (10, -7),
        "Gemini 3.5 Flash Lite": (10, -4),
        "DeepSeek-V4 Flash": (10, -7),
        "Mistral Small 4": (10, -5),
        "Gemma 4 31B": (10, -5),
    }

    # Alignment for labels.
    name_align = {
        "GPT-5.6 Luna": "right",
        "Claude Haiku 4.5": "left",
        "Gemini 3.5 Flash Lite": "left",
        "DeepSeek-V4 Flash": "left",
        "Mistral Small 4": "left",
        "Gemma 4 31B": "left",
    }

    # Value labels above points.
    value_offsets = {
        "GPT-5.6 Luna": (0, 14),
        "Claude Haiku 4.5": (0, 14),
        "Gemini 3.5 Flash Lite": (0, 14),
        "DeepSeek-V4 Flash": (0, 14),
        "Mistral Small 4": (0, 14),
        "Gemma 4 31B": (0, 14),
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.9), constrained_layout=False)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.88, bottom=0.16)

    panel(ax, "Safety Failure Trade-Off", points)

    x_mean = float(harmful.mean())
    y_mean = float(restrictive.mean())

    ax.axvline(
        x_mean,
        color=MUTED,
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )
    ax.axhline(
        y_mean,
        color=MUTED,
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    # More right padding so GPT and the bottom-right corner label both fit cleanly.
    x_min = float(harmful.min()) - 0.9
    x_max = float(harmful.max()) + 2.4

    # Small negative lower bound so GPT is fully visible.
    y_min = -0.16
    y_max = float(restrictive.max()) + 0.22

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    quadrant_style = dict(
        fontsize=points * 0.52,
        color=MUTED,
    )

    ax.text(
        0.02, 0.965, "Over-Restrictive",
        transform=ax.transAxes,
        ha="left", va="top",
        **quadrant_style,
    )
    ax.text(
        0.98, 0.965, "High on Both",
        transform=ax.transAxes,
        ha="right", va="top",
        **quadrant_style,
    )
    ax.text(
        0.02, 0.055, "Low on Both",
        transform=ax.transAxes,
        ha="left", va="bottom",
        **quadrant_style,
    )
    ax.text(
        0.98, 0.055, "Over-Permissive",
        transform=ax.transAxes,
        ha="right", va="bottom",
        **quadrant_style,
    )

    for model in analysis.ORDER:
        x = float(harmful.loc[model])
        y = float(restrictive.loc[model])
        colour = analysis.COLOUR[model]

        ax.scatter(
            x,
            y,
            s=88,
            marker=analysis.MARKER[model],
            color=colour,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        vdx, vdy = value_offsets.get(model, (0, 14))
        ax.annotate(
            f"H: {x:.1f}\nR: {y:.1f}",
            xy=(x, y),
            xytext=(vdx, vdy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=points * 0.52,
            color=colour,
            linespacing=1.0,
            zorder=4,
            clip_on=False,
        )

        ndx, ndy = name_offsets.get(model, (10, -5))
        ax.annotate(
            model,
            xy=(x, y),
            xytext=(ndx, ndy),
            textcoords="offset points",
            fontsize=points * 0.60,
            color=colour,
            ha=name_align.get(model, "left"),
            va="center",
            fontweight="semibold",
            zorder=4,
            clip_on=False,
        )

    ax.text(
        x_mean + 0.06,
        y_max - 0.02,
        "Mean",
        fontsize=points * 0.52,
        color=MUTED,
        ha="left",
        va="top",
    )
    ax.text(
        x_max - 0.02,
        y_mean + 0.01,
        "Mean",
        fontsize=points * 0.52,
        color=MUTED,
        ha="right",
        va="bottom",
    )

    ax.set_xlabel("Harmful Delivery (%)")
    ax.set_ylabel("Benign / Rights Refusal (%)")

    ax.tick_params(axis="x", labelsize=points * 0.88)
    ax.tick_params(axis="y", labelsize=points * 0.88)

    return save(fig, FIGURESPEC["tradeoff"][1])


def draw_safeguards_macro(returned, display):
    """Macro-average safeguard profile within strong refusals versus total compliance."""
    points = styled(display, width_inches=8.6, label_points=9.4)
    fields = available_fields(returned, SAFEGUARD_MAIN)
    labels = [label for _, label in fields]

    strong = returned[returned[CELL_COLUMNS["Strong Refusal"]].eq(1.0)]
    total = returned[returned[CELL_COLUMNS["Total Compliance"]].eq(1.0)]

    strong_rates = pd.Series(
        {label: float(by_scenario(strong, measure).mean()) * 100 for measure, label in fields}
    ).reindex(labels)
    total_rates = pd.Series(
        {label: float(by_scenario(total, measure).mean()) * 100 for measure, label in fields}
    ).reindex(labels)

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    panel(ax, "Conditional Safeguard Profiles", points)

    ax.hlines(y, strong_rates.values, total_rates.values, color="0.72", linewidth=1.2, zorder=1)
    ax.scatter(
        strong_rates.values, y,
        s=55,
        color="#2F6FAF",
        marker="o",
        label="Strong Refusal",
        zorder=3,
    )
    ax.scatter(
        total_rates.values, y,
        s=55,
        facecolors="white",
        edgecolors="#4FA3A5",
        linewidths=1.6,
        marker="o",
        label="Total Compliance",
        zorder=3,
    )

    for x, yy in zip(strong_rates.values, y):
        ax.annotate(f"{x:.1f}", (x, yy), xytext=(-6, 8), textcoords="offset points",
                    ha="right", fontsize=points * 0.62, color="#2F6FAF")
    for x, yy in zip(total_rates.values, y):
        ax.annotate(f"{x:.1f}", (x, yy), xytext=(6, -10), textcoords="offset points",
                    ha="left", fontsize=points * 0.62, color="#4FA3A5")

    ax.set_yticks(y, labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Conditional Rate (%)")
    ax.tick_params(axis="x", labelsize=points * 0.88)
    ax.tick_params(axis="y", labelsize=points * 0.88)
    ax.legend(frameon=False, loc="lower right", fontsize=points * 0.76)
    return save(fig, FIGURESPEC["safeguards_macro"][1])


def registry(returned, focus, register, display):
    return {
        "rates": (lambda: draw_rates(returned, display), "main"),
        "outcomes_refusal": (lambda: draw_outcomes_refusal(returned, display), "main"),
        "outcomes_compliance": (lambda: draw_outcomes_compliance(returned, display), "main"),
        "safeguards": (lambda: draw_safeguards(returned, display), "main"),
        "trajectory": (lambda: draw_trajectory(focus, display), "main"),
        "primary": (lambda: draw_primary(register, display), "main"),
        "controls": (lambda: draw_controls(returned, display), "supplement"),
        "cues": (lambda: draw_cues(returned, display), "supplement"),
        "fields_age": (lambda: draw_field_slice(returned, display, "age"), "supplement"),
        "fields_labels": (lambda: draw_field_slice(returned, display, "labels"), "supplement"),
        "fields_scenarios": (lambda: draw_field_slice(returned, display, "scenarios"), "supplement"),
        "failures": (lambda: draw_failures(returned, display), "supplement"),
        "age_steps": (lambda: draw_age_steps(focus, display), "supplement"),
        "tradeoff": (lambda: draw_tradeoff(returned, display), "supplement"),
        "safeguards_macro": (lambda: draw_safeguards_macro(returned, display), "supplement"),
    }


def print_manifest():
    print("Safety Figure Set")
    print("-" * 94)
    for key, (tier, filename, purpose) in FIGURESPEC.items():
        print(f"{key:<18}{tier:<12}{filename:<32}{purpose}")
    print("-" * 94)


def main(args):
    mpl.rcParams.update(analysis.STYLE)
    clean_stale()

    frame = analysis.load_corpus()
    returned = frame[frame["responded"]].copy()
    focus = returned[returned["scenario_type"].eq(analysis.FOCUS)].copy()
    register = load_register()

    fingerprint = frame.attrs["fingerprint"]
    print(
        f"{fingerprint['requests']:,} requests, "
        f"{fingerprint['blocked']:,} blocked, "
        f"{fingerprint['returned']:,} returned, "
        f"rubric {fingerprint['policy']}\n"
    )

    print_manifest()
    jobs = registry(returned, focus, register, args.display)

    print("\nGenerating")
    print("-" * 94)

    built = 0
    for name, (build, tier) in jobs.items():
        if args.only not in {"all", name}:
            continue
        if args.set not in {"both", tier}:
            continue

        print(f"{name:<18}[{tier}] {FIGURESPEC[name][2]}")
        build()
        built += 1

    print(
        f"\n{built} figure{'s' if built != 1 else ''} "
        f"written to {FIGURES.relative_to(ROOT)}/."
    )


def parser():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--display", type=float, default=1.0)
    cli.add_argument("--set", default="both", choices=("both", "main", "supplement"))
    cli.add_argument("--only", default="all", choices=("all", *FIGURESPEC.keys()))
    return cli


if __name__ == "__main__":
    main(parser().parse_args())
