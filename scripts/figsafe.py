"""Generate publication safety figures for Experiment 1.

Notes
-----
Create primary contrasts and age trajectories for the main paper,
with age-invariant controls for the supplement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


# Paths

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUTPUT = ROOT / "figures" / "safety"
sys.path.insert(0, str(SCRIPTS))

import analysis

REGISTER = analysis.MACHINE / "register_safety.csv"


# Study Order

MODELS = [
    "GPT-5.6 Luna",
    "Claude Haiku 4.5",
    "Gemini 3.5 Flash Lite",
    "DeepSeek-V4 Flash",
    "Mistral Small 4",
    "Gemma 4 31B",
]
MACRO = analysis.MACRO
MODEL_ORDER = MODELS + [MACRO]
MODEL_AXIS = {model: model for model in MODELS}
MODEL_AXIS[MACRO] = "Macro-Average"

AGE_MAP = {
    condition: int(condition.replace("age", ""))
    for condition in analysis.CONDITION_ORDER
    if condition.startswith("age")
}
AGE_CONDITIONS = sorted(AGE_MAP, key=AGE_MAP.get)
AGES = [AGE_MAP[condition] for condition in AGE_CONDITIONS]
AGE_CONDITION = {AGE_MAP[condition]: condition for condition in AGE_CONDITIONS}
MINOR_AGE_CONDITIONS = [
    condition for condition in AGE_CONDITIONS if AGE_MAP[condition] < 18
]
ADULT_AGE_CONDITIONS = [
    condition for condition in AGE_CONDITIONS if AGE_MAP[condition] >= 18
]

PRIMARY = [
    analysis.TRAJECTORY,
    analysis.THRESHOLD_CONTRAST,
    analysis.SIGNAL,
]
PRIMARY_LABEL = {
    analysis.TRAJECTORY: "Explicit Minor vs Explicit Adult",
    analysis.THRESHOLD_CONTRAST: "17 vs 18",
    analysis.SIGNAL: "Explicit Minor vs Implicit Minor",
}
CONTROL_SCENARIOS = ["Benign", "Rights", "Harmful"]

# Publication Style

TEXT_WIDTH = 7.15
GREEN = "#009E73"
PINK = "#CC79A7"
BLACK = "#111111"
GREY = "#6F6F6F"
LIGHT_GREY = "#D8D8D8"
GRID = "#C7C7C7"
EQUIVALENCE = "#F0F0F0"
MACRO_COLOUR = "#5E3C99"

MODEL_COLOUR = {
    "GPT-5.6 Luna": "#0072B2",
    "Claude Haiku 4.5": "#D55E00",
    "Gemini 3.5 Flash Lite": GREEN,
    "DeepSeek-V4 Flash": PINK,
    "Mistral Small 4": "#E69F00",
    "Gemma 4 31B": "#56B4E9",
    MACRO: MACRO_COLOUR,
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


def set_style() -> None:
    """Apply the shared publication style."""
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


def style_axis(ax, grid_axis: str = "both") -> None:
    """Style one plotting axis."""
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(0.8)
    ax.tick_params(length=2.8, width=0.7, direction="out")


def save_figure(fig, output: Path, name: str, png: bool = False) -> None:
    """Save a vector PDF and optionally a PNG."""
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{name}.pdf"
    fig.savefig(destination, bbox_inches="tight", pad_inches=0.03)
    if png:
        fig.savefig(
            output / f"{name}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)
    print(f"Figure: {destination.name}")


def legend_style(legend) -> None:
    """Apply the common light legend border."""
    legend.get_frame().set_linewidth(0.6)


def centered_xlabel(fig, axes, label: str, y: float) -> None:
    """Center an x-axis label beneath the actual panels."""
    axes = np.asarray(axes).ravel()
    fig.canvas.draw()
    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    fig.text((left + right) / 2, y, label, ha="center", va="center", fontsize=8.7)


# Analysis

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read classified replies and the frozen contrast register."""
    if not REGISTER.exists():
        raise FileNotFoundError(f"Missing {REGISTER}. Run the safety analysis first.")
    corpus = analysis.load_corpus()
    returned = corpus.loc[corpus["responded"]].copy()
    focus = returned.loc[returned["scenario_type"].eq(analysis.FOCUS)].copy()
    register = pd.read_csv(REGISTER)
    if hasattr(analysis, "adjust"):
        register = analysis.adjust(register)
    return returned, focus, register


def to_series(value) -> pd.Series:
    """Convert an estimate to a numerical scenario series."""
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("Expected one numerical measure")
        value = value.iloc[:, 0]
    return pd.Series(value, dtype=float)


def paired_difference(
    frame: pd.DataFrame, measure: str, first: list[str], second: list[str]
) -> pd.Series:
    """Match scenario-level values across two condition groups."""
    a = to_series(analysis.by_scenario(frame, measure, conditions=first))
    b = to_series(analysis.by_scenario(frame, measure, conditions=second))
    paired = pd.concat([a.rename("first"), b.rename("second")], axis=1).dropna()
    if paired.empty:
        raise ValueError(f"No complete scenario pairs for {measure}")
    return paired["first"] - paired["second"]


def model_effect(
    frame: pd.DataFrame, measure: str, first: list[str], second: list[str]
) -> tuple[float, float, float, pd.Series]:
    """Estimate one paired model contrast and its percentile interval."""
    difference = paired_difference(frame, measure, first, second)
    point, low, high = analysis.bootstrap_paired(difference)
    return point * 100, low * 100, high * 100, difference


def macro_effect(
    differences: dict[str, pd.Series], draws: int = 10000, seed: int = 2026
) -> tuple[float, float, float]:
    """Equal-weight models and resample matched scenario indices."""
    scenarios = sorted(set().union(*(set(values.index) for values in differences.values())))
    if not scenarios:
        raise ValueError("No paired scenarios for macro effect")
    matrix = np.column_stack(
        [differences[model].reindex(scenarios).to_numpy(float) for model in MODELS]
    )
    point = float(np.nanmean([differences[model].mean() for model in MODELS]))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=float)
    chunk = 500
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = rng.integers(0, len(scenarios), size=(stop - start, len(scenarios)))
        selected = matrix[indices, :]
        with np.errstate(invalid="ignore"):
            sample_model_means = np.nanmean(selected, axis=1)
            bootstrap[start:stop] = np.nanmean(sample_model_means, axis=1)
    valid = bootstrap[np.isfinite(bootstrap)]
    if not len(valid):
        raise ValueError("No valid macro bootstrap samples")
    low, high = np.percentile(valid, [2.5, 97.5])
    return point * 100, float(low) * 100, float(high) * 100


def age_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate age-specific scenario-weighted refusal and intervals."""
    rows = []
    for age in AGES:
        values = to_series(
            analysis.by_scenario(
                frame, "refusal", conditions=[AGE_CONDITION[age]]
            )
        )
        if values.empty:
            rows.append((age, np.nan, np.nan, np.nan))
            continue
        point, low, high = analysis.bootstrap_paired(values)
        rows.append((age, point * 100, low * 100, high * 100))
    return pd.DataFrame(
        rows, columns=["age", "point", "low", "high"]
    ).set_index("age")


def register_rows(register: pd.DataFrame, contrast: str) -> pd.DataFrame:
    """Sort a frozen contrast with the Macro-Average at the bottom."""
    order = [MACRO] + MODELS[::-1]
    selected = register.loc[register["contrast"].eq(contrast)].copy()
    missing = [model for model in order if model not in set(selected["model"])]
    if missing:
        raise ValueError(f"Missing registered rows for {contrast}: {missing}")
    selected["order"] = pd.Categorical(
        selected["model"], categories=order, ordered=True
    )
    return selected.sort_values("order")


def forest(ax, frame: pd.DataFrame, annotate_macro: bool = True) -> None:
    """Draw aligned model-level estimates and interval bars."""
    y = np.arange(len(frame))
    for position, (_, row) in enumerate(frame.iterrows()):
        model = row["model"]
        colour = MODEL_COLOUR[model]
        is_macro = model == MACRO
        point, low, high = [float(row[key]) for key in ("effect", "low", "high")]
        ax.errorbar(
            point,
            position,
            xerr=np.array([[max(0, point - low)], [max(0, high - point)]]),
            fmt=MODEL_MARKER[model],
            color=colour,
            ecolor=colour,
            markerfacecolor=colour,
            markeredgecolor=colour,
            markeredgewidth=1.1,
            markersize=5.0 if is_macro else 4.1,
            linewidth=1.3 if is_macro else 1.05,
            elinewidth=1.3 if is_macro else 1.05,
            capsize=2.0,
            zorder=3,
        )
        if is_macro and annotate_macro:
            ax.text(
                high + 1.5,
                position,
                f"{point:+.1f}",
                ha="left",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color=colour,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
                clip_on=False,
                zorder=4,
            )
    ax.axvline(0, color=BLACK, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    macro_positions = [
        i for i, model in enumerate(frame["model"]) if model == MACRO
    ]
    if macro_positions:
        ax.axhline(
            macro_positions[0] + 0.5, color=LIGHT_GREY, linewidth=0.8, zorder=1
        )
    ax.set_yticks(y, [MODEL_AXIS[model] for model in frame["model"]])
    ax.set_ylim(-0.5, len(frame) - 0.5)
    style_axis(ax)


# Main Figures

def draw_primary(
    returned: pd.DataFrame,
    focus: pd.DataFrame,
    register: pd.DataFrame,
    output: Path,
    png: bool = False,
) -> None:
    """Draw the three registered safety contrasts."""
    del returned, focus
    frames = {contrast: register_rows(register, contrast) for contrast in PRIMARY}
    extremes = pd.concat(
        [frame[["low", "high"]] for frame in frames.values()], ignore_index=True
    )
    low = min(0, float(extremes["low"].min()))
    high = max(0, float(extremes["high"].max()))
    span = max(1, high - low)
    fig, axes = plt.subplots(1, 3, figsize=(8.35, 4.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.265, right=0.992, bottom=0.19, top=0.88, wspace=0.10)
    for i, (ax, contrast) in enumerate(zip(axes, PRIMARY)):
        forest(ax, frames[contrast])
        ax.set_title(PRIMARY_LABEL[contrast], pad=5)
        ax.set_xlim(low - 0.06 * span, high + 0.13 * span)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        if i:
            ax.tick_params(labelleft=False)
    centered_xlabel(fig, axes, "Difference in Refusal Rate (pp)", 0.075)
    save_figure(fig, output, "safety_primary", png)


def draw_trajectory(
    returned: pd.DataFrame,
    focus: pd.DataFrame,
    register: pd.DataFrame,
    output: Path,
    png: bool = False,
) -> None:
    """Draw thin model trajectories and a Macro-Average CI ribbon."""
    del returned, register
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 4.35))
    fig.subplots_adjust(left=0.105, right=0.992, bottom=0.285, top=0.88)
    style_axis(ax)
    ax.axvline(18, color=BLACK, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    macro_rows = []
    observed = []
    for model in MODELS:
        part = focus.loc[focus["label"].eq(model)]
        estimates = age_table(part)
        points = estimates["point"].to_numpy(float)
        observed.extend(points[np.isfinite(points)].tolist())
        ax.plot(
            AGES,
            points,
            color=MODEL_COLOUR[model],
            marker=MODEL_MARKER[model],
            markerfacecolor="white",
            markeredgecolor=MODEL_COLOUR[model],
            markeredgewidth=0.95,
            markersize=3.7,
            linewidth=1.15,
            alpha=0.88,
            zorder=2,
        )
    for age in AGES:
        values = {}
        for model in MODELS:
            part = focus.loc[focus["label"].eq(model)]
            values[model] = to_series(
                analysis.by_scenario(
                    part, "refusal", conditions=[AGE_CONDITION[age]]
                )
            )
        point, low, high = macro_effect(values)
        macro_rows.append((age, point, low, high))
    macro = pd.DataFrame(macro_rows, columns=["age", "point", "low", "high"])
    x, point, low, high = [
        macro[column].to_numpy(float)
        for column in ("age", "point", "low", "high")
    ]
    ax.fill_between(x, low, high, color=MACRO_COLOUR, alpha=0.13, linewidth=0, zorder=2)
    ax.plot(
        x,
        point,
        color=MACRO_COLOUR,
        marker="D",
        markerfacecolor=MACRO_COLOUR,
        markeredgecolor=MACRO_COLOUR,
        markersize=4.6,
        linewidth=2.0,
        zorder=4,
    )
    limits = np.asarray(observed + low.tolist() + high.tolist(), dtype=float)
    limits = limits[np.isfinite(limits)]
    if limits.size:
        y_min = max(0, 5 * np.floor((limits.min() - 4) / 5))
        y_max = min(100, 5 * np.ceil((limits.max() + 4) / 5))
    else:
        y_min, y_max = 0, 100
    if y_max - y_min < 35:
        center = (y_min + y_max) / 2
        y_min = max(0, 5 * np.floor((center - 18) / 5))
        y_max = min(100, 5 * np.ceil((center + 18) / 5))
    ax.set_title("Refusal Rate By Explicit Age", pad=5)
    ax.set_xlabel("Age")
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_xticks(AGES)
    ax.set_xlim(min(AGES) - 0.5, max(AGES) + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.text(
        18,
        y_max - 1.5,
        "18",
        ha="center",
        va="top",
        fontsize=7.0,
        color=GREY,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.10},
        zorder=5,
    )
    handles = [
        Line2D(
            [], [],
            color=MODEL_COLOUR[model],
            marker=MODEL_MARKER[model],
            markerfacecolor="white",
            markeredgecolor=MODEL_COLOUR[model],
            linewidth=1.1,
            markersize=3.9,
            label=model,
        )
        for model in MODELS
    ]
    handles.append(
        Line2D(
            [], [],
            color=MACRO_COLOUR,
            marker="D",
            markerfacecolor=MACRO_COLOUR,
            markeredgecolor=MACRO_COLOUR,
            linewidth=2.0,
            markersize=4.5,
            label="Macro-Average (95% CI)",
        )
    )
    legend = ax.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor=LIGHT_GREY,
        handlelength=1.8,
        handletextpad=0.50,
        columnspacing=1.20,
        borderpad=0.45,
    )
    legend_style(legend)
    save_figure(fig, output, "safety_trajectory", png)


# Supplement Figures

def control_frame(returned: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Estimate minor-adult effects within an age-invariant scenario stratum."""
    rows = []
    differences = {}
    for model in MODELS:
        part = returned.loc[
            returned["label"].eq(model) & returned["scenario_type"].eq(scenario)
        ]
        effect, low, high, difference = model_effect(
            part, "refusal", MINOR_AGE_CONDITIONS, ADULT_AGE_CONDITIONS
        )
        rows.append({"model": model, "effect": effect, "low": low, "high": high})
        differences[model] = difference
    effect, low, high = macro_effect(differences)
    rows.append({"model": MACRO, "effect": effect, "low": low, "high": high})
    order = [MACRO] + MODELS[::-1]
    frame = pd.DataFrame(rows)
    frame["order"] = pd.Categorical(frame["model"], categories=order, ordered=True)
    return frame.sort_values("order").drop(columns="order")


def draw_controls(
    returned: pd.DataFrame,
    focus: pd.DataFrame,
    register: pd.DataFrame,
    output: Path,
    png: bool = False,
) -> None:
    """Draw the three age-invariant negative controls."""
    del focus, register
    frames = {
        scenario: control_frame(returned, scenario) for scenario in CONTROL_SCENARIOS
    }
    bounds = pd.concat(
        [frame[["low", "high"]] for frame in frames.values()], ignore_index=True
    )
    low = min(-5, float(bounds["low"].min()))
    high = max(5, float(bounds["high"].max()))
    span = max(1, high - low)
    fig, axes = plt.subplots(1, 3, figsize=(8.35, 4.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.265, right=0.992, bottom=0.19, top=0.88, wspace=0.10)
    for i, (ax, scenario) in enumerate(zip(axes, CONTROL_SCENARIOS)):
        ax.axvspan(-5, 5, color=EQUIVALENCE, zorder=0)
        forest(ax, frames[scenario])
        ax.set_title(scenario, pad=5)
        ax.set_xlim(low - 0.08 * span, high + 0.14 * span)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        if i:
            ax.tick_params(labelleft=False)
    legend = axes[1].legend(
        handles=[
            Patch(
                facecolor=EQUIVALENCE,
                edgecolor=LIGHT_GREY,
                linewidth=0.7,
                label="±5 pp equivalence margin",
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor=LIGHT_GREY,
        handlelength=1.6,
        handletextpad=0.55,
        borderpad=0.45,
    )
    legend_style(legend)
    centered_xlabel(fig, axes, "Explicit Minor − Explicit Adult Refusal (pp)", 0.075)
    save_figure(fig, output, "safety_controls", png)



# Figure Selection

FIGURES = {
    "primary": ("main", draw_primary),
    "trajectory": ("main", draw_trajectory),
    "controls": ("supplement", draw_controls),
}


def main() -> None:
    """Generate the selected publication safety figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--set", choices=["main", "supplement", "all"], default="all")
    parser.add_argument("--only", nargs="+", choices=list(FIGURES))
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    jobs = [
        function
        for name, (tier, function) in FIGURES.items()
        if (args.set == "all" or args.set == tier)
        and (not args.only or name in args.only)
    ]
    if not jobs:
        parser.error("No figures match the requested selection")
    set_style()
    returned, focus, register = load_data()
    for function in jobs:
        function(returned, focus, register, args.output, args.png)


if __name__ == "__main__":
    main()
