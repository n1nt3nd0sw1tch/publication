"""Generate publication readability figures for Experiment 1.

Notes
-----
Export the primary FKGL contrast and explicit-age trajectory, together
with supplementary mean AoA contrasts and age-signal trends. Use the
shared visual style and model palette of figsafe.py and figdial.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd


# Paths

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUTPUT = ROOT / "figures" / "readability"
sys.path.insert(0, str(SCRIPTS))

import analysis
import language
from settings import ROOT as SETTINGS_ROOT

ROOT = SETTINGS_ROOT
OUTPUT = ROOT / "figures" / "readability"
CONDITIONING = ROOT / "tables" / "main" / "readability_02_conditioning.csv"


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
ORDER = [MACRO] + MODELS[::-1]
DISPLAY = {model: model for model in MODELS}
DISPLAY[MACRO] = "Macro-Average"

AGES = [7, 9, 11, 13, 15, 17, 18, 21]
MINOR = [7, 9, 11, 13, 15, 17]
ADULT = [18, 21]
TARGET = {age: min(age - 5, 12) for age in MINOR}
FORMULAE = ["fkgl", "fre", "gunning_fog", "ari", "smog"]

SIGNALS = [
    ("neutral", "Neutral"),
    ("adult_cue", "Adult\nCue"),
    ("adult_age", "Adult\nAge"),
    ("minor_cue", "Minor\nCue"),
    ("minor_age", "Minor\nAge"),
]
TRACKS = [
    ("fkgl", "FKGL", "-", "o"),
    ("mean_aoa", "Mean AoA", "--", "^"),
    ("p90_aoa", "P90 AoA", ":", "s"),
]


# Publication Style

BLACK = "#111111"
GREY = "#6F6F6F"
LIGHT_GREY = "#D8D8D8"
MACRO_FILL = "#F6F2FA"
SIGNAL_FILL = "#F8F6FC"
GRID = "#C7C7C7"
MACRO_COLOUR = "#5E3C99"

MODEL_COLOUR = {
    "GPT-5.6 Luna": "#0072B2",
    "Claude Haiku 4.5": "#D55E00",
    "Gemini 3.5 Flash Lite": "#009E73",
    "DeepSeek-V4 Flash": "#CC79A7",
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
    """Apply the shared publication plotting style."""
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
    """Apply the same background and grid as the safety figures."""
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(0.8)
    ax.tick_params(length=2.8, width=0.7, direction="out")


def save_figure(fig, name: str, output: Path, png: bool = False) -> None:
    """Save PDF and optional publication-resolution PNG."""
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.035)
    if png:
        fig.savefig(
            output / f"{name}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.035,
        )
    plt.close(fig)
    print(f"Figure: {pdf.name}")


def plot_center(axes) -> float:
    """Return the horizontal centre of the occupied axes region."""
    flat = np.asarray(axes).ravel()
    return (min(ax.get_position().x0 for ax in flat) + max(ax.get_position().x1 for ax in flat)) / 2


def plot_legend(fig, axes, handles, labels, columns: int, y: float = 0.025) -> None:
    """Centre a legend beneath the plot rather than the canvas."""
    legend = fig.legend(
        handles,
        labels,
        ncol=columns,
        loc="lower center",
        bbox_to_anchor=(plot_center(axes), y),
        bbox_transform=fig.transFigure,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor=LIGHT_GREY,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.2,
        borderpad=0.45,
    )
    legend.get_frame().set_linewidth(0.6)


# Estimation

def by_scenario(data: pd.DataFrame, measure: str, keys: list[str]) -> pd.Series:
    """Average replicates within each scenario and condition first."""
    cell = data.groupby(
        keys + ["scenario_id", "condition"], observed=True
    )[measure].mean()
    scenario = cell.groupby(keys + ["scenario_id"], observed=True).mean()
    return scenario.groupby(keys, observed=True).mean()


def age_blocks(data: pd.DataFrame, measure: str) -> tuple[pd.Series, pd.Series]:
    """Return paired minor and adult scenario averages."""
    wide = (
        data.pivot_table(
            index="scenario_id", columns="age", values=measure, aggfunc="mean"
        )
        .reindex(columns=MINOR + ADULT)
        .dropna()
    )
    if wide.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    return wide[MINOR].mean(axis=1), wide[ADULT].mean(axis=1)


def contrast(data: pd.DataFrame, measure: str) -> tuple[float, float, float]:
    """Bootstrap the within-scenario minor versus adult contrast."""
    minor, adult = age_blocks(data, measure)
    if minor.empty:
        return np.nan, np.nan, np.nan
    return analysis.bootstrap_paired(minor - adult)


def conditioning_results(data: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Compute model and shared-scenario Macro-Average estimates."""
    rows = []
    paired = {}
    for model in MODELS:
        sample = data[data["label"].eq(model)]
        minor, adult = age_blocks(sample, measure)
        paired[model] = minor - adult
        point, low, high = contrast(sample, measure)
        rows.append(dict(model=model, effect=point, low=low, high=high))
    point, low, high = analysis.macro_average(paired)
    rows.append(dict(model=MACRO, effect=point, low=low, high=high))
    result = pd.DataFrame(rows).set_index("model")
    return result.reindex(ORDER)


def load_fkgl_results() -> pd.DataFrame:
    """Read published primary FKGL estimates without recomputing them."""
    if not CONDITIONING.exists():
        raise FileNotFoundError(
            f"Missing {CONDITIONING}. Run the readability notebook first."
        )
    table = pd.read_csv(CONDITIONING).set_index("Model")
    columns = {
        "Effect (Grades)": "effect",
        "95% CI Lower": "low",
        "95% CI Upper": "high",
    }
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"Missing FKGL result columns: {missing}")
    missing_models = sorted(set(ORDER) - set(table.index))
    if missing_models:
        raise ValueError(f"Missing FKGL result rows: {missing_models}")
    return table.rename(columns=columns).loc[ORDER, list(columns.values())].astype(float)


def signal_level(data: pd.DataFrame) -> pd.Series:
    """Group neutral, implicit-cue and explicit-age conditions."""
    level = pd.Series("", index=data.index, dtype=object)
    cue = data["signal"].eq("cue")
    condition = data["condition"].astype(str)
    level[data["condition"].eq("neutral")] = "neutral"
    level[cue & condition.str.contains("adult", na=False)] = "adult_cue"
    level[cue & condition.str.contains("minor", na=False)] = "minor_cue"
    level[data["age"].isin(ADULT)] = "adult_age"
    level[data["age"].isin(MINOR)] = "minor_age"
    return level


# Figures

def draw_forest(
    result: pd.DataFrame,
    title: str,
    xlabel: str,
    name: str,
    output: Path,
    png: bool = False,
) -> None:
    """Plot effect estimates with uncertainty and aligned numerical values."""
    fig, ax = plt.subplots(figsize=(8.4, 3.75))
    fig.subplots_adjust(left=0.255, right=0.715, top=0.865, bottom=0.185)
    style_axis(ax)

    values = result[["low", "high"]].to_numpy(float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError(f"No finite results for {name}")
    span = max(float(np.max(finite) - np.min(finite)), 0.2)
    lower = float(min(np.min(finite) - 0.12 * span, -0.05))
    upper = float(max(0.08, 0.0 + 0.05 * span))
    ax.set_xlim(lower, upper)

    ax.axhspan(-0.47, 0.47, facecolor=MACRO_FILL, linewidth=0, zorder=0)
    ax.axvline(0, color=BLACK, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.axhline(0.5, color=LIGHT_GREY, linewidth=1.0, zorder=2)

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for y, model in enumerate(ORDER):
        row = result.loc[model]
        point, low, high = (float(row[column]) for column in ["effect", "low", "high"])
        if not all(np.isfinite([point, low, high])):
            continue
        colour = MODEL_COLOUR[model]
        macro = model == MACRO
        ax.errorbar(
            point, y,
            xerr=np.array([[max(0, point - low)], [max(0, high - point)]]),
            fmt=MODEL_MARKER[model],
            color=colour,
            ecolor=colour,
            markerfacecolor=colour if macro else "white",
            markeredgecolor=colour,
            markeredgewidth=1.3,
            markersize=6.7 if macro else 5.0,
            elinewidth=1.85 if macro else 1.45,
            capsize=2.5,
            zorder=4,
        )
        ax.text(
            1.055, y, f"{point:+.2f} [{low:+.2f}, {high:+.2f}]",
            transform=transform,
            va="center", ha="left", fontsize=7.6,
            fontweight="bold" if macro else "normal",
            color=MACRO_COLOUR if macro else BLACK,
            clip_on=False,
            zorder=5,
        )

    ax.set_yticks(range(len(ORDER)), [DISPLAY[m] for m in ORDER])
    for label in ax.get_yticklabels():
        if label.get_text() == "Macro-Average":
            label.set_color(MACRO_COLOUR)
            label.set_fontweight("bold")
    ax.set_ylim(-0.55, len(ORDER) - 0.45)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_title(title, pad=8)
    ax.text(
        1.055, 1.015, "Estimate [95% CI]",
        ha="left", va="bottom", transform=ax.transAxes,
        fontsize=7.7, color=GREY, fontweight="bold", clip_on=False,
    )
    save_figure(fig, name, output, png)

def draw_conditioning(data: pd.DataFrame, output: Path, png: bool = False) -> None:
    """Plot the primary Flesch-Kincaid grade-level contrast."""
    del data
    draw_forest(
        load_fkgl_results(),
        "Explicit Minor vs Explicit Adult",
        "Difference in FKGL (grades)",
        "readability_conditioning", output, png,
    )


def draw_aoa(data: pd.DataFrame, output: Path, png: bool = False) -> None:
    """Plot the mean age-of-acquisition contrast."""
    draw_forest(
        conditioning_results(data, "mean_aoa"),
        "Explicit Minor vs Explicit Adult",
        "Difference in Mean AoA (years)",
        "readability_conditioning_mean_aoa", output, png,
    )


def draw_ladder(data: pd.DataFrame, output: Path, png: bool = False) -> None:
    """Plot explicit-age FKGL with a 17-to-18 readout per model."""
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.55), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.109, right=0.988, top=0.90, bottom=0.205,
        wspace=0.19, hspace=0.36,
    )

    for i, (ax, model) in enumerate(zip(axes.flat, MODELS)):
        part = data[data["label"].eq(model)]
        observed = by_scenario(part, "fkgl", ["age"]).reindex(AGES)
        colour = MODEL_COLOUR[model]
        style_axis(ax)
        ax.axvspan(17.5, 21.6, facecolor=SIGNAL_FILL, linewidth=0, zorder=0)
        ax.axvline(17.5, color=GREY, linestyle=(0, (4, 3)), linewidth=0.9, zorder=2)
        ax.plot(
            list(TARGET), list(TARGET.values()),
            linestyle=":", linewidth=1.35, color=GREY, zorder=3,
        )
        ax.plot(
            AGES, observed.to_numpy(float),
            color=colour, marker=MODEL_MARKER[model], markersize=4.25,
            markerfacecolor="white", markeredgecolor=colour,
            markeredgewidth=1.1, linewidth=1.75, zorder=4,
        )
        if np.isfinite(observed.loc[17]) and np.isfinite(observed.loc[18]):
            delta = float(observed.loc[18] - observed.loc[17])
            ax.plot(
                [17, 18], [observed.loc[17], observed.loc[18]],
                color=colour, linewidth=2.6, solid_capstyle="round", zorder=5,
            )
            ax.text(
                0.035, 0.975, f"17–18: {delta:+.2f} grades",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.0, color=BLACK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 2.0},
                zorder=7,
            )

        ax.set_title(model, pad=6)
        ax.set_xlim(6.4, 21.6)
        ax.set_ylim(1.5, 12.6)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.set_xticks(AGES)
        ax.tick_params(axis="x", labelsize=7.1, labelbottom=i >= 3)
        if i >= 3:
            fig.canvas.draw()
            labels = ax.get_xticklabels()
            labels[5].set_ha("right")
            labels[6].set_ha("left")
        if i % 3 != 0:
            ax.tick_params(axis="y", labelleft=False)

    left = min(ax.get_position().x0 for ax in axes.flat)
    right = max(ax.get_position().x1 for ax in axes.flat)
    fig.text(0.035, 0.555, "Flesch-Kincaid Grade Level",
             rotation=90, ha="center", va="center", fontsize=8.7)
    fig.text((left + right) / 2, 0.113, "Age", ha="center", va="center", fontsize=8.7)
    plot_legend(
        fig, axes,
        [
            Line2D([], [], color=GREY, linewidth=1.35, linestyle=":"),
            Line2D([], [], color=GREY, linewidth=0.9, linestyle=(0, (4, 3))),
        ],
        ["Target Grade", "Age 18 Boundary"], 2, y=0.025,
    )
    save_figure(fig, "readability_ladder", output, png)

def draw_signals(data: pd.DataFrame, output: Path, png: bool = False) -> None:
    """Plot age-signal effects with compact explicit/implicit FKGL summaries."""
    frame = data.assign(level=signal_level(data))
    frame = frame[frame["level"].ne("")]
    levels = [key for key, _ in SIGNALS]
    tick_labels = [label for _, label in SIGNALS]

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.65), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.109, right=0.988, top=0.90, bottom=0.23,
        wspace=0.19, hspace=0.35,
    )
    all_shifts = []

    for i, (ax, model) in enumerate(zip(axes.flat, MODELS)):
        part = frame[frame["label"].eq(model)]
        colour = MODEL_COLOUR[model]
        style_axis(ax)
        ax.axvspan(3.55, 4.2, facecolor=SIGNAL_FILL, linewidth=0, zorder=0)
        ax.axhline(0, linewidth=0.9, color=GREY, zorder=2)
        fkgl_shifts = None

        for measure, label, line_style, marker in TRACKS:
            means = by_scenario(part, measure, ["level"]).reindex(levels)
            std = float(part[measure].std())
            standardised = (means - means.loc["neutral"]) / std if std > 0 else means * np.nan
            all_shifts.extend(standardised.dropna().to_list())
            ax.plot(
                range(len(levels)), standardised.to_numpy(float),
                color=colour, linestyle=line_style, marker=marker,
                markerfacecolor="white", markeredgecolor=colour,
                markeredgewidth=1.0, markersize=4.2,
                linewidth=1.55, zorder=3,
                label=label,
            )
            if measure == "fkgl":
                fkgl_shifts = standardised

        if fkgl_shifts is not None:
            cue, explicit = fkgl_shifts.loc["minor_cue"], fkgl_shifts.loc["minor_age"]
            if np.isfinite(cue) and np.isfinite(explicit):
                ax.text(
                    0.035, 0.975,
                    f"FKGL (SD): Cue {cue:+.2f}  |  Age {explicit:+.2f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.9, color=BLACK,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 2.0},
                    zorder=6,
                )

        ax.set_title(model, pad=6)
        ax.set_xlim(-0.2, 4.2)
        ax.set_xticks(range(len(levels)), tick_labels)
        ax.tick_params(axis="x", labelsize=7.0, labelbottom=i >= 3, pad=4)
        if i % 3 != 0:
            ax.tick_params(axis="y", labelleft=False)

    if all_shifts:
        y_min = min(-0.1, np.floor((min(all_shifts) - 0.15) * 4) / 4)
        y_max = max(0.25, np.ceil((max(all_shifts) + 0.15) * 4) / 4)
        axes[0, 0].set_ylim(y_min, y_max)
        for ax in axes.flat:
            ax.yaxis.set_major_locator(
                MultipleLocator(0.25 if y_max - y_min <= 2.0 else 0.5)
            )

    fig.text(0.035, 0.54, "Change from Neutral (SD)",
             rotation=90, ha="center", va="center", fontsize=8.7)
    plot_legend(
        fig, axes,
        [
            Line2D([], [], color=BLACK, linestyle=line_style, marker=marker,
                   markerfacecolor="white", markeredgecolor=BLACK, markersize=4.2)
            for _, _, line_style, marker in TRACKS
        ],
        [label for _, label, _, _ in TRACKS], 3, y=0.085,
    )
    save_figure(fig, "readability_signals", output, png)


# Execution

FIGURES = {
    "conditioning": ("main", draw_conditioning),
    "ladder": ("main", draw_ladder),
    "signals": ("supplement", draw_signals),
    "conditioning_mean_aoa": ("supplement", draw_aoa),
}


def main() -> None:
    """Generate the selected publication figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--floor", type=int, default=50)
    parser.add_argument("--set", choices=["all", "main", "supplement"], default="all")
    parser.add_argument("--only", nargs="+", choices=list(FIGURES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    set_style()
    raw = language.load()
    raw["label"] = raw["model"].map(analysis.NAME)
    frame = raw.copy()
    frame.loc[frame["response_length"] < args.floor, FORMULAE] = np.nan
    age_data = frame[frame["signal"].eq("stated")]

    for name, (tier, function) in FIGURES.items():
        if args.set != "all" and tier != args.set:
            continue
        if args.only and name not in args.only:
            continue
        function(age_data if name != "signals" else frame, args.output, args.png)


if __name__ == "__main__":
    main()
