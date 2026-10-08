"""Compare explicit-age refusal and readability trajectories on one page.

This figure is an optional *replacement* for the separate trajectories in a
space-limited manuscript. It does not change or pool their denominators:
refusal is measured on Age Restricted scenarios; FKGL uses measurable replies
across all scenario types (50-word floor). The two panels have separate axes.

Run from the publication repository: python scripts/figjoint.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import analysis
import language

AGES = [7, 9, 11, 13, 15, 17, 18, 21]
MODELS = ["GPT-5.6 Luna", "Claude Haiku 4.5", "Gemini 3.5 Flash Lite",
          "DeepSeek-V4 Flash", "Mistral Small 4", "Gemma 4 31B"]
MACRO_COLOUR = "#5E3C99"
GREY = "#666666"
GRID = "#C7C7C7"
OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "joint"


def scenario_rates(frame: pd.DataFrame, measure: str, age: int) -> dict[str, pd.Series]:
    """Return model-indexed scenario means for one explicit age."""
    condition = f"age{age:02d}"
    series = {}
    for model in MODELS:
        selected = frame.loc[frame["label"].eq(model)]
        series[model] = analysis.by_scenario(selected, measure, conditions=[condition])
    return series


def trajectories(safety: pd.DataFrame, readable: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate paired-scenario bootstrap macro-estimates without pooling replies."""
    result = {}
    for name, source, measure, multiplier in (
        ("refusal", safety, "refusal", 100),
        ("fkgl", readable, "fkgl", 1),
    ):
        rows = []
        for age in AGES:
            values = scenario_rates(source, measure, age)
            if any(series.empty for series in values.values()):
                raise ValueError(f"Missing {name} results at explicit age {age}")
            point, low, high = analysis.macro_average(values)
            rows.append((age, point * multiplier, low * multiplier, high * multiplier))
        result[name] = pd.DataFrame(rows, columns=["age", "point", "low", "high"])
    return result


def draw(results: dict[str, pd.DataFrame], output: Path, png: bool = False) -> Path:
    """Render two proportionate panels with matched age ticks and separate y-axes."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.3,
        "axes.titlesize": 9.2, "axes.titleweight": "bold",
        "axes.labelsize": 8.7, "xtick.labelsize": 7.9,
        "ytick.labelsize": 7.9, "grid.color": GRID,
        "grid.alpha": 0.72, "grid.linewidth": 0.55,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.75, 3.65), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.27, top=0.88, wspace=0.24)
    meta = [
        ("refusal", "Refusal Rate", "Refusal Rate (%)"),
        ("fkgl", "Reading Level", "Flesch–Kincaid Grade Level"),
    ]
    for ax, (key, title, ylabel) in zip(axes, meta):
        frame = results[key]
        x = frame["age"].to_numpy(float)
        mean = frame["point"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        high = frame["high"].to_numpy(float)
        ax.axvspan(18, 21.5, facecolor="#F6F2FA", edgecolor="none", zorder=0)
        ax.axvline(18, color=GREY, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
        ax.fill_between(x, low, high, color=MACRO_COLOUR, alpha=0.15, zorder=2, linewidth=0)
        ax.plot(x, mean, color=MACRO_COLOUR, lw=2, marker="D", markersize=4.4,
                markerfacecolor="white", markeredgewidth=1.2, zorder=4)
        ax.set_title(title, pad=7)
        ax.set_xlabel("Explicit Age")
        ax.set_ylabel(ylabel)
        ax.set_xticks(AGES)
        ax.set_xlim(6.5, 21.5)
        finite = np.r_[low, high]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            padding = max(float(np.ptp(finite)) * 0.15, 0.6 if key == "fkgl" else 3.5)
            ax.set_ylim(float(finite.min()) - padding, float(finite.max()) + padding)
        ax.grid(axis="both")
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3)
        for spine in ax.spines.values():
            spine.set_color("#111111")
            spine.set_linewidth(0.8)

    center = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    fig.legend(
        handles=[Line2D([], [], color=MACRO_COLOUR, marker="D",
                        markerfacecolor="white", linewidth=2,
                        label="Macro-Average (95% CI)"),
                 Line2D([], [], color=GREY, linestyle=(0, (4, 3)),
                        linewidth=1, label="Age 18 Boundary")],
        loc="lower center", ncol=2, bbox_to_anchor=(center, 0.035),
        frameon=True, fancybox=True, edgecolor="#D8D8D8", fontsize=7.8,
    )
    output.mkdir(parents=True, exist_ok=True)
    filename = output / "age_conditioning_profiles.pdf"
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.04)
    if png:
        fig.savefig(output / "age_conditioning_profiles.png", dpi=300,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Figure: {filename.name}")
    return filename


def main():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--output", type=Path, default=OUTPUT)
    cli.add_argument("--floor", type=int, default=50)
    cli.add_argument("--png", action="store_true")
    args = cli.parse_args()

    original = analysis.load_corpus()
    safety = original.loc[
        original["responded"] & original["scenario_type"].eq(analysis.FOCUS)
    ].copy()

    readable = language.load()
    readable["label"] = readable["model"].map(analysis.NAME)
    readable = readable.loc[readable["age"].isin(AGES)].copy()
    readable.loc[readable["response_length"] < args.floor, "fkgl"] = np.nan

    draw(trajectories(safety, readable), args.output, png=args.png)


if __name__ == "__main__":
    main()
