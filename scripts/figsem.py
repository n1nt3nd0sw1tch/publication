"""Generate publication semantic figures.

Notes
-----
Create the semantic figures kept for the paper:
1. Experiment 1 semantic specificity by contrast.
2. Experiment 2 semantic drift by attack method and by model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


# Paths

def resolve_root() -> Path:
    """Resolve the repository root from common script locations."""
    here = Path(__file__).resolve()
    for candidate in [here.parent] + list(here.parents):
        if (candidate / "results" / "semantics").exists() or (candidate / "tables").exists():
            return candidate
    return here.parents[1]


ROOT = resolve_root()
SEMANTICS = ROOT / "results" / "semantics"
OUTPUT = ROOT / "figures" / "semantic"


# Study Order

MODELS = [
    "GPT-5.6 Luna",
    "Claude Haiku 4.5",
    "Gemini 3.5 Flash Lite",
    "DeepSeek-V4 Flash",
    "Mistral Small 4",
    "Gemma 4 31B",
]

MODEL_NAME = {
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "deepseek-v4-flash": "DeepSeek-V4 Flash",
    "mistral-small-2603": "Mistral Small 4",
    "gemma4:31b-cloud": "Gemma 4 31B",
}

MODEL_AXIS = {
    "GPT-5.6 Luna": "GPT-5.6 Luna",
    "Claude Haiku 4.5": "Claude Haiku 4.5",
    "Gemini 3.5 Flash Lite": "Gemini 3.5 Flash Lite",
    "DeepSeek-V4 Flash": "DeepSeek-V4 Flash",
    "Mistral Small 4": "Mistral Small 4",
    "Gemma 4 31B": "Gemma 4 31B",
}

METHODS = [
    "Emotional Pushback",
    "Role Play",
    "Purpose Reverse",
]

METHOD_AXIS = {
    "Emotional Pushback": "Emotional\nPushback",
    "Role Play": "Role Play",
    "Purpose Reverse": "Purpose\nReverse",
}

CONTRASTS = [
    "Control vs Explicit Age (Minor)",
    "Explicit Age (Minor) vs Explicit Age (Adult)",
    "Explicit Age (Minor) vs Implicit Cue (Minor)",
    "Explicit Age (17) vs Explicit Age (18)",
    "Control vs Implicit Cue (Minor)",
    "Implicit Cue (Minor) vs Implicit Cue (Adult)",
]

CONTRAST_AXIS = {
    "Control vs Explicit Age (Minor)": "Neutral vs Explicit Minor",
    "Explicit Age (Minor) vs Explicit Age (Adult)": "Explicit Minor vs Explicit Adult",
    "Explicit Age (Minor) vs Implicit Cue (Minor)": "Explicit Minor vs Implicit Minor",
    "Explicit Age (17) vs Explicit Age (18)": "17 vs 18",
    "Control vs Implicit Cue (Minor)": "Neutral vs Implicit Minor",
    "Implicit Cue (Minor) vs Implicit Cue (Adult)": "Implicit Minor vs Implicit Adult",
}


# Publication Style

TEXT_WIDTH = 7.15

BLUE = "#009E73"
ORANGE = "#CC79A7"
BLACK = "#111111"
GREY = "#6F6F6F"
LIGHT_GREY = "#D8D8D8"
GRID = "#C7C7C7"

ENCODER_COLOUR = {
    "minilm": BLUE,
    "mpnet": ORANGE,
}

ENCODER_MARKER = {
    "minilm": "o",
    "mpnet": "s",
}

ENCODER_LABEL = {
    "minilm": "MiniLM",
    "mpnet": "MPNet",
}

TURN_STYLE = {
    "T2": dict(color=BLUE, marker="o", markerfacecolor="white", markeredgecolor=BLUE),
    "T3": dict(color=ORANGE, marker="s", markerfacecolor=ORANGE, markeredgecolor=ORANGE),
}


def set_style() -> None:
    """Apply the shared journal plotting theme."""
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
    """Apply shared axis styling."""
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(0.8)
    ax.tick_params(length=2.8, width=0.7, direction="out")


def read_frame(encoder: str, stem: str) -> pd.DataFrame:
    """Read a frozen semantic table."""
    path = SEMANTICS / encoder / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Expected frozen results under results/semantics/{encoder}/."
        )
    return pd.read_csv(path)


def save_figure(fig, output: Path, name: str, png: bool = False) -> None:
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


def ordered_rows(frame: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    """Return a frame sorted by a publication order."""
    rank = {name: idx for idx, name in enumerate(order)}
    out = frame.copy()
    out["_order"] = out[column].map(rank)
    missing = out.loc[out["_order"].isna(), column].drop_duplicates().tolist()
    if missing:
        raise ValueError(f"Unexpected values in {column}: {missing}")
    return out.sort_values("_order").drop(columns="_order")


def read_specificity(encoder: str) -> pd.DataFrame:
    """Read Experiment 1 semantic specificity."""
    frame = read_frame(encoder, "e1_specificity")
    required = {"contrast", "specificity", "low", "high"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{encoder}/e1_specificity.csv is missing {sorted(missing)}. "
            f"Columns are {list(frame.columns)}"
        )
    frame = ordered_rows(frame, "contrast", CONTRASTS)
    return frame.set_index("contrast").reindex(CONTRASTS).reset_index()


def read_drift_summary(encoder: str = "minilm") -> pd.DataFrame:
    """Read Experiment 2 semantic drift summary."""
    frame = read_frame(encoder, "e2_summary")
    required = {
        "axis",
        "method",
        "model",
        "drift_12",
        "drift_12_low",
        "drift_12_high",
        "drift_13",
        "drift_13_low",
        "drift_13_high",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{encoder}/e2_summary.csv is missing {sorted(missing)}. "
            f"Columns are {list(frame.columns)}"
        )
    return frame


def draw_specificity(output: Path, png: bool = False) -> None:
    """Create the Experiment 1 specificity figure."""
    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(TEXT_WIDTH, 3.85))
    fig.subplots_adjust(left=0.34, right=0.99, bottom=0.23, top=0.88)

    style_axis(ax, "both")
    ax.axvline(0, color=BLACK, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)

    y = np.arange(len(CONTRASTS))
    offsets = {"minilm": -0.12, "mpnet": 0.12}

    handles = []
    all_low = []
    all_high = []

    for encoder in ("minilm", "mpnet"):
        frame = read_specificity(encoder)
        estimate = frame["specificity"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        high = frame["high"].to_numpy(float)
        all_low.extend(low.tolist())
        all_high.extend(high.tolist())

        colour = ENCODER_COLOUR[encoder]
        marker = ENCODER_MARKER[encoder]
        yy = y + offsets[encoder]

        for index in range(len(frame)):
            ax.errorbar(
                estimate[index],
                yy[index],
                xerr=np.array(
                    [
                        [estimate[index] - low[index]],
                        [high[index] - estimate[index]],
                    ]
                ),
                fmt=marker,
                color=colour,
                ecolor=colour,
                markerfacecolor=("white" if encoder == "mpnet" else colour),
                markeredgecolor=colour,
                markeredgewidth=1.1,
                markersize=4.4,
                linewidth=1.15,
                elinewidth=1.1,
                capsize=2.0,
                zorder=3,
            )

        value_offset = 0.0035 if encoder == "minilm" else 0.0055
        for index, value in enumerate(estimate):
            label_x = high[index] + value_offset
            ax.text(
                label_x,
                yy[index],
                f"{value:+.2f}",
                ha="left",
                va="center",
                color=colour,
                fontsize=7.0,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "pad": 0.12,
                    "alpha": 0.9,
                },
                zorder=4,
                clip_on=False,
            )

        handles.append(
            Line2D(
                [],
                [],
                color=colour,
                marker=marker,
                markerfacecolor=("white" if encoder == "mpnet" else colour),
                markeredgecolor=colour,
                markeredgewidth=1.1,
                linewidth=1.2,
                label=ENCODER_LABEL[encoder],
            )
        )

    ax.set_yticks(y, [CONTRAST_AXIS[name] for name in CONTRASTS])
    ax.set_ylim(len(CONTRASTS) - 0.5, -0.5)
    ax.set_xlabel("Specificity (Age Restricted − Control Mean)")
    ax.set_title("Semantic Specificity By Contrast", pad=5)

    lower = min(min(all_low), -0.01)
    upper = max(max(all_high), 0.01)
    ax.set_xlim(lower - 0.012, upper + 0.028)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))

    legend = ax.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor=LIGHT_GREY,
        handlelength=1.8,
        handletextpad=0.55,
        columnspacing=1.6,
        borderpad=0.45,
    )
    legend.get_frame().set_linewidth(0.6)
    for line in legend.get_lines():
        line.set_linewidth(1.2)

    save_figure(fig, output, "semantic_specificity", png)


def draw_drift(output: Path, png: bool = False) -> None:
    """Create the Experiment 2 drift figure."""
    frame = read_drift_summary("minilm")
    set_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.85, 4.20),
        gridspec_kw={"width_ratios": [0.92, 1.22]},
    )
    fig.subplots_adjust(left=0.11, right=0.992, bottom=0.27, top=0.88, wspace=0.82)

    # Panel A: methods
    ax = axes[0]
    style_axis(ax, "both")
    methods = ordered_rows(frame[frame["axis"].eq("method")].copy(), "method", METHODS)
    y = np.arange(len(METHODS))
    offsets = {"T2": -0.14, "T3": 0.14}

    for turn, columns in {
        "T2": ("drift_12", "drift_12_low", "drift_12_high"),
        "T3": ("drift_13", "drift_13_low", "drift_13_high"),
    }.items():
        est = methods[columns[0]].to_numpy(float)
        low = methods[columns[1]].to_numpy(float)
        high = methods[columns[2]].to_numpy(float)
        yy = y + offsets[turn]
        style = TURN_STYLE[turn]

        for index in range(len(methods)):
            ax.errorbar(
                est[index],
                yy[index],
                xerr=np.array(
                    [
                        [est[index] - low[index]],
                        [high[index] - est[index]],
                    ]
                ),
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=1.05,
                markersize=4.4,
                linewidth=1.15,
                elinewidth=1.1,
                capsize=2.0,
                zorder=3,
            )

        for index, value in enumerate(est):
            xpad = 0.010 if turn == "T2" else 0.012
            ax.text(
                high[index] + xpad,
                yy[index],
                f"{value:.2f}",
                ha="left",
                va="center",
                color=style["color"],
                fontsize=7.0,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "pad": 0.10,
                    "alpha": 0.9,
                },
                zorder=4,
                clip_on=False,
            )

    ax.set_yticks(y, [METHOD_AXIS[name] for name in METHODS])
    ax.set_ylim(len(METHODS) - 0.5, -0.5)
    ax.set_xlabel("Semantic Drift")
    ax.set_title("(a) Drift By Attack Method", pad=5)

    # Panel B: models
    ax = axes[1]
    style_axis(ax, "both")
    models = frame[frame["axis"].eq("model")].copy()
    models["model"] = models["model"].replace(MODEL_NAME)
    models = ordered_rows(models, "model", MODELS)
    y = np.arange(len(MODELS))

    for turn, columns in {
        "T2": ("drift_12", "drift_12_low", "drift_12_high"),
        "T3": ("drift_13", "drift_13_low", "drift_13_high"),
    }.items():
        est = models[columns[0]].to_numpy(float)
        low = models[columns[1]].to_numpy(float)
        high = models[columns[2]].to_numpy(float)
        yy = y + offsets[turn]
        style = TURN_STYLE[turn]

        for index in range(len(models)):
            ax.errorbar(
                est[index],
                yy[index],
                xerr=np.array(
                    [
                        [est[index] - low[index]],
                        [high[index] - est[index]],
                    ]
                ),
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=1.05,
                markersize=4.4,
                linewidth=1.15,
                elinewidth=1.1,
                capsize=2.0,
                zorder=3,
            )

        for index, value in enumerate(est):
            xpad = 0.010 if turn == "T2" else 0.012
            ax.text(
                high[index] + xpad,
                yy[index],
                f"{value:.2f}",
                ha="left",
                va="center",
                color=style["color"],
                fontsize=6.9,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "pad": 0.10,
                    "alpha": 0.9,
                },
                zorder=4,
                clip_on=False,
            )

    ax.set_yticks(y, [MODEL_AXIS[name] for name in MODELS])
    ax.set_ylim(len(MODELS) - 0.5, -0.5)
    ax.set_xlabel("Semantic Drift")
    ax.set_title("(b) Drift By Model", pad=5)

    # Shared x-limits
    x_low = min(
        float(methods["drift_12_low"].min()),
        float(methods["drift_13_low"].min()),
        float(models["drift_12_low"].min()),
        float(models["drift_13_low"].min()),
    )
    x_high = max(
        float(methods["drift_12_high"].max()),
        float(methods["drift_13_high"].max()),
        float(models["drift_12_high"].max()),
        float(models["drift_13_high"].max()),
    )
    xmin = max(0.0, x_low - 0.03)
    xmax = x_high + 0.09

    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(MultipleLocator(0.10))

    handles = [
        Line2D(
            [],
            [],
            color=TURN_STYLE[turn]["color"],
            marker=TURN_STYLE[turn]["marker"],
            markerfacecolor=TURN_STYLE[turn]["markerfacecolor"],
            markeredgecolor=TURN_STYLE[turn]["markeredgecolor"],
            markeredgewidth=1.05,
            linewidth=1.2,
            label=turn,
        )
        for turn in ("T2", "T3")
    ]

    legend = fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.025),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.6,
    )
    for line in legend.get_lines():
        line.set_linewidth(1.2)

    save_figure(fig, output, "semantic_drift", png)


BUILDERS = {
    "specificity": draw_specificity,
}


def main() -> None:
    """Parse arguments and build figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(BUILDERS),
        help="Draw only the named figure(s).",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also export 300-dpi PNG previews.",
    )
    args = parser.parse_args()

    jobs = BUILDERS
    if args.only:
        jobs = {name: fn for name, fn in BUILDERS.items() if name in args.only}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"Semantic figures -> {OUTPUT}/")
    for _, builder in jobs.items():
        builder(OUTPUT, args.png)


if __name__ == "__main__":
    main()
