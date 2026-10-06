"""Semantic-similarity figures for the thesis.

Reads the frozen semantic results directly from:
    results/semantics/minilm/
    results/semantics/mpnet/

Produces only three main figures:
    figures/semantic/semantic_age.pdf
    figures/semantic/semantic_drift.pdf
    figures/semantic/semantic_outcomes.pdf

Run:
    python scripts/figuresem.py
    python scripts/figuresem.py --only age drift
    python scripts/figuresem.py --png

The script does not recompute embeddings or statistics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis
import figuresafe as house


# ---------------------------------------------------------------------
# Paths and reporting order
# ---------------------------------------------------------------------

SEMANTICS = analysis.ROOT / "results" / "semantics"
FIGURES = analysis.FIGURES / "semantic"

ENCODERS = {
    "minilm": "MiniLM",
    "mpnet": "MPNet",
}

# Keep the frozen file keys, but display them with the thesis wording.
CONTRASTS = [
    "Control vs Explicit Age (Minor)",
    "Explicit Age (Minor) vs Explicit Age (Adult)",
    "Explicit Age (17) vs Explicit Age (18)",
    "Explicit Age (Minor) vs Implicit Cue (Minor)",
    "Control vs Implicit Cue (Minor)",
    "Implicit Cue (Minor) vs Implicit Cue (Adult)",
]

CONTRAST_LABEL = {
    "Control vs Explicit Age (Minor)": "Neutral vs Minor\n(Age)",
    "Explicit Age (Minor) vs Explicit Age (Adult)": "Minor vs Adult\n(Age)",
    "Explicit Age (17) vs Explicit Age (18)": "Age 17 vs 18",
    "Explicit Age (Minor) vs Implicit Cue (Minor)": "Minor Age vs\nMinor Cue",
    "Control vs Implicit Cue (Minor)": "Neutral vs Minor\n(Cue)",
    "Implicit Cue (Minor) vs Implicit Cue (Adult)": "Minor vs Adult\n(Cue)",
}

SCENARIOS = list(analysis.SCENARIO_ORDER)
METHODS = ["Emotional Pushback", "Role Play", "Purpose Reverse"]

METHOD_STYLE = {
    "Emotional Pushback": ("#1F6E8C", "o", "-"),
    "Role Play": ("#D17A22", "s", "--"),
    "Purpose Reverse": ("#6A4C93", "^", "-."),
}

TRANSITIONS = [
    "Aligned to Aligned",
    "Aligned to Defect",
    "Defect to Aligned",
    "Defect to Defect",
]

TRANSITION_LABEL = {
    "Aligned to Aligned": "Aligned → Aligned",
    "Aligned to Defect": "Aligned → Defect",
    "Defect to Aligned": "Defect → Aligned",
    "Defect to Defect": "Defect → Defect",
}

ENCODER_STYLE = {
    "minilm": ("#376B8C", "o"),
    "mpnet": ("#8E8E93", "s"),
}


# ---------------------------------------------------------------------
# I/O and house style
# ---------------------------------------------------------------------

def read_semantic(encoder: str, stem: str) -> pd.DataFrame:
    path = SEMANTICS / encoder / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. "
            f"Expected the frozen semantic output in results/semantics/{encoder}/."
        )
    return pd.read_csv(path)


def layout(rows=1, cols=2, width=13.8, height=6.8):
    plt.rcParams.update(analysis.STYLE)
    points = house.styled(1.0, width, label_points=9.0)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(width, height),
        squeeze=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(
        w_pad=0.18,
        h_pad=0.18,
        wspace=0.18,
        hspace=0.18,
    )
    return fig, axes, points


def panel(ax, points):
    house.panel(ax, None, points)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def _stagger(values, min_gap=0.028):
    """Return collision-reduced y positions for end labels."""
    pairs = sorted(enumerate(values), key=lambda x: x[1])
    placed = [None] * len(values)
    last = None
    for idx, val in pairs:
        here = float(val)
        if last is not None and here - last < min_gap:
            here = last + min_gap
        placed[idx] = here
        last = here
    return placed


def save(fig, name: str, png=False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    if png:
        fig.savefig(FIGURES / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {pdf.relative_to(analysis.ROOT)}")


def ordered(frame: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    rank = {name: i for i, name in enumerate(order)}
    out = frame.copy()
    out["_order"] = out[column].map(rank).fillna(999)
    return out.sort_values("_order").drop(columns="_order")


def pad_limits(values, lower=None, upper=None, pad=0.02, floor=None, ceiling=None):
    arr = np.asarray(values, dtype=float)
    lo = np.nanmin(arr if lower is None else np.asarray(lower, dtype=float))
    hi = np.nanmax(arr if upper is None else np.asarray(upper, dtype=float))
    lo -= pad
    hi += pad
    if floor is not None:
        lo = min(lo, floor)
    if ceiling is not None:
        hi = max(hi, ceiling)
    return lo, hi


# ---------------------------------------------------------------------
# Experiment 1: age-conditioned semantic separation
# ---------------------------------------------------------------------

def e1_summary(encoder="minilm") -> pd.DataFrame:
    d = read_semantic(encoder, "e1_summary").copy()
    required = {
        "contrast",
        "stratum",
        "semantic_separation",
        "low",
        "high",
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            f"{encoder}/e1_summary.csv is missing {sorted(missing)}. "
            f"Columns are {list(d.columns)}"
        )
    d = ordered(d, "contrast", CONTRASTS)
    return d


def e1_specificity(encoder: str) -> pd.DataFrame:
    d = read_semantic(encoder, "e1_specificity").copy()
    required = {"contrast", "specificity", "low", "high"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            f"{encoder}/e1_specificity.csv is missing {sorted(missing)}. "
            f"Columns are {list(d.columns)}"
        )
    return ordered(d, "contrast", CONTRASTS)


def draw_age(png=False):
    """Experiment 1: scenario separation and semantic specificity."""
    d = e1_summary("minilm")

    # Extra height is deliberate: four estimates are shown for each contrast.
    fig, axes, p = layout(1, 2, width=14.8, height=8.2)

    # Left. Semantic separation by scenario type.
    ax = axes[0, 0]
    panel(ax, p)
    ax.axvline(0, color=analysis.MUTED, lw=0.9, ls="--")

    # Space each contrast group apart, then offset the four strata within it.
    base = np.arange(len(CONTRASTS))[::-1] * 1.28
    offsets = {
        "Benign": +0.27,
        "Rights": +0.09,
        "Age Restricted": -0.09,
        "Harmful": -0.27,
    }
    upper_values = []

    for scenario in SCENARIOS:
        part = (
            d[d["stratum"].eq(scenario)]
            .set_index("contrast")
            .reindex(CONTRASTS)
        )
        est = part["semantic_separation"].to_numpy(dtype=float)
        low = part["low"].to_numpy(dtype=float)
        high = part["high"].to_numpy(dtype=float)
        upper_values.extend(high.tolist())
        y = base + offsets[scenario]
        colour = analysis.SCENARIO_COLOUR[scenario]

        ax.errorbar(
            est,
            y,
            xerr=np.vstack([est - low, high - est]),
            fmt="o",
            color=colour,
            ecolor=colour,
            markersize=4.8,
            elinewidth=1.35,
            lw=1.35,
            capsize=0,
            label=scenario,
            zorder=3,
        )

        # Put the value just beyond the CI rather than on top of the point/bar.
        for hi, yi, value in zip(high, y, est):
            ax.text(
                hi + 0.004,
                yi,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=p * 0.56,
                fontweight="bold",
                color=colour,
                clip_on=False,
            )

    ax.set_yticks(base, [CONTRAST_LABEL[x] for x in CONTRASTS])
    ax.set_xlabel("Semantic Separation")
    ax.set_ylabel("")
    ax.set_ylim(base.min() - 0.55, base.max() + 0.55)
    ax.set_xlim(-0.01, max(0.27, max(upper_values) + 0.055))

    scenario_handles = [
        Line2D([], [], color=analysis.SCENARIO_COLOUR[s], marker="o",
               ls="", markersize=4.8, label=s)
        for s in SCENARIOS
    ]
    ax.legend(
        handles=scenario_handles,
        labels=SCENARIOS,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.12),
        fontsize=p * 0.78,
        ncol=2,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    # Right. Semantic specificity, encoder sensitivity.
    ax = axes[0, 1]
    panel(ax, p)
    ax.axvline(0, color=analysis.MUTED, lw=0.9, ls="--")

    encoder_offsets = {"minilm": +0.12, "mpnet": -0.12}
    highs = []
    lows = []

    for encoder in ("minilm", "mpnet"):
        part = (
            e1_specificity(encoder)
            .set_index("contrast")
            .reindex(CONTRASTS)
        )
        est = part["specificity"].to_numpy(dtype=float)
        low = part["low"].to_numpy(dtype=float)
        high = part["high"].to_numpy(dtype=float)
        highs.extend(high.tolist())
        lows.extend(low.tolist())
        y = base + encoder_offsets[encoder]
        colour, marker = ENCODER_STYLE[encoder]

        ax.errorbar(
            est,
            y,
            xerr=np.vstack([est - low, high - est]),
            fmt=marker,
            color=colour,
            ecolor=colour,
            markersize=4.8,
            elinewidth=1.35,
            lw=1.35,
            capsize=0,
            label=ENCODERS[encoder],
            zorder=3,
        )

        for hi, yi, value in zip(high, y, est):
            ax.text(
                hi + 0.003,
                yi,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=p * 0.58,
                fontweight="bold",
                color=colour,
                clip_on=False,
            )

    # The left panel already names the six contrasts.
    ax.set_yticks(base, [""] * len(base))
    ax.set_xlabel("Semantic Specificity")
    ax.set_ylabel("")
    ax.set_ylim(base.min() - 0.55, base.max() + 0.55)
    xlo = min(-0.03, min(lows) - 0.012)
    xhi = max(0.135, max(highs) + 0.025)
    ax.set_xlim(xlo, xhi)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.12),
        fontsize=p * 0.80,
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    save(fig, "semantic_age", png)


# ---------------------------------------------------------------------
# Experiment 2: semantic drift across turns
# ---------------------------------------------------------------------

def e2_summary(encoder="minilm") -> pd.DataFrame:
    d = read_semantic(encoder, "e2_summary").copy()
    required = {
        "axis",
        "n",
        "drift_12",
        "drift_12_low",
        "drift_12_high",
        "drift_13",
        "drift_13_low",
        "drift_13_high",
        "method",
        "model",
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            f"{encoder}/e2_summary.csv is missing {sorted(missing)}. "
            f"Columns are {list(d.columns)}"
        )
    return d


def model_display(raw: str) -> str:
    return analysis.NAME.get(raw, raw)


def draw_drift(png=False):
    """Experiment 2: drift by attack method and by model."""
    d = e2_summary("minilm")

    fig, axes, p = layout(1, 2, width=14.7, height=7.2)

    # Left. Attack method.
    ax = axes[0, 0]
    panel(ax, p)
    method_rows = d[d["axis"].eq("method")].set_index("method")

    # Fixed text offsets keep the numerically close T2 estimates legible.
    method_dy = {
        "Emotional Pushback": +10,
        "Role Play": 0,
        "Purpose Reverse": -10,
    }

    for method in METHODS:
        if method not in method_rows.index:
            raise ValueError(f"Missing method row in e2_summary.csv: {method}")

        row = method_rows.loc[method]
        values = [0.0, float(row["drift_12"]), float(row["drift_13"])]
        lows = [0.0, float(row["drift_12_low"]), float(row["drift_13_low"])]
        highs = [0.0, float(row["drift_12_high"]), float(row["drift_13_high"])]
        colour, marker, linestyle = METHOD_STYLE[method]
        x = np.arange(3)

        ax.plot(
            x, values,
            color=colour,
            marker=marker,
            ls=linestyle,
            lw=2.0,
            ms=5.5,
            label=method,
            zorder=3,
        )
        ax.errorbar(
            x[1:], values[1:],
            yerr=np.vstack([
                np.asarray(values[1:]) - np.asarray(lows[1:]),
                np.asarray(highs[1:]) - np.asarray(values[1:]),
            ]),
            fmt="none",
            ecolor=colour,
            elinewidth=1.2,
            capsize=0,
            zorder=2,
        )

        # T2 and T3 values; the T2 offsets are staggered by method.
        ax.annotate(
            f"{values[1]:.3f}", (1, values[1]),
            xytext=(7, method_dy[method]), textcoords="offset points",
            va="center", ha="left", fontsize=p * 0.58,
            color=colour, fontweight="bold",
        )
        ax.annotate(
            f"{values[2]:.3f}", (2, values[2]),
            xytext=(7, 0), textcoords="offset points",
            va="center", ha="left", fontsize=p * 0.62,
            color=colour, fontweight="bold",
        )

    ax.set_xticks(range(3), ["T1", "T2", "T3"])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Semantic Drift")
    ax.set_xlim(-0.08, 2.18)
    ax.set_ylim(-0.02, max(0.58, float(d["drift_13_high"].max()) + 0.05))
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.14),
        fontsize=p * 0.78,
        ncol=2,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    # Right. Model.
    ax = axes[0, 1]
    panel(ax, p)
    model_rows = d[d["axis"].eq("model")].copy()
    model_rows["display"] = model_rows["model"].astype(str).map(model_display)
    model_rows = model_rows.set_index("display")

    t2_values = []
    t3_values = []
    model_names = []

    for model in analysis.ORDER:
        if model not in model_rows.index:
            raise ValueError(
                f"Missing model row in e2_summary.csv after display-name mapping: {model}"
            )
        row = model_rows.loc[model]
        values = [0.0, float(row["drift_12"]), float(row["drift_13"])]
        ax.plot(
            range(3), values,
            color=analysis.COLOUR[model],
            marker=analysis.MARKER[model],
            lw=analysis.LINEWIDTH,
            ms=analysis.MARKERSIZE,
            label=model,
            zorder=3,
        )
        model_names.append(model)
        t2_values.append(values[1])
        t3_values.append(values[2])

    # Stagger both label columns and connect labels back to their points.
    t2_text = _stagger(t2_values, min_gap=0.022)
    t3_text = _stagger(t3_values, min_gap=0.022)

    for model, y2, y2_text, y3, y3_text in zip(
        model_names, t2_values, t2_text, t3_values, t3_text
    ):
        colour = analysis.COLOUR[model]
        ax.plot([1.00, 1.055], [y2, y2_text], color=colour, lw=0.75, alpha=0.80)
        ax.text(
            1.07, y2_text, f"{y2:.3f}",
            va="center", ha="left", fontsize=p * 0.51,
            color=colour, fontweight="bold",
        )
        ax.plot([2.00, 2.055], [y3, y3_text], color=colour, lw=0.75, alpha=0.80)
        ax.text(
            2.07, y3_text, f"{y3:.3f}",
            va="center", ha="left", fontsize=p * 0.54,
            color=colour, fontweight="bold",
        )

    ax.set_xticks(range(3), ["T1", "T2", "T3"])
    ax.set_xlabel("Turn")
    ax.set_ylabel("")
    ymax = max(
        0.58,
        max(t2_text) + 0.04,
        max(t3_text) + 0.04,
        float(d["drift_13_high"].max()) + 0.03,
    )
    ax.set_ylim(-0.02, ymax)
    ax.set_xlim(-0.08, 2.35)

    handles = [
        Line2D(
            [], [], color=analysis.COLOUR[m], marker=analysis.MARKER[m],
            lw=analysis.LINEWIDTH, ms=analysis.MARKERSIZE, label=m
        )
        for m in analysis.ORDER
    ]
    ax.legend(
        handles=handles,
        labels=analysis.ORDER,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.14),
        fontsize=p * 0.68,
        ncol=2,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    save(fig, "semantic_drift", png)


# ---------------------------------------------------------------------
# Experiment 2: relation between semantic drift and safety outcome
# ---------------------------------------------------------------------

def e2_transitions(encoder: str) -> pd.DataFrame:
    d = read_semantic(encoder, "e2_transitions").copy()
    required = {
        "transition",
        "drift_13",
        "drift_low",
        "drift_high",
        "n",
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            f"{encoder}/e2_transitions.csv is missing {sorted(missing)}. "
            f"Columns are {list(d.columns)}"
        )
    return ordered(d, "transition", TRANSITIONS)


def e2_adjusted(encoder: str) -> tuple[float, float, float]:
    d = read_semantic(encoder, "e2_adjusted").copy()
    required = {"beta", "low", "high"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            f"{encoder}/e2_adjusted.csv is missing {sorted(missing)}. "
            f"Columns are {list(d.columns)}"
        )
    row = d.iloc[0]
    return float(row["beta"]), float(row["low"]), float(row["high"])


def draw_outcomes(png=False):
    """Experiment 2: drift by action transition and adjusted defect association."""
    fig, axes, p = layout(1, 2, width=14.4, height=6.7)
    base = np.arange(len(TRANSITIONS))[::-1] * 1.10

    # Left. Turn-3 drift by action transition.
    ax = axes[0, 0]
    panel(ax, p)
    encoder_offsets = {"minilm": +0.11, "mpnet": -0.11}
    lows_all = []
    highs_all = []

    for encoder in ("minilm", "mpnet"):
        d = (
            e2_transitions(encoder)
            .set_index("transition")
            .reindex(TRANSITIONS)
        )
        est = d["drift_13"].to_numpy(dtype=float)
        low = d["drift_low"].to_numpy(dtype=float)
        high = d["drift_high"].to_numpy(dtype=float)
        lows_all.extend(low.tolist())
        highs_all.extend(high.tolist())
        y = base + encoder_offsets[encoder]
        colour, marker = ENCODER_STYLE[encoder]

        ax.errorbar(
            est, y,
            xerr=np.vstack([est - low, high - est]),
            fmt=marker,
            color=colour,
            ecolor=colour,
            markersize=4.9,
            elinewidth=1.35,
            lw=1.35,
            capsize=0,
            label=ENCODERS[encoder],
            zorder=3,
        )
        for hi, yi, value in zip(high, y, est):
            ax.text(
                hi + 0.004, yi, f"{value:.3f}",
                va="center", ha="left",
                fontsize=p * 0.58,
                fontweight="bold",
                color=colour,
                clip_on=False,
            )

    ax.set_yticks(base, [TRANSITION_LABEL[x] for x in TRANSITIONS])
    ax.set_xlabel("Semantic Drift")
    ax.set_ylabel("")
    ax.set_ylim(base.min() - 0.50, base.max() + 0.50)
    ax.set_xlim(min(lows_all) - 0.02, max(highs_all) + 0.04)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.15),
        fontsize=p * 0.80,
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    # Right. Adjusted similarity association.
    ax = axes[0, 1]
    panel(ax, p)
    ax.axvline(0, color=analysis.MUTED, lw=0.9, ls="--")
    ypos = {"minilm": 1.0, "mpnet": 0.0}
    lowers = []
    uppers = []

    for encoder in ("minilm", "mpnet"):
        est, low, high = e2_adjusted(encoder)
        lowers.append(low)
        uppers.append(high)
        colour, marker = ENCODER_STYLE[encoder]
        y = ypos[encoder]
        ax.errorbar(
            [est], [y],
            xerr=[[est - low], [high - est]],
            fmt=marker,
            color=colour,
            ecolor=colour,
            markersize=5.3,
            elinewidth=1.4,
            lw=1.4,
            capsize=0,
            zorder=3,
        )
        # Put the estimate after the right CI endpoint to keep the bar unobscured.
        ax.text(
            high + 0.003, y, f"{est:+.3f}",
            va="center", ha="left",
            fontsize=p * 0.70,
            color=colour,
            fontweight="bold",
            clip_on=False,
        )

    ax.set_yticks([1.0, 0.0], ["MiniLM", "MPNet"])
    ax.set_ylim(-0.65, 1.65)
    ax.set_xlabel("Adjusted Similarity Difference\nWhen Turn 3 Is Defective")
    ax.set_ylabel("")
    ax.set_xlim(min(lowers) - 0.018, max(uppers) + 0.018)

    save(fig, "semantic_outcomes", png)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

BUILDERS = {
    "age": draw_age,
    "drift": draw_drift,
    "outcomes": draw_outcomes,
}


def main():
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
        help="Also export 160-dpi PNG previews.",
    )
    args = parser.parse_args()

    jobs = BUILDERS
    if args.only:
        jobs = {name: fn for name, fn in jobs.items() if name in args.only}

    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"Semantic figures -> {FIGURES.relative_to(analysis.ROOT)}/")

    for name, builder in jobs.items():
        builder(args.png)


if __name__ == "__main__":
    main()
