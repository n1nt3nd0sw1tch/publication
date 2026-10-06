"""Experiment 2 retention figures from the frozen Notebook 19 tables.

Run: python scripts/figuremulti.py [--set main|supplement|all]
Requires the latest Notebook 19 exports.
No inference calls, classification, bootstrap reruns, or changes to source data.
Style and model/scenario colours are imported from figuresafe and analysis.

The default figure set is deliberately small. It prioritises the two retention
questions used in the thesis: whether an established boundary survives, and
whether the three matched attack methods differ. Dense diagnostic heatmaps and
plots that duplicate appendix tables are retired automatically.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import analysis
import figuresafe as house

MODELS = list(analysis.ORDER) + [analysis.MACRO]
MODEL_LABELS = list(analysis.ORDER) + ["Macro-Average"]
STRATA = ["Age Restricted", "Harmful"]
METHOD_RADAR_STRATA = ["Age Restricted", "Harmful"]
MEMORY_RADAR_STRATA = ["Age Restricted", "Harmful"]
TURNS = ["Turn 1", "Turn 2", "Turn 3"]
TURN_SHORT = {turn: f"T{i}" for i, turn in enumerate(TURNS, 1)}
METHODS = ["Emotional Pushback", "Purpose Reverse", "Role Play"]

RADAR_STRATUM_COLOUR = {
    "Age Restricted": analysis.SCENARIO_COLOUR["Age Restricted"],
    "Harmful": analysis.SCENARIO_COLOUR["Harmful"],
}
RADAR_STRATUM_MARKER = {
    "Age Restricted": "s",
    "Harmful": "D",
}

# Old outputs are removed so the folder contains only the lean figure set.
RETIRED = [
    "dialogue_outcomes",
    "dialogue_defects",
    "dialogue_memory",
    "dialogue_adjusted",
    "dialogue_ladder",
    "dialogue_directional",
    "dialogue_failure",
    "dialogue_first_break",
    "dialogue_methods_age_restricted",
    "dialogue_methods_harmful",
    "dialogue_changes_age_restricted",
    "dialogue_changes_harmful",
    "dialogue_age_control",
    "dialogue_age_age_restricted",
    "dialogue_age_harmful",
    "dialogue_memory_gap",
    "dialogue_domains",
    "dialogue_trajectories_age_restricted",
    "dialogue_trajectories_harmful",
    "dialogue_failure_age_restricted",
    "dialogue_failure_harmful",
]


class Figures:
    def __init__(self, tables, output, png=False, preview=True):
        self.tables, self.output, self.png = tables, output, png
        self.output.mkdir(parents=True, exist_ok=True)
        self.preview = preview
        self.manifest = []

    def clean_retired(self):
        for name in RETIRED:
            for suffix in (".pdf", ".png"):
                (self.output / f"{name}{suffix}").unlink(missing_ok=True)

    def read(self, name):
        paths = [self.tables / tier / f"{name}.csv" for tier in ("main", "supplement")]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing {name}.csv. Run the latest Notebook 19 first.")
        return pd.read_csv(path)

    def layout(self, rows=1, cols=2, width=13.5, height=6.4):
        plt.rcParams.update(analysis.STYLE)
        points = house.styled(1.0, width, label_points=9.0)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(width, height),
            squeeze=False,
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=.12, h_pad=.12, wspace=.12, hspace=.16)
        return fig, axes, points

    def save(self, fig, key, tier, caption, source):
        forests = [ax for ax in fig.axes if getattr(ax, "_multi_forest", False)]
        if forests:
            limits = [ax.get_xlim() for ax in forests]
            shared = (min(v[0] for v in limits), max(v[1] for v in limits))
            for ax in forests:
                ax.set_xlim(shared)

        name = f"dialogue_{key}"
        target = self.output / f"{name}.pdf"
        for _ in range(3):
            handle, temporary = tempfile.mkstemp(suffix=".pdf", dir=self.output)
            os.close(handle)
            fig.savefig(temporary, bbox_inches="tight")
            if Path(temporary).read_bytes()[-1024:].rstrip().endswith(b"%%EOF"):
                os.replace(temporary, target)
                break
            Path(temporary).unlink(missing_ok=True)
        else:
            raise OSError(f"Could not write a complete PDF after three attempts: {target}")

        if self.png:
            fig.savefig(self.output / f"{name}.png", dpi=120, bbox_inches="tight")

        self.manifest.append(
            dict(figure=name, tier=tier, source=source, caption=caption)
        )
        plt.close(fig)
        print(f"  {name}.pdf")

    def close(self):
        if self.preview and self.manifest:
            pdfunite = shutil.which("pdfunite")
            if pdfunite:
                target = self.output / "dialogue_preview.pdf"
                handle, temporary = tempfile.mkstemp(suffix=".pdf", dir=self.output)
                os.close(handle)
                Path(temporary).unlink()
                inputs = [self.output / f"{item['figure']}.pdf" for item in self.manifest]
                subprocess.run([pdfunite, *map(str, inputs), temporary], check=True)
                os.replace(temporary, target)
            else:
                print("  dialogue_preview.pdf skipped: pdfunite is unavailable")

        (self.output / "captions.json").write_text(
            json.dumps(self.manifest, indent=2) + "\n"
        )


def subset(frame, **filters):
    for column, value in filters.items():
        frame = frame[frame[column].eq(value)]
    return frame


def model_frame(frame):
    if frame.Model.duplicated().any():
        raise ValueError("A figure selection contains more than one row per model")
    result = frame.set_index("Model").reindex(MODELS)
    if result.isna().all(axis=1).any():
        raise ValueError("Missing model in figure selection")
    return result


def forest(ax, frame, estimate, low, high, title, points, show_labels=True):
    ax._multi_forest = True
    data = model_frame(frame)
    values, lower, upper = (
        data[column].to_numpy(float) for column in (estimate, low, high)
    )
    if not np.isfinite(np.r_[values, lower, upper]).all() or (lower > upper).any():
        raise ValueError(f"Invalid interval in {title}")

    house.panel(ax, title, points)
    ax.axvline(0, color=analysis.MUTED, linewidth=.8)

    for y, (model, value, lo, hi) in enumerate(zip(MODELS, values, lower, upper)):
        colour = analysis.COLOUR.get(model, "#222222")
        ax.plot([lo, hi], [y, y], color=colour, lw=2)
        ax.plot(
            value,
            y,
            marker="D" if model == analysis.MACRO else "o",
            color=colour,
            markersize=6,
        )
        ax.text(
            1.025,
            y,
            f"{value:+.1f}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=points * .83,
            color=colour,
            fontweight="bold",
            clip_on=False,
        )

    ax.set_yticks(
        range(7),
        MODEL_LABELS if show_labels else [""] * 7,
        fontsize=points * .78,
    )
    ax.set_ylim(6.65, -.65)
    span = max(upper.max() - lower.min(), 10)
    ax.set_xlim(
        min(lower.min(), 0) - span * .07,
        max(upper.max(), 0) + span * .07,
    )
    ax.axhline(5.5, color=analysis.MUTED, alpha=.35, lw=.7)
    ax.tick_params(axis="y", length=0)


def radar_panel(ax, title, points, rlabel_position=198):
    ax.set_facecolor(house.PANEL_FILL)
    ax.grid(color=analysis.MUTED, linewidth=.6, alpha=.25)
    ax.set_axisbelow(True)
    ax.spines["polar"].set_color(analysis.MUTED)
    ax.spines["polar"].set_linewidth(.7)
    ax.set_title(title, pad=points * 1.15, color="black")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(
        ["25", "50", "75", ""],
        fontsize=points * .61,
        color=analysis.MUTED,
    )
    ax.set_rlabel_position(rlabel_position)


def close_values(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite value in radar figure")
    if ((values < -.03) | (values > 100.03)).any():
        raise ValueError("Radar value outside 0 to 100")
    values = np.clip(values, 0, 100)
    return np.r_[values, values[0]]


def draw_methods_radar(figures):
    """Attack-method profiles, using Action Defect directly as reported in the thesis."""
    data = figures.read("dialogue_05_methods")
    data = data[
        data["Scenario Type"].isin(METHOD_RADAR_STRATA)
        & data["Method"].isin(METHODS)
        & data["Model"].isin(analysis.ORDER)
    ].copy()

    expected = len(METHOD_RADAR_STRATA) * len(METHODS) * len(analysis.ORDER)
    if len(data) != expected:
        raise ValueError(
            f"Method radar expects {expected} stratum-method-model rows; found {len(data)}"
        )
    if data.duplicated(["Scenario Type", "Method", "Model"]).any():
        raise ValueError("Duplicate stratum-method-model row in method radar")

    opening_spread = data.groupby(["Scenario Type", "Model"])[
        "Defect Turn 1 (%)"
    ].agg(lambda values: values.max() - values.min())
    if not np.allclose(opening_spread, 0, atol=.03):
        raise ValueError("Matched method branches do not share the same Turn 1 rate")

    labels = [
        "Emotional Pushback\nT2",
        "Emotional Pushback\nT3",
        "Purpose Reverse\nT2",
        "Purpose Reverse\nT3",
        "Role Play\nT2",
        "Role Play\nT3",
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    plt.rcParams.update(analysis.STYLE)
    points = house.styled(1.0, 13.5, label_points=8.7)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 9.6),
        subplot_kw={"projection": "polar"},
        squeeze=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=.15, h_pad=.18, wspace=.12, hspace=.17)

    for ax, model in zip(axes.flat, analysis.ORDER):
        radar_panel(ax, model, points)
        ax.set_xticks(angles, labels)
        ax.tick_params(axis="x", pad=8, labelsize=points * .62, colors="black")

        for stratum in METHOD_RADAR_STRATA:
            part = subset(data, Model=model, **{"Scenario Type": stratum})
            part = part.set_index("Method").reindex(METHODS)
            if part.isna().all(axis=1).any():
                raise ValueError(f"Missing method value for {stratum}: {model}")

            values = []
            for method in METHODS:
                row = part.loc[method]
                opening = float(row["Defect Turn 1 (%)"])
                values.extend(
                    [
                        opening + float(row["Defect Change Turn 2 (pp)"]),
                        opening + float(row["Defect Change Turn 3 (pp)"]),
                    ]
                )

            closed = close_values(values)
            colour = RADAR_STRATUM_COLOUR[stratum]
            marker = RADAR_STRATUM_MARKER[stratum]
            ax.plot(
                closed_angles,
                closed,
                color=colour,
                marker=marker,
                markersize=3.5,
                linewidth=1.6,
                label=stratum,
                zorder=3,
            )
            ax.fill(closed_angles, closed, color=colour, alpha=.035, zorder=2)

    handles = [
        Line2D(
            [],
            [],
            color=RADAR_STRATUM_COLOUR[stratum],
            marker=RADAR_STRATUM_MARKER[stratum],
            linewidth=1.6,
            markersize=4.5,
            label=stratum,
        )
        for stratum in METHOD_RADAR_STRATA
    ]
    fig.legend(
        handles,
        METHOD_RADAR_STRATA,
        ncol=2,
        fontsize=points * .74,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(.5, -.055),
        handlelength=2.0,
        handletextpad=.55,
        columnspacing=1.4,
    )

    figures.save(
        fig,
        "methods_radar",
        "main",
        "Action Defect under Emotional Pushback, Purpose Reverse and Role Play at the two "
        "pressed turns. Each panel is one model and each profile is one scenario stratum. "
        "The branches share the same opening reply, so Turn 1 is omitted from the spokes. "
        "Higher values indicate more replies whose action differs from the benchmark expectation.",
        "dialogue_05_methods",
    )


def draw_memory_radar(figures):
    """Safety Memory over dialogue turns for Age Restricted and Harmful scenarios."""
    data = figures.read("dialogue_03_memory")
    data = data[
        data["Scenario Type"].isin(MEMORY_RADAR_STRATA)
        & data["Model"].isin(analysis.ORDER)
    ].copy()

    expected = len(MEMORY_RADAR_STRATA) * len(analysis.ORDER)
    if len(data) != expected:
        raise ValueError(
            f"Memory radar expects {expected} stratum-model rows; found {len(data)}"
        )
    if data.duplicated(["Scenario Type", "Model"]).any():
        raise ValueError("Duplicate stratum-model row in memory radar")

    labels = ["T1", "T2", "T3"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    plt.rcParams.update(analysis.STYLE)
    points = house.styled(1.0, 13.5, label_points=8.7)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 9.6),
        subplot_kw={"projection": "polar"},
        squeeze=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=.15, h_pad=.18, wspace=.12, hspace=.17)

    for ax, model in zip(axes.flat, analysis.ORDER):
        radar_panel(ax, model, points, rlabel_position=58)
        ax.set_xticks(angles, labels)
        ax.tick_params(axis="x", pad=8, labelsize=points * .72, colors="black")

        part = (
            subset(data, Model=model)
            .set_index("Scenario Type")
            .reindex(MEMORY_RADAR_STRATA)
        )
        if part.isna().all(axis=1).any():
            raise ValueError(f"Missing Safety Memory stratum for {model}")

        for stratum in MEMORY_RADAR_STRATA:
            row = part.loc[stratum]
            values = [
                100.0,
                float(row["Held Turn 2 (%)"]),
                float(row["Held Both (%)"]),
            ]
            closed = close_values(values)
            colour = RADAR_STRATUM_COLOUR[stratum]
            marker = RADAR_STRATUM_MARKER[stratum]
            ax.plot(
                closed_angles,
                closed,
                color=colour,
                marker=marker,
                markersize=4.0,
                linewidth=1.9,
                label=stratum,
                zorder=3,
            )
            ax.fill(
                closed_angles,
                closed,
                color=colour,
                alpha=.11 if stratum == "Age Restricted" else .09,
                zorder=2,
            )

    handles = [
        Line2D(
            [],
            [],
            color=RADAR_STRATUM_COLOUR[stratum],
            marker=RADAR_STRATUM_MARKER[stratum],
            linewidth=1.9,
            markersize=4.8,
            label=stratum,
        )
        for stratum in MEMORY_RADAR_STRATA
    ]
    fig.legend(
        handles,
        MEMORY_RADAR_STRATA,
        ncol=2,
        fontsize=points * .76,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(.5, -.055),
        handlelength=2.0,
        handletextpad=.55,
        columnspacing=1.5,
    )

    figures.save(
        fig,
        "memory_radar",
        "main",
        "Safety Memory for Age Restricted and Harmful scenarios. Turn 1 is the opening "
        "reference and is 100% by construction because this cohort contains dialogues that "
        "opened with Strong Refusal. Turn 2 is the share that still holds at the first pressed "
        "turn; Turn 3 is the share that held at both later turns. Higher values indicate "
        "stronger retention of the opening boundary.",
        "dialogue_03_memory",
    )


def draw_age(figures):
    data = figures.read("dialogue_04_age")
    fig, axes, points = figures.layout(2, 3, height=10.8)

    for row, stratum in enumerate(STRATA):
        for col, turn in enumerate(TURNS):
            forest(
                axes[row, col],
                subset(
                    data,
                    Measure="Strong Refusal",
                    Turn=turn,
                    **{"Scenario Type": stratum},
                ),
                "Gap (pp)",
                "Gap CI Lower",
                "Gap CI Upper",
                f"{stratum} ({TURN_SHORT[turn]})",
                points,
                show_labels=col == 0,
            )
            axes[row, col].set_xlabel("")

    fig.supxlabel(
        "Strong Refusal Age Gap: Minor - Age 18 (pp)",
        fontsize=points,
    )
    figures.save(
        fig,
        "age",
        "main",
        "Strong Refusal age gap for Age Restricted and Harmful scenarios. The gap is the mean "
        "Strong Refusal rate across the explicit minor ages minus the age-18 rate at the same "
        "turn. Whiskers are paired scenario-bootstrap 95% intervals.",
        "dialogue_04_age",
    )


def draw_roleplay(figures):
    data = figures.read("dialogue_06_roleplay")
    later_turns = ["Turn 2", "Turn 3"]
    fig, axes, points = figures.layout(2, 2, width=13.5, height=9.6)

    for row, age in enumerate(["Age 9", "Age 17"]):
        part = subset(
            data,
            Measure="Strong Refusal",
            Condition=age + " minus Control",
        )
        for col, turn in enumerate(later_turns):
            frame = part.rename(
                columns={
                    f"Change {turn} (pp)": "Estimate",
                    f"Change {turn} CI Lower": "Lower",
                    f"Change {turn} CI Upper": "Upper",
                }
            )
            forest(
                axes[row, col],
                frame,
                "Estimate",
                "Lower",
                "Upper",
                f"{age} ({TURN_SHORT[turn]})",
                points,
                show_labels=col == 0,
            )
            axes[row, col].set_xlabel("")

    fig.supxlabel(
        "Change in Strong Refusal Gap Against Neutral from Turn 1 (pp)",
        fontsize=points,
    )
    figures.save(
        fig,
        "roleplay",
        "main",
        "Role Play against the Neutral condition. Values show how the Strong Refusal gap "
        "between an explicit age and Neutral changes from the shared Turn-1 reference. "
        "Turn 1 is omitted because this change is zero by construction. Negative values "
        "indicate erosion of that age-conditioned gap. Whiskers are paired scenario-bootstrap "
        "95% intervals.",
        "dialogue_06_roleplay",
    )


def draw_boundary(figures):
    data = figures.read("dialogue_s14_boundary")
    fig, axes, points = figures.layout(1, 3, height=6.8)

    for col, (ax, turn) in enumerate(zip(axes.flat, TURNS)):
        forest(
            ax,
            subset(data, Measure="Strong Refusal", Turn=turn),
            "17 minus 18 (pp)",
            "CI Lower",
            "CI Upper",
            TURN_SHORT[turn],
            points,
            show_labels=col == 0,
        )
        ax.set_xlabel("")

    fig.supxlabel(
        "Strong Refusal: Age 17 - Age 18 (pp)",
        fontsize=points,
    )
    figures.save(
        fig,
        "boundary",
        "supplement",
        "Strong Refusal difference between ages 17 and 18 at each turn, with paired "
        "scenario-bootstrap 95% intervals. The cohort requires only those two ages and is "
        "therefore distinct from the broader minor-versus-age-18 comparison.",
        "dialogue_s14_boundary",
    )


MAIN = {
    "methods_radar": draw_methods_radar,
    "memory_radar": draw_memory_radar,
    "age": draw_age,
    "roleplay": draw_roleplay,
}
SUPPLEMENT = {
    "boundary": draw_boundary,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=analysis.TABLES)
    parser.add_argument("--output", type=Path, default=analysis.FIGURES / "dialogue")
    parser.add_argument("--set", choices=["main", "supplement", "all"], default="all")
    parser.add_argument("--only", nargs="+", choices=list(MAIN) + list(SUPPLEMENT))
    parser.add_argument("--png", action="store_true", help="Also export 120-dpi PNGs")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    jobs = (
        MAIN
        if args.set == "main"
        else SUPPLEMENT
        if args.set == "supplement"
        else MAIN | SUPPLEMENT
    )
    if args.only:
        jobs = {key: value for key, value in jobs.items() if key in args.only}
    if not jobs:
        parser.error("No figures match --set and --only")

    figures = Figures(args.tables, args.output, args.png, not args.no_preview)
    figures.clean_retired()
    try:
        for job in jobs.values():
            job(figures)
    finally:
        figures.close()


if __name__ == "__main__":
    main()
