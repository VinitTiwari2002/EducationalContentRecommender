"""Generate the workplan Gantt chart for Chapter 3 (§3.8).

Produces Draft-Report/figures/fig_3_4_workplan_gantt.png.

Timeline: 2026-05-10 (project start) → 2026-09-28 (CM3070 final submission).
Twenty weeks total, split into completed phases (rendered with hatch fill)
and remaining phases (solid fill). Bars are per-milestone durations;
dependencies drawn as thin grey arrows between the end of a predecessor
and the start of its successor. Today's date (2026-08-18) is marked with
a vertical grey dashed line; the 28 September submission with a red dashed
line.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Full project span.
START = date(2026, 5, 10)
TODAY = date(2026, 8, 18)
SUBMIT = date(2026, 9, 28)


# (label, start_offset_days, duration_days, category, dependency_indices, done)
# Categories: 'setup', 'engineering', 'report', 'polish', 'submit'.
# Offsets are days from the 10 May start; the whole plan must fit
# in 141 days (10 May → 28 Sep inclusive). The `done` flag marks
# milestones that finish on or before TODAY (day 100 = 18 Aug).
# Draft-report submission is 19 August (offset 101), so Week 15 is a
# two-day window before the forward-engineering phase begins.
MILESTONES = [
    # Retrospective phases (Weeks 1–14, days 0–100).
    ("Project setup + proposal",                       0, 21, "setup",       [], True),
    ("Data ingestion + EDA + baselines",              21, 28, "engineering", [0], True),
    ("Preliminary report writing + submission",       49,  7, "report",      [1], True),
    ("Post-feedback core implementation",             56, 28, "engineering", [2], True),
    ("Advanced experiments (ALS, gate, tuning, CV)",  84, 16, "engineering", [3], True),
    # Forward phases (Weeks 15–20, days 100–141).
    ("Draft report submission (19 Aug)",             100,  2, "report",      [4], False),
    ("FastAPI service scaffold",                     102,  7, "engineering", [5], False),
    ("Streamlit dashboard MVP",                      109,  7, "engineering", [6], False),
    ("Regenerate figures + robustness runs",         116,  7, "engineering", [7], False),
    ("Final report writing + peer review",           123,  9, "report",      [8], False),
    ("Video (3–5 min MP4) + PDF polish",             132,  7, "polish",      [9], False),
    ("Submit final report (28 Sep)",                 141,  1, "submit",      [10], False),
]

CATEGORY_COLOURS = {
    "setup":       "#7f7f7f",
    "engineering": "#3d6fa8",
    "report":      "#c8985a",
    "polish":      "#5a9c68",
    "submit":      "#b2432b",
}


def _add_dependency_arrow(ax, x_from: float, x_to: float, y_from: float, y_to: float) -> None:
    ax.annotate(
        "",
        xy=(x_to, y_to),
        xytext=(x_from, y_from),
        arrowprops=dict(
            arrowstyle="->",
            color="#888",
            lw=0.7,
            connectionstyle="angle,angleA=0,angleB=90,rad=0",
        ),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    n = len(MILESTONES)
    for i, (label, offset, duration, category, deps, done) in enumerate(MILESTONES):
        start_dt = START + timedelta(days=offset)
        end_dt = start_dt + timedelta(days=duration)
        colour = CATEGORY_COLOURS[category]
        y = n - 1 - i  # top-down layout
        ax.barh(
            y,
            duration,
            left=mdates.date2num(start_dt),
            height=0.6,
            color=colour,
            edgecolor="#333",
            linewidth=0.5,
            hatch="//" if done else None,
            alpha=0.55 if done else 1.0,
        )
        ax.text(
            mdates.date2num(end_dt) + 0.8,
            y,
            label,
            va="center",
            ha="left",
            fontsize=8.5,
        )
        for dep_idx in deps:
            _, dep_offset, dep_duration, _, _, _ = MILESTONES[dep_idx]
            dep_end = START + timedelta(days=dep_offset + dep_duration)
            dep_y = n - 1 - dep_idx
            _add_dependency_arrow(
                ax,
                mdates.date2num(dep_end),
                mdates.date2num(start_dt),
                dep_y,
                y,
            )

    # Submission marker.
    ax.axvline(
        mdates.date2num(SUBMIT),
        color="#b2432b",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
    )
    ax.text(
        mdates.date2num(SUBMIT) + 0.4,
        n - 0.3,
        "  28 Sep\nfinal submission",
        fontsize=7.5,
        color="#b2432b",
        ha="left",
        va="top",
    )

    # X axis: date ticks every three weeks (7 tick marks for 20 weeks + submit).
    tick_dates = [START + timedelta(days=21 * i) for i in range(8)]
    ax.set_xticks([mdates.date2num(d) for d in tick_dates])
    ax.set_xticklabels([d.strftime("%d %b") for d in tick_dates], fontsize=8.5)
    ax.set_xlim(
        mdates.date2num(START) - 2,
        mdates.date2num(SUBMIT) + 22,  # trailing room for row labels
    )

    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLOURS["setup"], label="Setup"),
        mpatches.Patch(color=CATEGORY_COLOURS["engineering"], label="Engineering"),
        mpatches.Patch(color=CATEGORY_COLOURS["report"], label="Report writing"),
        mpatches.Patch(color=CATEGORY_COLOURS["polish"], label="Polish / video"),
        mpatches.Patch(color=CATEGORY_COLOURS["submit"], label="Submission"),
        mpatches.Patch(facecolor="lightgrey", hatch="//", edgecolor="#333", label="Completed"),
    ]
    # Legend below the plot in a single horizontal row so it cannot
    # overlap with any bar label (previously collided with the bottom
    # forward-milestone rows).
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=8,
        frameon=False,
        ncol=6,
    )

    ax.set_title(
        "Workplan Gantt: 10 May 2026 → 28 September 2026 (20 weeks). "
        "Hatched bars are completed as of the draft-report submission.",
        fontsize=10,
        loc="left",
    )

    fig.tight_layout()

    out = Path(__file__).resolve().parents[1] / "Draft-Report" / "figures" / "fig_3_4_workplan_gantt.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
