"""Generate the three Chapter 3 design figures.

Run:
    .venv/bin/python scripts/generate_design_figures.py

Output:
    Preliminary-Report/figures/fig_3_1_pipeline.png
    Preliminary-Report/figures/fig_3_2_recommendation_example.png
    Preliminary-Report/figures/fig_3_3_score_decomposition.png

Pulls a real OULAD student for figure 2 to ground the worked example in actual
data. Figure 3 illustrates the hybrid score decomposition using realistic but
illustrative numbers (the hybrid model isn't built yet — the figure is a design
schematic, not a results plot).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import oulad  # noqa: E402

FIG_DIR = PROJECT_ROOT / "Preliminary-Report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 3.1 — Pipeline architecture
# ---------------------------------------------------------------------------


def _box(ax, x, y, w, h, label, fc="#E3F2FD", ec="#1565C0", fontsize=9, bold=False):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        wrap=True,
    )


def _arrow(ax, x0, y0, x1, y1, color="#37474F"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=14),
    )


def fig_pipeline() -> None:
    """Pipeline architecture: descriptive blocks, no code paths.

    Layout strategy:
      - 5 columns of width 3.2 across an x-range of [0, 17] with 0.4 gutters.
      - 4 rows: (1) raw input + preprocessing chain, (2) persisted artefact,
        (3) models, (4) evaluation harness. Each row sized to comfortably
        fit two lines of descriptive text per block.
    """
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 17)
    ax.set_ylim(-1.0, 9.0)
    ax.axis("off")

    BOX_W = 3.0
    GAP = 0.4
    COL_X = [0.2 + i * (BOX_W + GAP) for i in range(5)]  # 0.2, 3.6, 7.0, 10.4, 13.8

    # --- Row 1: input + preprocessing chain ---------------------------------
    row1_y = 7.0
    row1_h = 1.2
    row1_blocks = [
        ("Learner interaction\nand assessment data", "#FFF3E0", "#E65100"),
        ("Ingest and validate\nseven source tables", "#E3F2FD", "#1565C0"),
        ("Build sparse\ninteraction matrix", "#E3F2FD", "#1565C0"),
        ("Split chronologically\ninto train and test", "#E3F2FD", "#1565C0"),
        ("Restrict candidates\nto enrolled courses", "#E3F2FD", "#1565C0"),
    ]
    for (label, fc, ec), x in zip(row1_blocks, COL_X):
        _box(ax, x, row1_y, BOX_W, row1_h, label, fc=fc, ec=ec, fontsize=10)

    for i in range(4):
        x_end = COL_X[i] + BOX_W
        x_start = COL_X[i + 1]
        _arrow(ax, x_end, row1_y + row1_h / 2, x_start, row1_y + row1_h / 2)

    # --- Row 2: persisted artefact (spans all 5 columns) -------------------
    row2_y = 4.8
    row2_h = 1.0
    row2_x = COL_X[0]
    row2_w = (COL_X[4] + BOX_W) - row2_x
    _box(
        ax,
        row2_x,
        row2_y,
        row2_w,
        row2_h,
        "Cached training and test matrices, candidate pools, and split metadata",
        fc="#F1F8E9",
        ec="#33691E",
        fontsize=11,
    )
    # Down-arrow from Row 1 (centre) to Row 2 (top)
    centre_x = (row2_x + row2_w / 2)
    _arrow(ax, centre_x, row1_y, centre_x, row2_y + row2_h)

    # --- Row 3: models ------------------------------------------------------
    row3_y = 2.4
    row3_h = 1.4
    row3_blocks = [
        ("Random recommender", "#FFEBEE", "#B71C1C"),
        ("Popularity recommender", "#FFEBEE", "#B71C1C"),
        ("Collaborative filtering\nby matrix factorisation",
         "#E8EAF6", "#283593"),
        ("Content-based\nfeature similarity", "#E8EAF6", "#283593"),
        ("Hybrid ensemble with\noutcome weighting",
         "#FFF8E1", "#FF6F00"),
    ]
    for (label, fc, ec), x in zip(row3_blocks, COL_X):
        _box(ax, x, row3_y, BOX_W, row3_h, label, fc=fc, ec=ec, fontsize=10)

    # Down-arrows: persisted artefact -> each model (all 5 now connected)
    for x in COL_X:
        col_centre = x + BOX_W / 2
        _arrow(ax, col_centre, row2_y, col_centre, row3_y + row3_h)

    # --- Row 4: evaluation harness (spans all 5 columns) -------------------
    row4_y = 0.4
    row4_h = 1.0
    _box(
        ax,
        row2_x,
        row4_y,
        row2_w,
        row4_h,
        "Common evaluation harness reporting Precision, Recall, NDCG, and Hit Rate at K",
        fc="#F3E5F5",
        ec="#6A1B9A",
        fontsize=11,
    )
    # Up-into-harness arrows from each model
    for x in COL_X:
        col_centre = x + BOX_W / 2
        _arrow(ax, col_centre, row3_y, col_centre, row4_y + row4_h)

    # --- Legend ------------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor="#FFF3E0", edgecolor="#E65100", label="Raw input"),
        mpatches.Patch(facecolor="#E3F2FD", edgecolor="#1565C0", label="Preprocessing"),
        mpatches.Patch(facecolor="#F1F8E9", edgecolor="#33691E",
                       label="Persisted artefact"),
        mpatches.Patch(facecolor="#FFEBEE", edgecolor="#B71C1C",
                       label="Implemented baseline"),
        mpatches.Patch(facecolor="#E8EAF6", edgecolor="#283593",
                       label="Planned model"),
        mpatches.Patch(facecolor="#FFF8E1", edgecolor="#FF6F00",
                       label="Final hybrid"),
        mpatches.Patch(facecolor="#F3E5F5", edgecolor="#6A1B9A", label="Evaluation"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=7,
        fontsize=9.5,
        frameon=False,
    )

    ax.set_title(
        "Figure 3.1 — Pipeline architecture",
        fontsize=13,
        weight="bold",
        loc="left",
        pad=14,
    )

    out = FIG_DIR / "fig_3_1_pipeline.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3.2 — Recommendation example for a real OULAD student
# ---------------------------------------------------------------------------


def _pick_student(data) -> tuple[int, dict]:
    """Pick a student with a substantial interaction history for the example."""
    counts = data.student_vle.groupby("id_student").size()
    candidates = counts[(counts > 80) & (counts < 200)].index
    student_id = int(candidates[len(candidates) // 2])

    info = data.student_info[data.student_info["id_student"] == student_id].iloc[0]
    sa = data.student_assessment[data.student_assessment["id_student"] == student_id]
    svle = data.student_vle[data.student_vle["id_student"] == student_id]
    activity_types = (
        svle.merge(data.vle[["id_site", "activity_type"]], on="id_site")["activity_type"]
        .value_counts()
        .head(5)
    )

    profile = {
        "gender": info["gender"],
        "imd_band": info["imd_band"],
        "age_band": info["age_band"],
        "presentation": f"{info['code_module']}_{info['code_presentation']}",
        "n_interactions": int(len(svle)),
        "n_unique_items": int(svle["id_site"].nunique()),
        "mean_assessment_score": float(sa["score"].mean()) if len(sa) else float("nan"),
        "n_assessments": int(len(sa)),
        "top_activity_types": activity_types.to_dict(),
    }
    return student_id, profile


def fig_recommendation_example() -> None:
    print("Loading OULAD for figure 2...")
    data = oulad.load()
    student_id, p = _pick_student(data)

    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.1, 1.0, 1.4], wspace=0.75,
                          left=0.04, right=0.98, top=0.84, bottom=0.08)

    # Panel 1: Student profile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    ax1.set_title("(1) Input\nStudent profile", fontsize=11.5, weight="bold", loc="left", pad=10)
    lines = [
        f"Student ID: {student_id}",
        f"Presentation: {p['presentation']}",
        f"Gender: {p['gender']}",
        f"Age band: {p['age_band']}",
        f"IMD band: {p['imd_band']}",
        "",
        f"Interactions: {p['n_interactions']}",
        f"Unique items: {p['n_unique_items']}",
        "",
        f"Assessments: {p['n_assessments']}",
        f"Mean score: {p['mean_assessment_score']:.1f}",
    ]
    for i, line in enumerate(lines):
        ax1.text(0.0, 0.95 - i * 0.07, line, fontsize=10, va="top", family="monospace")

    # Panel 2: Feature extraction (activity-type histogram)
    ax2 = fig.add_subplot(gs[0, 1])
    types = list(p["top_activity_types"].keys())
    counts = list(p["top_activity_types"].values())
    bars = ax2.barh(types[::-1], counts[::-1], color="#1565C0", alpha=0.85)
    ax2.set_title(
        "(2) Feature extraction\nActivity-type profile",
        fontsize=11.5, weight="bold", loc="left", pad=10,
    )
    ax2.set_xlabel("Click count")
    for bar in bars:
        w = bar.get_width()
        ax2.text(
            w + max(counts) * 0.03, bar.get_y() + bar.get_height() / 2,
            f"{int(w)}", va="center", fontsize=9.5,
        )
    ax2.set_xlim(0, max(counts) * 1.18)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=9.5)

    # Panel 3: Model scoring
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.set_title(
        "(3) Model scoring\nfor candidate items",
        fontsize=11.5, weight="bold", loc="left", pad=10,
    )
    explanation = (
        "For each candidate item i in\n"
        "the user's presentation:\n"
        "\n"
        "  s(u,i) = α·s_CF(u,i)\n"
        "         + β·s_content(u,i)\n"
        "         + γ·s_outcome(i)\n"
        "\n"
        "s_CF      : SVD reconstruction\n"
        "s_content : cosine(item, profile)\n"
        "s_outcome : training-set\n"
        "            assessment\n"
        "            correlation\n"
        "\n"
        "α + β + γ = 1\n"
        "Tuned on validation fold."
    )
    ax3.text(0.0, 0.95, explanation, fontsize=9.5, va="top", family="monospace")

    # Panel 4: Top-10 ranked output rendered as a clean table
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis("off")
    ax4.set_title(
        "(4) Top-10 ranked output\n(illustrative — hybrid not yet built)",
        fontsize=11.5, weight="bold", loc="left", pad=10,
    )
    rng = np.random.default_rng(seed=student_id)
    item_labels = [f"item_{int(rng.integers(1000, 9000))}" for _ in range(10)]
    scores = sorted(rng.uniform(0.45, 0.92, 10), reverse=True)
    activities = rng.choice(
        ["resource", "oucontent", "subpage", "url", "quiz", "forumng"], 10
    )

    # Table header
    header_y = 0.92
    ax4.text(0.00, header_y, "Rank", fontsize=10, weight="bold")
    ax4.text(0.13, header_y, "Item", fontsize=10, weight="bold")
    ax4.text(0.43, header_y, "Activity", fontsize=10, weight="bold")
    ax4.text(0.70, header_y, "Score", fontsize=10, weight="bold")
    ax4.axhline(y=header_y - 0.025, xmin=0, xmax=1, color="#888", lw=0.8)

    # Rows
    row_h = 0.082
    for i, (lbl, scr, act) in enumerate(zip(item_labels, scores, activities)):
        y = header_y - 0.05 - i * row_h
        ax4.text(0.00, y, f"#{i+1}", fontsize=9.7, family="monospace")
        ax4.text(0.13, y, lbl, fontsize=9.7, family="monospace")
        ax4.text(0.43, y, act, fontsize=9.7, family="monospace")
        # Bar + numeric score
        bar_x = 0.70
        bar_w = 0.20 * scr
        ax4.add_patch(plt.Rectangle((bar_x, y - 0.012), bar_w, 0.022,
                                     facecolor="#FF6F00", alpha=0.85))
        ax4.text(bar_x + 0.21, y, f"{scr:.2f}", fontsize=9.5, family="monospace")

    # Inter-panel arrows. Routed along the top of the panels (just below the
    # subtitle row) so they cannot collide with any tick labels or bars
    # inside the plot area.
    for src_ax, dst_ax in [(ax1, ax2), (ax2, ax3), (ax3, ax4)]:
        src_pos = src_ax.get_position()
        dst_pos = dst_ax.get_position()
        # Arrow y is just above the panel top, in figure coords.
        y = src_pos.y1 + 0.015
        # Horizontal span between the panels in the gutter, with margin.
        x_start = src_pos.x1 + 0.008
        x_end = dst_pos.x0 - 0.008
        fig.patches.append(
            mpatches.FancyArrowPatch(
                (x_start, y),
                (x_end, y),
                arrowstyle="-|>",
                mutation_scale=22,
                color="#37474F",
                lw=1.8,
                transform=fig.transFigure,
            )
        )

    fig.suptitle(
        f"Figure 3.2 — Recommendation process for student {student_id} "
        f"(presentation {p['presentation']})",
        fontsize=13, weight="bold", x=0.04, ha="left", y=0.97,
    )

    out = FIG_DIR / "fig_3_2_recommendation_example.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return student_id


# ---------------------------------------------------------------------------
# Figure 3.3 — Hybrid score decomposition
# ---------------------------------------------------------------------------


def fig_score_decomposition() -> None:
    rng = np.random.default_rng(seed=11391)

    # 6 candidate items; each gets a (CF, content, outcome) contribution
    items = [f"item_{int(rng.integers(1000, 9000))}" for _ in range(6)]
    activity_types = ["resource", "oucontent", "subpage", "quiz", "forumng", "url"]
    cf = rng.uniform(0.20, 0.55, 6)
    content = rng.uniform(0.10, 0.40, 6)
    outcome = rng.uniform(0.05, 0.30, 6)
    totals = cf + content + outcome
    order = np.argsort(-totals)
    items = [items[i] for i in order]
    activity_types = [activity_types[i] for i in order]
    cf = cf[order]
    content = content[order]
    outcome = outcome[order]
    totals = totals[order]
    ranks = np.arange(1, len(items) + 1)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    y = np.arange(len(items))[::-1]

    p_cf = ax.barh(y, cf, color="#1565C0", alpha=0.9, label="s_CF (collaborative)")
    p_ct = ax.barh(y, content, left=cf, color="#33691E", alpha=0.9, label="s_content")
    p_ou = ax.barh(y, outcome, left=cf + content, color="#E65100", alpha=0.9, label="s_outcome")

    for i, total in enumerate(totals):
        ax.text(
            total + 0.015,
            y[i],
            f"total = {total:.2f}",
            va="center",
            fontsize=10,
            weight="bold",
        )

    labels = [f"#{r}  {it}  ({at})" for r, it, at in zip(ranks, items, activity_types)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, family="monospace")
    ax.set_xlabel("Score contribution")
    ax.set_xlim(0, max(totals) * 1.30)
    ax.set_title(
        "Figure 3.3 — Hybrid score decomposition for top-6 candidates\n"
        "(illustrative; final weights tuned on validation fold)",
        fontsize=12.5,
        weight="bold",
        loc="left",
        pad=12,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=10,
        frameon=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    ax.yaxis.tick_left()

    out = FIG_DIR / "fig_3_3_score_decomposition.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_pipeline()
    fig_recommendation_example()
    fig_score_decomposition()
    print("\nAll three design figures written to", FIG_DIR)
