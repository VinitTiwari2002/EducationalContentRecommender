"""Generate the three Chapter 3 design figures.

Run:
    .venv/bin/python scripts/generate_design_figures.py

Output:
    Final-Report/figures/fig_3_1_pipeline.png
    Final-Report/figures/fig_3_2_recommendation_example.png
    Final-Report/figures/fig_3_3_score_decomposition.png

Figures 3.2 and 3.3 pull real output from the persisted Hybrid recommender
(under data/processed/models/, produced by `python -m src.pipeline
--persist-models`). If those artefacts are missing, the script falls back
to illustrative numbers and prints a warning — the CI/reproducibility
smoke test can still run without a full pipeline execution.
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
from src.hybrid import HybridRecommender  # noqa: E402
from src.persistence import DEFAULT_MODELS_DIR, artefacts_exist, load_models  # noqa: E402

FIG_DIR = PROJECT_ROOT / "Final-Report" / "figures"
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
    """Pipeline + serving architecture, as delivered.

    Layout: (1) preprocessing chain, (2) persisted split, (3) two rows of
    recommenders (baselines + CF, then content/hybrid/gated + LambdaMART
    Stage 2), (4) evaluation harness, (5) persisted model artefacts,
    (6) service layer (FastAPI + Streamlit) inside a Docker wrapper.
    """
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_xlim(0, 17)
    ax.set_ylim(-0.8, 13.5)
    ax.axis("off")

    BOX_W = 3.0
    GAP = 0.4
    COL_X5 = [0.2 + i * (BOX_W + GAP) for i in range(5)]  # 5-col grid
    FULL_X = COL_X5[0]
    FULL_W = (COL_X5[4] + BOX_W) - FULL_X

    # 4-col grid for the recommender rows, centred under the 5-col header
    BOX_W4 = 3.5
    GAP4 = 0.5
    total4 = 4 * BOX_W4 + 3 * GAP4
    COL_X4 = [FULL_X + (FULL_W - total4) / 2 + i * (BOX_W4 + GAP4) for i in range(4)]

    # --- Row 1: preprocessing chain ----------------------------------------
    row1_y = 11.4
    row1_h = 1.1
    row1_blocks = [
        ("Learner interaction\nand assessment data", "#FFF3E0", "#E65100"),
        ("Ingest and validate\nseven source tables", "#E3F2FD", "#1565C0"),
        ("Build sparse\ninteraction matrix", "#E3F2FD", "#1565C0"),
        ("Split chronologically\ninto train and test", "#E3F2FD", "#1565C0"),
        ("Restrict candidates\nto enrolled courses", "#E3F2FD", "#1565C0"),
    ]
    for (label, fc, ec), x in zip(row1_blocks, COL_X5):
        _box(ax, x, row1_y, BOX_W, row1_h, label, fc=fc, ec=ec, fontsize=10)
    for i in range(4):
        _arrow(ax, COL_X5[i] + BOX_W, row1_y + row1_h / 2,
               COL_X5[i + 1], row1_y + row1_h / 2)

    # --- Row 2: persisted split artefact -----------------------------------
    row2_y = 10.0
    row2_h = 0.8
    _box(ax, FULL_X, row2_y, FULL_W, row2_h,
         "Persisted split: train/test matrices, per-user course-scoped candidates, "
         "outcome vector, cutoff date",
         fc="#F1F8E9", ec="#33691E", fontsize=10.5)
    centre_x = FULL_X + FULL_W / 2
    _arrow(ax, centre_x, row1_y, centre_x, row2_y + row2_h)

    # --- Rows 3+4: recommender bank ----------------------------------------
    # Layout: two internal rows of four boxes, wrapped in a "Recommender bank"
    # frame so only two arrows cross the row boundary (split -> bank -> eval).
    row3_y = 8.0
    row3_h = 1.3
    row4_y = 6.2
    row4_h = 1.3

    bank_pad = 0.35
    bank_x = min(COL_X4) - bank_pad
    bank_w = (max(COL_X4) + BOX_W4) - bank_x + bank_pad
    bank_bottom = row4_y - bank_pad
    bank_top = row3_y + row3_h + bank_pad
    bank_frame = FancyBboxPatch(
        (bank_x, bank_bottom),
        bank_w,
        bank_top - bank_bottom,
        boxstyle="round,pad=0.08",
        linewidth=1.2,
        edgecolor="#78909C",
        facecolor="#FAFAFA",
    )
    ax.add_patch(bank_frame)
    ax.text(bank_x + 0.1, bank_top - 0.05,
            "Recommender bank (each model fitted on train, scored on test)",
            fontsize=9.5, weight="bold", color="#455A64", va="top")

    row3_blocks = [
        ("Random\nrecommender", "#FFEBEE", "#B71C1C"),
        ("Popularity\nrecommender", "#FFEBEE", "#B71C1C"),
        ("SVD collaborative\nfiltering", "#E8EAF6", "#283593"),
        ("ALS implicit-feedback\ncollaborative filtering", "#E8EAF6", "#283593"),
    ]
    for (label, fc, ec), x in zip(row3_blocks, COL_X4):
        _box(ax, x, row3_y, BOX_W4, row3_h, label, fc=fc, ec=ec, fontsize=10)

    row4_blocks = [
        ("Content-based\nfeature similarity", "#E8EAF6", "#283593"),
        ("Weighted Hybrid\n(CF + content + outcome)", "#FFF8E1", "#FF6F00"),
        ("Gated Hybrid\n(cold/warm switching)", "#FFF8E1", "#FF6F00"),
        ("LambdaMART two-stage\nreranker (Stage 2)", "#EDE7F6", "#4527A0"),
    ]
    for (label, fc, ec), x in zip(row4_blocks, COL_X4):
        _box(ax, x, row4_y, BOX_W4, row4_h, label, fc=fc, ec=ec, fontsize=10)

    # Dashed intra-bank arrow: Gated Hybrid -> LambdaMART (Stage-1 → Stage-2)
    gc = COL_X4[2] + BOX_W4
    lc = COL_X4[3]
    ax.annotate("", xy=(lc, row4_y + row4_h / 2),
                xytext=(gc, row4_y + row4_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#4527A0",
                                lw=1.4, mutation_scale=14, ls="dashed"))
    ax.text((gc + lc) / 2, row4_y + row4_h / 2 + 0.25,
            "Stage 1\n→ Stage 2",
            ha="center", va="bottom", fontsize=8, color="#4527A0", style="italic")

    # Single arrow: persisted split -> bank
    _arrow(ax, centre_x, row2_y, centre_x, bank_top)

    # --- Row 5: evaluation harness -----------------------------------------
    row5_y = 4.5
    row5_h = 1.1
    _box(ax, FULL_X, row5_y, FULL_W, row5_h,
         "Evaluation harness: Precision/Recall/NDCG/Hit-Rate/OWP@K, coverage + Gini, "
         "bootstrap CIs,\npaired-t with Bonferroni, 5-fold temporal CV, ablation, "
         "cold-start, fairness, robustness",
         fc="#F3E5F5", ec="#6A1B9A", fontsize=10.5)
    # Single arrow: bank -> evaluation harness
    _arrow(ax, centre_x, bank_bottom, centre_x, row5_y + row5_h)

    # --- Row 6: persisted model artefacts ----------------------------------
    row6_y = 3.0
    row6_h = 0.8
    _box(ax, FULL_X, row6_y, FULL_W, row6_h,
         "Persisted models: models.joblib · serving_context.joblib · "
         "manifest.json · reranker.joblib",
         fc="#F1F8E9", ec="#33691E", fontsize=10.5)
    _arrow(ax, centre_x, row5_y, centre_x, row6_y + row6_h)

    # --- Row 7: service layer (FastAPI + Streamlit) inside a Docker wrapper -
    row7_y = 0.4
    row7_h = 1.7
    # Docker wrapper (spanning both service boxes)
    docker_pad = 0.25
    docker_box = FancyBboxPatch(
        (FULL_X - docker_pad, row7_y - docker_pad),
        FULL_W + 2 * docker_pad, row7_h + 2 * docker_pad,
        boxstyle="round,pad=0.06",
        linewidth=1.5, edgecolor="#0277BD",
        facecolor="#E1F5FE", alpha=0.35,
    )
    ax.add_patch(docker_box)
    ax.text(FULL_X + docker_pad, row7_y + row7_h + docker_pad - 0.05,
            "Two-stage Docker image", fontsize=9.5, weight="bold",
            color="#0277BD", va="top")

    # Two service boxes inside
    svc_w = (FULL_W - 0.6) / 2
    _box(ax, FULL_X + 0.05, row7_y + 0.05, svc_w, row7_h - 0.1,
         "FastAPI service\n/health · /recommend/{student_id} · /decompose/{student_id}/{item_id}",
         fc="#E0F2F1", ec="#00695C", fontsize=10.5)
    _box(ax, FULL_X + svc_w + 0.55, row7_y + 0.05, svc_w, row7_h - 0.1,
         "Streamlit transparency dashboard\nRecommendation Explorer · Fairness View · Ablation Comparison",
         fc="#E0F2F1", ec="#00695C", fontsize=10.5)
    _arrow(ax, centre_x, row6_y, centre_x, row7_y + row7_h + docker_pad + 0.03)
    # API -> Dashboard arrow (within the Docker box)
    api_right = FULL_X + 0.05 + svc_w
    dash_left = FULL_X + svc_w + 0.55
    ax.annotate("", xy=(dash_left, row7_y + row7_h / 2),
                xytext=(api_right, row7_y + row7_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#00695C",
                                lw=1.4, mutation_scale=14))

    # --- Legend ------------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor="#FFF3E0", edgecolor="#E65100", label="Raw input"),
        mpatches.Patch(facecolor="#E3F2FD", edgecolor="#1565C0", label="Preprocessing"),
        mpatches.Patch(facecolor="#F1F8E9", edgecolor="#33691E", label="Persisted artefact"),
        mpatches.Patch(facecolor="#FFEBEE", edgecolor="#B71C1C", label="Baseline recommender"),
        mpatches.Patch(facecolor="#E8EAF6", edgecolor="#283593", label="Personalised recommender"),
        mpatches.Patch(facecolor="#FFF8E1", edgecolor="#FF6F00", label="Hybrid ensemble"),
        mpatches.Patch(facecolor="#EDE7F6", edgecolor="#4527A0", label="Two-stage reranker"),
        mpatches.Patch(facecolor="#F3E5F5", edgecolor="#6A1B9A", label="Evaluation"),
        mpatches.Patch(facecolor="#E0F2F1", edgecolor="#00695C", label="Service layer"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=5,
        fontsize=9.5,
        frameon=False,
    )

    ax.set_title(
        "Figure 3.1 — Pipeline and serving architecture (as delivered)",
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
# Shared context for figures 3.2 and 3.3 — loads the persisted Hybrid, picks
# a real student, and returns their real top-10 output + decomposition.
# ---------------------------------------------------------------------------


def _prepare_hybrid_context() -> dict | None:
    """Load persisted models + pick a sample student + compute real top-10.

    Returns a dict with `hybrid`, `student_id`, `row`, `profile`, `top_k_cols`,
    `top_k_item_ids`, `activity_types`, `breakdown`. Returns None if the
    persisted artefacts are missing — callers fall back to illustrative
    numbers with a warning.
    """
    if not artefacts_exist(DEFAULT_MODELS_DIR):
        print(
            f"Warning: no persisted models under {DEFAULT_MODELS_DIR}. "
            "Falling back to illustrative numbers. Run "
            "`python -m src.pipeline --persist-models` for real output."
        )
        return None
    models, context, _manifest = load_models(DEFAULT_MODELS_DIR)
    hybrid = models.get("Hybrid")
    if not isinstance(hybrid, HybridRecommender):
        print("Warning: persisted models do not include a Hybrid; falling back.")
        return None

    print("Loading OULAD for figure context...")
    data = oulad.load()

    # Pick a student that is (i) in the loaded split, (ii) warm (many
    # clicks), (iii) has an assessment score to show, and (iv) has enough
    # candidates that a top-10 is well-defined. Then break ties
    # deterministically by taking the median from the sorted candidates.
    student_index = context.student_index
    counts = data.student_vle.groupby("id_student").size()
    counts = counts[counts.index.isin(student_index)]
    row_by_id = {int(s): int(r) for r, s in enumerate(student_index)}
    candidates_pool_size = np.array(
        [len(context.user_candidates[row_by_id[int(s)]]) for s in counts.index]
    )
    keep = (counts.to_numpy() >= 80) & (counts.to_numpy() <= 200) & (candidates_pool_size >= 20)
    keep_ids = counts.index[keep].tolist()
    if not keep_ids:
        keep_ids = counts.index.tolist()
    keep_ids = sorted(keep_ids)
    student_id = int(keep_ids[len(keep_ids) // 2])
    row = row_by_id[student_id]

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

    # Real top-10 from Hybrid, restricted to the student's course scope.
    candidates = context.user_candidates[row]
    top_k_cols = hybrid.recommend(row, k=10, candidates=candidates)
    top_k_cols_arr = np.asarray(top_k_cols, dtype=np.int64)
    top_k_item_ids = context.item_index[top_k_cols_arr]

    # Look up each recommended item's activity_type.
    vle_lookup = data.vle.drop_duplicates(subset=["id_site"]).set_index("id_site")["activity_type"]
    top_k_activity_types = [
        str(vle_lookup.get(int(iid), "unknown")) for iid in top_k_item_ids
    ]

    # Score breakdown over the candidate pool, filtered to the top-10 items
    # (matches what /recommend and Table 5.1 report).
    bd = hybrid.score_breakdown(user_row=row, items=np.asarray(candidates))
    order = [int(np.where(bd["items"] == c)[0][0]) for c in top_k_cols_arr]
    breakdown = {
        "cf": bd["cf_weighted"][order],
        "content": bd["content_weighted"][order],
        "outcome": bd["outcome_weighted"][order],
        "total": bd["total"][order],
    }

    return {
        "hybrid": hybrid,
        "student_id": student_id,
        "row": row,
        "profile": profile,
        "top_k_cols": top_k_cols_arr,
        "top_k_item_ids": top_k_item_ids.astype(int),
        "top_k_activity_types": top_k_activity_types,
        "breakdown": breakdown,
        "hybrid_weights": (hybrid.alpha, hybrid.beta, hybrid.gamma),
    }


# Cached across figure 3.2 and 3.3 calls (see __main__).
_HYBRID_CONTEXT: dict | None = None


def _get_or_build_context() -> dict | None:
    global _HYBRID_CONTEXT
    if _HYBRID_CONTEXT is None:
        _HYBRID_CONTEXT = _prepare_hybrid_context()
    return _HYBRID_CONTEXT


# ---------------------------------------------------------------------------
# Figure 3.2 — Recommendation example for a real OULAD student
# ---------------------------------------------------------------------------


def _pick_student(data) -> tuple[int, dict]:
    """Illustrative fallback when persisted models are unavailable."""
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
    ctx = _get_or_build_context()
    if ctx is not None:
        student_id = ctx["student_id"]
        p = ctx["profile"]
        real_items = list(zip(
            ctx["top_k_item_ids"].tolist(),
            ctx["top_k_activity_types"],
            ctx["breakdown"]["total"].tolist(),
        ))
    else:
        print("Loading OULAD for figure 2 (fallback path)...")
        data = oulad.load()
        student_id, p = _pick_student(data)
        real_items = None

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
    if real_items is not None:
        panel4_title = "(4) Top-10 ranked output\nfrom the fitted Hybrid recommender"
        item_labels = [f"item_{iid}" for iid, _, _ in real_items]
        activities = [act for _, act, _ in real_items]
        scores = np.array([total for _, _, total in real_items], dtype=float)
    else:
        panel4_title = "(4) Top-10 ranked output\n(illustrative — hybrid not yet trained)"
        rng = np.random.default_rng(seed=student_id)
        item_labels = [f"item_{int(rng.integers(1000, 9000))}" for _ in range(10)]
        scores = np.array(sorted(rng.uniform(0.45, 0.92, 10), reverse=True))
        activities = rng.choice(
            ["resource", "oucontent", "subpage", "url", "quiz", "forumng"], 10
        ).tolist()
    ax4.set_title(panel4_title, fontsize=11.5, weight="bold", loc="left", pad=10)

    # Table header
    header_y = 0.92
    ax4.text(0.00, header_y, "Rank", fontsize=10, weight="bold")
    ax4.text(0.13, header_y, "Item", fontsize=10, weight="bold")
    ax4.text(0.43, header_y, "Activity", fontsize=10, weight="bold")
    ax4.text(0.70, header_y, "Score", fontsize=10, weight="bold")
    ax4.axhline(y=header_y - 0.025, xmin=0, xmax=1, color="#888", lw=0.8)

    # Rows. Normalise bar widths against the observed max so the visual
    # scale is comparable to the illustrative version — real hybrid totals
    # occupy [0, ~1] under the min-max design.
    row_h = 0.082
    score_max = float(scores.max()) if len(scores) else 1.0
    for i, (lbl, scr, act) in enumerate(zip(item_labels, scores, activities)):
        y = header_y - 0.05 - i * row_h
        ax4.text(0.00, y, f"#{i+1}", fontsize=9.7, family="monospace")
        ax4.text(0.13, y, lbl, fontsize=9.7, family="monospace")
        ax4.text(0.43, y, act, fontsize=9.7, family="monospace")
        bar_x = 0.70
        bar_w = 0.20 * (scr / score_max if score_max > 0 else 0)
        ax4.add_patch(plt.Rectangle(
            (bar_x, y - 0.012), bar_w, 0.022,
            facecolor="#FF6F00", alpha=0.85,
        ))
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
    ctx = _get_or_build_context()
    if ctx is not None:
        # Use the top-6 of the real recommendation list; the full top-10 is
        # too crowded to render legibly at 12 x 6.5 inches.
        n_show = min(6, len(ctx["top_k_item_ids"]))
        items = [f"item_{int(iid)}" for iid in ctx["top_k_item_ids"][:n_show]]
        activity_types = ctx["top_k_activity_types"][:n_show]
        cf = np.asarray(ctx["breakdown"]["cf"][:n_show], dtype=float)
        content = np.asarray(ctx["breakdown"]["content"][:n_show], dtype=float)
        outcome = np.asarray(ctx["breakdown"]["outcome"][:n_show], dtype=float)
        totals = cf + content + outcome
        # Items are already in rank order from the recommender.
        ranks = np.arange(1, n_show + 1)
        title_suffix = (
            f" for student {ctx['student_id']}\n"
            f"(weights α={ctx['hybrid_weights'][0]}, "
            f"β={ctx['hybrid_weights'][1]}, "
            f"γ={ctx['hybrid_weights'][2]}; "
            "top-6 of the real top-10)"
        )
    else:
        rng = np.random.default_rng(seed=11391)
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
        title_suffix = "\n(illustrative; final weights tuned on validation fold)"

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
        "Figure 3.3 — Hybrid score decomposition for top-6 candidates" + title_suffix,
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
