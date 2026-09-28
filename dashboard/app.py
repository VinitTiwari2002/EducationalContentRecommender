"""Streamlit dashboard for the OULAD hybrid recommender.

Three pages sitting on top of the FastAPI service (`src.api`):

    1. Recommendation Explorer — per-student top-K with score decomposition.
    2. Fairness View          — per-attribute metric breakdown from the
                                 evaluation-time fairness audit.
    3. Ablation Comparison    — side-by-side top-K for Content vs Hybrid
                                 vs GatedHybrid on a single student.

Run with:
    # In one terminal: start the API
    uvicorn src.api:app --port 8000
    # In another: start the dashboard
    streamlit run dashboard/app.py

The dashboard is a *transparency* artefact — it exists so a marker can
query any student in the test set and see both the recommendations and
the reasons behind them, per §3.5 of the design chapter.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "evaluation"
API_URL = os.environ.get("RECSYS_API_URL", "http://localhost:8000")


# --------------------------------------------------------------------------- #
# Cached API calls.
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=60)
def api_health() -> dict:
    try:
        return httpx.get(f"{API_URL}/health", timeout=5.0).json()
    except httpx.HTTPError as exc:
        return {"status": "unreachable", "error": str(exc)}


@st.cache_data(ttl=300)
def api_recommend(student_id: int, k: int, model: str) -> dict:
    r = httpx.get(f"{API_URL}/recommend/{student_id}",
                  params={"k": k, "model": model}, timeout=15.0)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def api_decompose(student_id: int, item_id: int) -> dict:
    r = httpx.get(f"{API_URL}/decompose/{student_id}/{item_id}", timeout=15.0)
    r.raise_for_status()
    return r.json()


@st.cache_data
def load_fairness_csv() -> pd.DataFrame | None:
    path = EVAL_DIR / "fairness_audit.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_cold_start_csv() -> pd.DataFrame | None:
    path = EVAL_DIR / "cold_start_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Sidebar — service health + navigation.
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Educational Content Recommender", layout="wide")
st.sidebar.title("Educational Content Recommender")

health = api_health()
status_colour = {"ready": "🟢", "uninitialised": "🟡", "unreachable": "🔴"}
badge = status_colour.get(health.get("status", "unreachable"), "🔴")
st.sidebar.markdown(
    f"**API status:** {badge} `{health.get('status', 'unknown')}`  \n"
    f"**API URL:** `{API_URL}`"
)
if health.get("status") == "ready":
    st.sidebar.markdown(
        f"- {health['n_models']} models loaded  \n"
        f"- {health['n_students']:,} students  \n"
        f"- {health['n_items']:,} items  \n"
        f"- cutoff date: `{health['cutoff_date']}`"
    )
elif health.get("status") == "unreachable":
    st.sidebar.error(
        f"Cannot reach {API_URL}. Start the API with "
        f"`uvicorn src.api:app --port 8000` in another terminal."
    )
else:
    st.sidebar.warning(
        "API is running but model artefacts are not loaded. Run "
        "`python -m src.pipeline --persist-models` first."
    )

page = st.sidebar.radio(
    "Page",
    ["Recommendation Explorer", "Fairness View", "Ablation Comparison"],
    index=0,
)


# --------------------------------------------------------------------------- #
# Page 1: Recommendation Explorer.
# --------------------------------------------------------------------------- #

def _render_recommendation_explorer() -> None:
    st.title("Recommendation Explorer")
    st.caption(
        "Query the recommender for any student in the test set. For the "
        "Hybrid and GatedHybrid models the response includes a per-item "
        "score decomposition — the CF / content / outcome weighted "
        "contributions summed to the final rank score."
    )

    if health.get("status") != "ready":
        st.stop()

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        student_id = st.number_input(
            "Student ID",
            min_value=1,
            value=int(os.environ.get("RECSYS_DEMO_STUDENT", "6516")),
            step=1,
            help="Any id_student from the OULAD split (see cutoff date in sidebar).",
        )
    with col_b:
        k = st.slider("K", min_value=1, max_value=20, value=10)
    with col_c:
        model_options = health.get("model_names", ["Hybrid"])
        default_idx = model_options.index("Hybrid") if "Hybrid" in model_options else 0
        model = st.selectbox("Model", model_options, index=default_idx)

    try:
        result = api_recommend(int(student_id), int(k), model)
    except httpx.HTTPStatusError as exc:
        st.error(f"API returned {exc.response.status_code}: {exc.response.text}")
        st.stop()

    items = result["items"]
    if not items:
        st.warning("No recommendations for this student (candidate pool empty after excluding seen items).")
        return

    df = pd.DataFrame(items)
    has_breakdown = "cf_weighted" in df.columns
    st.subheader(f"Top {len(items)} for student {result['student_id']} ({result['model']})")
    display_cols = ["rank", "item_id"]
    if has_breakdown:
        display_cols += ["cf_weighted", "content_weighted", "outcome_weighted", "total"]
    st.dataframe(
        df[display_cols].style.format({
            "cf_weighted": "{:.4f}",
            "content_weighted": "{:.4f}",
            "outcome_weighted": "{:.4f}",
            "total": "{:.4f}",
        } if has_breakdown else {}),
        use_container_width=True,
        hide_index=True,
    )

    if has_breakdown:
        st.subheader("Score decomposition")
        st.caption(
            "Each bar is one recommendation. Height = final hybrid score; "
            "colours = CF (blue), content (orange), outcome (green) weighted contributions."
        )
        chart_df = df.set_index("rank")[["cf_weighted", "content_weighted", "outcome_weighted"]]
        chart_df.columns = ["CF", "Content", "Outcome"]
        st.bar_chart(chart_df, stack=True, use_container_width=True)

        st.subheader("Per-item audit")
        selected_rank = st.selectbox(
            "Explain a specific recommendation",
            df["rank"].tolist(),
            format_func=lambda r: f"Rank {r} — item {int(df[df['rank']==r]['item_id'].iloc[0])}",
        )
        selected_item = int(df[df["rank"] == selected_rank]["item_id"].iloc[0])
        try:
            bd = api_decompose(int(student_id), selected_item)
        except httpx.HTTPStatusError as exc:
            st.error(f"/decompose failed: {exc.response.status_code}")
            return
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CF (weighted)", f"{bd['cf_weighted']:.4f}", f"raw {bd['cf_raw']:.3f}")
        col2.metric("Content (weighted)", f"{bd['content_weighted']:.4f}", f"raw {bd['content_raw']:.3f}")
        col3.metric("Outcome (weighted)", f"{bd['outcome_weighted']:.4f}", f"raw mean-score {bd['outcome_raw']:.1f}")
        col4.metric("Total", f"{bd['total']:.4f}")


# --------------------------------------------------------------------------- #
# Page 2: Fairness View.
# --------------------------------------------------------------------------- #

def _render_fairness_view() -> None:
    st.title("Fairness Audit")
    st.caption(
        "Per-attribute metric breakdown for the deployed GatedHybrid, "
        "read from `evaluation/fairness_audit.csv`. This matches §5.9 of the report."
    )
    df = load_fairness_csv()
    if df is None:
        st.warning(
            "No `evaluation/fairness_audit.csv` on disk. Run "
            "`python -m src.pipeline` to produce it."
        )
        return

    attributes = df["attribute"].unique().tolist()
    metric_cols = [c for c in df.columns if c not in ("attribute", "level", "n_users")]
    metric = st.selectbox("Metric", metric_cols, index=metric_cols.index("precision@10") if "precision@10" in metric_cols else 0)

    for attr in attributes:
        st.subheader(attr)
        sub = df[df["attribute"] == attr].sort_values(metric, ascending=False)
        table_cols = ["level", "n_users", metric]
        st.dataframe(
            sub[table_cols].style.format({metric: "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(sub.set_index("level")[metric], use_container_width=True)

    cs = load_cold_start_csv()
    if cs is not None:
        st.subheader("Cold-start vs warm (per §5.7)")
        st.caption(
            "Metrics broken down by whether the user's training-window "
            "clicks exceed the cold-start threshold (default 10)."
        )
        st.dataframe(
            cs[["model", "stratum", "n_users", "precision@10", "ndcg@10"]].style.format({
                "precision@10": "{:.4f}", "ndcg@10": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------------------------------- #
# Page 3: Ablation Comparison.
# --------------------------------------------------------------------------- #

def _render_ablation() -> None:
    st.title("Ablation Comparison")
    st.caption(
        "Side-by-side top-K from Content, Hybrid, and GatedHybrid for the "
        "same student. Where the lists diverge is where the switching "
        "behaviour of the Gated variant kicks in."
    )
    if health.get("status") != "ready":
        st.stop()

    col_a, col_b = st.columns([2, 1])
    with col_a:
        student_id = st.number_input(
            "Student ID",
            min_value=1,
            value=int(os.environ.get("RECSYS_DEMO_STUDENT", "6516")),
            step=1,
            key="ablation_student",
        )
    with col_b:
        k = st.slider("K", min_value=1, max_value=15, value=10, key="ablation_k")

    columns = st.columns(3)
    for col, model_name in zip(columns, ["Content", "Hybrid", "GatedHybrid"]):
        with col:
            st.subheader(model_name)
            try:
                result = api_recommend(int(student_id), int(k), model_name)
            except httpx.HTTPStatusError as exc:
                st.error(f"{exc.response.status_code}: {exc.response.text}")
                continue
            df = pd.DataFrame(result["items"])
            if df.empty:
                st.info("No recommendations.")
                continue
            display = df[["rank", "item_id"]]
            if "total" in df.columns:
                display = df[["rank", "item_id", "total"]]
                display = display.style.format({"total": "{:.4f}"})
            st.dataframe(display, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Dispatch.
# --------------------------------------------------------------------------- #

if page == "Recommendation Explorer":
    _render_recommendation_explorer()
elif page == "Fairness View":
    _render_fairness_view()
elif page == "Ablation Comparison":
    _render_ablation()
