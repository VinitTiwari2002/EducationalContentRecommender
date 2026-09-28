# Implementation {#sec:implementation}

## Overview and Project Layout

The system is implemented in Python 3.14 (Docker runtime image uses 3.12) with `numpy`, `scipy`, `pandas`, `scikit-learn`, `implicit`, `lightgbm`, `fastapi`, `uvicorn`, and `streamlit`. The repository separates data code from model code from evaluation code from serving code so each concern is testable in isolation.

```
src/
├── oulad.py             # OULAD CSV loader + dtype enforcement
├── preprocess.py        # temporal split, sparse matrix, course-scoping, decay
├── features.py          # item feature extractor (train-only)
├── baselines.py         # Random, Popularity
├── svd.py               # truncated-SVD collaborative filter
├── als.py               # implicit-feedback ALS (Hu, Koren & Volinsky, 2008)
├── content.py           # cosine-similarity content-based recommender
├── hybrid.py            # weighted CF + content + outcome ensemble
├── gated_hybrid.py      # per-user switching hybrid
├── lambdamart.py        # two-stage LambdaMART reranker (Stage 2)
├── reranker_features.py # per-(user, item) features for Stage 2
├── metrics.py           # ranking metrics + outcome-weighted precision
├── evaluation.py        # CV folds, bootstrap CIs, paired-t, ablation, fairness
├── tuning.py            # grid searches on a temporal validation sub-split
├── pipeline.py          # end-to-end runner producing evaluation/*.csv
├── persistence.py       # joblib artefacts + serving context + manifest
└── api.py               # FastAPI service (/health, /recommend, /decompose)
tests/                   # 116 unit tests (recommenders, metrics, cold-start,
                         # tuning, persistence round-trip, FastAPI endpoints)
dashboard/               # Streamlit transparency dashboard on top of the API
scripts/                 # LambdaMART training, robustness sweep, figure gens
Dockerfile               # two-stage image for the FastAPI service
```

Every recommender implements the same three-method contract — `fit`, `score(user_row)`, and `recommend(user_row, k, candidates, exclude_seen)` — so the evaluation harness treats them interchangeably. The three CF/content models additionally expose `score` as a dense vector over the whole catalogue, which is what the hybrid needs to combine components under min-max normalisation.

## Data Ingestion and Preprocessing

`oulad.py` loads the seven OULAD CSVs into a frozen dataclass; `"?"` sentinels (notably `imd_band` and unmarked assessments) become NaN.

`preprocess.py::build_split` aggregates `sum_click` per `(id_student, id_site)` inside each temporal window, producing `scipy.sparse.csr_matrix` matrices of shape (26,074 × 6,268), and records the per-user course-scoped candidate index. **Course-scoping** is a hard constraint: each `id_site` belongs to exactly one `(code_module, code_presentation)`, and per-user candidate arrays are precomputed from `studentRegistration ∩ studentVle`. **Time-decay** weights training clicks by `exp(-λ × (cutoff − date))` before CSR construction (test labels are undecayed binary); empirically the best λ was 0.01 (Chapter 5).

## Item Features

`features.py::build_item_features` returns a dense `(n_items, 25)` float32 matrix aligned with the sparse-matrix column order. Features are:

- `activity_type` one-hot (20 categories from `vle`)
- `week_from_norm`, `week_to_norm` (min-max), plus a `week_known` flag for NaN cases
- `log_access_count` (log1p, min-max normalised)
- `mean_score_of_accessers`: mean assessment score of training-window users who clicked the item, in [0, 100] / 100

The last feature also serves as the outcome vector consumed by the outcome-weighted precision metric — but crucially the *raw* score (in [0, 100]) is exposed separately as `ItemFeatures.outcome_score` so the metric baseline is on the natural score scale, not the normalised one. Every quantity is computed strictly from `train` and `data.student_assessment`, so no test-window signal leaks into features (Ricci et al., 2015).

## The Seven Recommenders

**Random** — uniform sample from the candidate pool via seeded `numpy.random.default_rng`.

**Popularity** — precomputes `item_totals = train.sum(axis=0)` once, then per-user ranks their intersection with the candidate pool. Surprisingly hard to beat because course-central resources dominate.

**SVDRecommender** — `scipy.sparse.linalg.svds` on `log1p(train)`, scoring `user_factors[u] · S · V^T[:, i]`. Truncation is the sole regulariser (Koren, Bell & Volinsky, 2009) — no explicit λ, documented as a design deviation below.

**ALSRecommender** — wraps `implicit.als.AlternatingLeastSquares` with confidence `c(u,i) = 1 + α · clicks(u,i)` (Hu, Koren & Volinsky, 2008); `n_factors`, `regularization`, `alpha` all tuned. Added at implementation time when tuning `SVD` surfaced the missing-λ gap.

**ContentRecommender** — L2-normalises the item feature matrix at fit time, computes each user profile as the click-weighted mean of accessed-item features (`train @ features_norm`, row-normalised), and scores by dot product — cosine similarity.

**HybridRecommender** — combines CF and content sub-models with the outcome vector: `α · minmax(cf) + β · minmax(content) + γ · minmax(outcome)`, min-max applied *within the candidate pool* so weights remain interpretable. `set_weights()` updates weights without refitting, which makes the 21-point simplex-grid tuning cheap.

**GatedHybridRecommender** — routes users with fewer than `threshold` training clicks to Popularity, others to the tuned Hybrid. Gate is a single comparison at recommend time.

## Design Deviations Discovered During Implementation

Four deviations form an iteration narrative rather than isolated defects: SVD had no λ to tune -> added ALS; ALS underperformed SVD -> dropped CF from the simplex; ablation confirmed CF's negative contribution -> stratified warm/cold; Popularity beat every personalised model on cold users -> gated hybrid.

**Truncated SVD has no λ.** The design chapter listed `λ ∈ {0.01, 0.05, 0.1}`, but `scipy.sparse.linalg.svds` is unregularised beyond truncation itself. I added ALS as a first-class recommender rather than force regularisation onto SVD — converting a design gap into a clean empirical CF-family comparison.

**ALS underperforms SVD on OULAD.** The 18-point ALS grid produced best NDCG@10 = 0.149 vs SVD's 0.167 on the validation split. Contrary to the implicit-feedback literature, plain SVD beat ALS at every confidence level, and higher α made ALS worse. My interpretation: OULAD click distributions have heavy positive per-user tails (browsing behaviour), so strong confidence overfits within-course structure and destroys held-out generalisation. This is a real empirical finding to report.

**Hybrid weight tuning drops CF entirely.** The simplex grid selected `(α=0.0, β=0.8, γ=0.2)` for both SVD- and ALS-backed configurations; the ablation's `hybrid_no_cf` beats `hybrid_full` at fixed weights. Kept as a documented finding, not papered over.

**Gated hybrid introduced after cold-start analysis.** Cold users (<10 training clicks) turned out to be a distinct regime where Popularity beat every personalised model. A Burke-switching hybrid (Burke, 2002) — Popularity for cold, tuned Hybrid for warm — recovers the cold-user regime without hurting warm performance, and is the default deployed recommender.

## Two-Stage Reranking with LambdaMART

I also implemented a **LambdaMART two-stage reranker** (Burges et al., 2010), the industry-standard learning-to-rank layer used by Bing, YouTube, Amazon, and LinkedIn. Stage 1 (retrieval) uses the tuned Hybrid ensemble to fetch top-N candidates (`src/lambdamart.py:47` type-checks that `stage1` is a `HybridRecommender`); Stage 2 (rerank) uses a LightGBM `LGBMRanker` with `objective='lambdarank'` to reorder them by directly optimising NDCG. The comparison baseline in Table 5.5 is the deployed GatedHybrid — the strongest single-model system in §5.3 — not the intermediate Hybrid that feeds Stage 2.

**Features** (`src/reranker_features.py`) — 35 per (user, item) pair: 3 Stage-1 signals (min-max normalised CF, content, outcome), 25 item features (reusing `ItemFeatures.matrix`), and 7 user features (gender one-hot, IMD/age ordinal, disability, log-clicks, mean past assessment). All train-only.

**Training** (`scripts/train_reranker.py`) uses an honest three-window design: Stage 1 refit on `tune_train`, positive labels from the later `tune_val`, evaluation on `split.test`. Because features are min-max normalised within each user's candidate pool, the trained booster transfers unchanged when Stage 1 is swapped for the persisted full-train Hybrid at inference.

**Result**: the reranker underperforms the linear hybrid on precision/NDCG — a real empirical finding, discussed in the evaluation chapter (§5.11). Implementation is unit-tested; the trained booster is persisted alongside the other model artefacts.

## Evaluation Harness Internals

`metrics.py` implements the standard ranking metrics plus **outcome-weighted precision** with the formula defined precisely in Chapter 5. The implementation caches the training-only outcome baseline $\bar{o}$ once per `evaluate()` call so the per-user loop is O(K), not O(n_items). A parallel `per_user_metrics` function returns the five unaggregated per-user metric arrays (Precision, Recall, NDCG, Hit-Rate, OWP@K when `outcome_score` is supplied) used by bootstrap CIs and paired-*t* tests.

`evaluation.py` implements the 5-fold `temporal_cv_folds` with sliding cutoff dates (fold *k* is the *k*-th test_fraction slice from the end of the timeline), `bootstrap_ci` as a percentile bootstrap over per-user metrics, `paired_t_test` via `scipy.stats.ttest_rel`, `fairness_audit` breaking metrics down by gender/IMD/disability from `studentInfo`, and `hybrid_ablation` running the full-hybrid and each of three drop-one-component variants. `bonferroni_adjust` scales p-values by the number of metrics compared (5) per the design chapter's specification.

`tuning.py::build_tuning_split` re-slices the training window into `tune_train` (older) and `tune_val` (newer) so hyperparameter selection sees no test-set information. `tune_svd`, `tune_als`, and `tune_hybrid_weights` all evaluate on `tune_val` and return the best configuration by NDCG@10.

## Reproducibility and Testing

Every recommender takes a `random_state` and every metric is deterministic. The test suite covers all seven recommenders, the LambdaMART reranker, all metrics including outcome-weighted precision (with cases where OWP reduces to precision under uniform outcomes, up-weights hits with above-average outcomes, and down-weights hits with below-average outcomes), the cold-start masks, Bonferroni adjustment, model persistence round-trips, and the FastAPI endpoints. `.venv/bin/python -m pytest tests/ -q` reports **116 passed** on a clean checkout. Dependencies are pinned in `requirements.txt`.

## Pipeline CLI and Outputs

`python -m src.pipeline` runs the three-pass pipeline end-to-end; CLI flags expose the interesting knobs — `--rebuild-split`, `--k 5 10 20`, `--no-course-scoping` (reproduces the prelim's Pop@10 = 0.00003 diagnostic), `--no-cv`, `--no-tuning`, `--cold-start-threshold`, `--hybrid-grid-step`, `--decay-rate`, `--persist-models` (writes to `data/processed/models/` for the FastAPI service). The LambdaMART reranker is trained via `python scripts/train_reranker.py`. The pipeline writes 11 CSVs to `evaluation/` (`baseline_results`, `metrics_with_ci`, `popularity_bias`, `fairness_audit`, `hybrid_ablation`, `paired_t_tests`, `cold_start_results`, `tuning_{svd,als,hybrid}`, `cv_results`) each directly cited from Chapter 5.

## Selected Results Preview

Table 4.1 previews headline P@10 on the primary temporal split (decay = 0.01, tuned hyperparameters); full CI/CV/ablation/cold-start analysis is in [@sec:evaluation].

| Model | P@10 | NDCG@10 | HR@10 | OWP@10 |
|---|---:|---:|---:|---:|
| Random | 0.069 | 0.069 | 0.446 | 0.071 |
| Popularity | 0.251 | 0.272 | 0.759 | 0.257 |
| SVD | 0.219 | 0.246 | 0.799 | 0.223 |
| ALS | 0.218 | 0.229 | 0.767 | 0.222 |
| Content | 0.269 | 0.293 | 0.790 | 0.279 |
| Hybrid | 0.275 | 0.303 | 0.784 | 0.287 |
| **GatedHybrid** | **0.276** | **0.303** | 0.784 | **0.287** |

*Table 4.1 — Primary-split metrics at K=10 (decay=0.01, tuned weights). GatedHybrid leads P@10 and OWP@10, ties Hybrid on NDCG@10 and HR@10, and covers the cold-user regime pure Hybrid loses to Popularity.*

The single-split total is an 18% relative P@10 gain for GatedHybrid over the pre-tuning fixed-weight hybrid (0.233 in the prelim) — the pay-off of the design deviations above. Figure 3.3 renders the per-candidate score decomposition; the Streamlit dashboard (§4.9) surfaces it interactively for any student.

## Service Layer: FastAPI, Streamlit, Docker {#sec:impl-service}

The offline pipeline is packaged behind a service layer so a marker can query it interactively without re-running any experiment. Three components: a **FastAPI** service, a **Streamlit** transparency dashboard, and a **two-stage Docker image** that installs `implicit` deterministically.

**Persistence (`src/persistence.py`).** `python -m src.pipeline --persist-models` writes `models.joblib` (fitted recommenders + LambdaMART booster), `serving_context.joblib` (student/item index, per-user candidates, outcome vector, cutoff date), and `manifest.json` (tuned hyperparameters, git commit, shape metadata) so the marker can verify what they loaded.

**FastAPI (`src/api.py`).** Three read-only endpoints, loading the persisted artefacts once at lifespan startup so request cost is a small numpy operation:

```
GET /health
    -> loaded roster, n_students, n_items, cutoff_date

GET /recommend/{student_id}?k=10&model=Hybrid
    -> top-K item_ids with rank; Hybrid and GatedHybrid responses
       additionally include (cf_weighted, content_weighted,
       outcome_weighted, total) per item.

GET /decompose/{student_id}/{item_id}
    -> un-normalised (cf_raw, content_raw, outcome_raw)
       plus the weighted decomposition within the user's
       candidate pool.
```

A real round-trip against the running service (persisted Hybrid, $\alpha=0.0, \beta=0.8, \gamma=0.2$):

```json
$ curl -s 'http://localhost:8000/recommend/6516?k=3&model=Hybrid'
{
  "student_id": 6516,
  "model": "Hybrid",
  "k": 3,
  "items": [
    {"item_id": 877059, "rank": 1,
     "cf_weighted": 0.00, "content_weighted": 0.80,
     "outcome_weighted": 0.12, "total": 0.92},
    {"item_id": 877032, "rank": 3,
     "cf_weighted": 0.00, "content_weighted": 0.76,
     "outcome_weighted": 0.03, "total": 0.79},
    {"item_id": 877042, "rank": 4,
     "cf_weighted": 0.00, "content_weighted": 0.62,
     "outcome_weighted": 0.06, "total": 0.68}
  ]
}
```

`/decompose/6516/877059` returns the raw component scores (CF = 0.91, content cosine = 0.94, mean-score = 73.2/100) alongside the pool-normalised weighted contributions summing to `total` — what makes each recommendation explainable.

Course-scoping is enforced server-side: `/recommend` intersects the model's output with `serving_context.user_candidates[row]` before returning, so the JSON response can never leak items from presentations the student is not enrolled in. Sixteen unit tests (`tests/test_api.py` + `tests/test_persistence.py`) cover 404 paths for unknown students/items, the uninitialised-service `/health` degraded response, decomposition math parity with `hybrid.score_breakdown`, and the round-trip of every fitted recommender through joblib.

**Streamlit (`dashboard/app.py`).** Three pages sitting on top of the FastAPI service:

1. *Recommendation Explorer* — student ID input, K slider, model selector; renders the top-*K* table with a stacked bar chart of `(cf, content, outcome)` weighted contributions and a per-item audit view that calls `/decompose`.
2. *Fairness View* — grouped bar charts of Precision@10, NDCG@10, and OWP@10 broken down by gender / IMD band / disability, driven by `evaluation/fairness_audit.csv`, with the warm/cold cold-start breakdown underneath.
3. *Ablation Comparison* — side-by-side top-*K* lists for Content, Hybrid, and GatedHybrid on the same student, highlighting rank divergences that indicate the switching branch has activated.

The dashboard is deliberately a *transparency* artefact: it exists so a marker can query any student in the test set and see both the recommendations *and* the reasons behind them. Screenshots of the three pages appear as Figures 4.1–4.3.

![Figure 4.1 — Recommendation Explorer screenshot: student 6516, K=10, model=Hybrid. Top-10 table, stacked-bar score decomposition (CF blue at $\alpha=0$; content orange dominates; outcome red is the $\gamma$-weighted residual), and Rank-1 per-item audit tiles (CF 0.0000/raw 0.910; Content 0.7940/raw cosine 0.944; Outcome 0.1088/raw mean-score 73.2; Total 0.9028).](figures/fig_4_1_dashboard_recommendation_explorer.png){width=95%}

![Figure 4.2a — Fairness View, gender + IMD band, driven by `fairness_audit.csv`. Gender: M (n=9,233) P@10 = 0.29, F (n=8,329) 0.26. IMD: 11 bands, P@10 range 0.26–0.31, no clear deprivation gradient.](figures/fig_4_2a_dashboard_fairness_gender+imd.png){width=95%}

![Figure 4.2b — Fairness View, disability + cold-start table (`cold_start_results.csv`). Disability N vs Y: 0.28 vs 0.26. On the cold cohort ($n=155$) Popularity's P@10 = 0.20 beats every personalised model — the observation that motivated the GatedHybrid switching recommender.](figures/fig_4_2b_dashboard_fairness_disability+coldWarm.png){width=95%}

![Figure 4.3 — Ablation Comparison for student 6516 (K=10). Content, Hybrid, GatedHybrid columns side-by-side. Hybrid and GatedHybrid coincide (warm user; switching branch routes to Hybrid); Content diverges from rank 4 onward.](figures/fig_4_3_dashboard_ablation.png){width=95%}

**Docker (`Dockerfile`).** Two-stage build. Stage 1 (`build`) installs `build-essential`, `libopenblas-dev`, `liblapack-dev`, and `gfortran` on `python:3.12-slim-bookworm` and compiles `implicit` + every other requirement into a venv. Stage 2 (`runtime`) copies the venv into a fresh slim image with only `libopenblas0` and `libgomp1`, serves the FastAPI app on port 8000 via uvicorn, and health-checks `/health`. One `docker build ... && docker run` reproduces the running service on any Docker-compatible host — the concrete answer to the `implicit` install risk flagged in the design chapter.

**Clean-environment verification.** The build was reproduced from scratch inside a stock `containerd v2.2.1 / runc 1.4.0` runtime: `docker build -t recsys:latest .` compiles `implicit` and all Python deps into the stage-1 venv in ~76 s, and the stage-2 runtime image exports in a further ~52 s. `docker run -p 8000:8000 -v $(pwd)/data:/app/data recsys:latest`, then `/health` returns `{"status":"ready","n_models":7,"n_students":26074,"n_items":6268,"cutoff_date":172}`; `/recommend/6516?k=3&model=Hybrid` returns the same top-3 as the host run (item 877059 with content=0.80, outcome=0.11, total=0.91 — matching the JSON above to float rounding). This exercises the full stack — Cython compile, joblib load, FastAPI startup, course-scoped intersection — on a stock Debian-slim image with no host-Python leakage, the clean-environment reproducibility check the draft feedback requested.
