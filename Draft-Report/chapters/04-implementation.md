# Implementation {#sec:implementation}

## Overview and Project Layout

The system is implemented in Python 3.14 with `numpy`, `scipy`, `pandas`, `scikit-learn`, and `implicit`. The repository layout separates data code from model code from evaluation code so each concern can be tested in isolation.

```
src/
├── oulad.py         # OULAD CSV loader + dtype enforcement
├── preprocess.py    # temporal split, sparse matrix, course-scoping, time-decay
├── features.py      # item feature extractor (train-only)
├── baselines.py     # Random, Popularity
├── svd.py           # truncated-SVD collaborative filter
├── als.py           # implicit-feedback ALS (Hu, Koren & Volinsky, 2008)
├── content.py       # cosine-similarity content-based recommender
├── hybrid.py        # weighted CF + content + outcome ensemble
├── gated_hybrid.py  # per-user switching hybrid
├── metrics.py       # ranking metrics + outcome-weighted precision
├── evaluation.py    # CV folds, bootstrap CIs, paired-t, ablation, fairness
├── tuning.py        # grid searches on a temporal validation sub-split
└── pipeline.py      # end-to-end runner producing evaluation/*.csv
tests/               # 88 unit tests
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

Four deviations form an iteration narrative rather than isolated defects: SVD had no λ to tune → added ALS; ALS underperformed SVD → dropped CF from the simplex; ablation confirmed CF's negative contribution → stratified warm/cold; Popularity beat every personalised model on cold users → gated hybrid.

**Truncated SVD has no λ.** The design chapter listed `λ ∈ {0.01, 0.05, 0.1}`, but `scipy.sparse.linalg.svds` is unregularised beyond truncation itself. I added ALS as a first-class recommender rather than force regularisation onto SVD — converting a design gap into a clean empirical CF-family comparison.

**ALS underperforms SVD on OULAD.** The 18-point ALS grid produced best NDCG@10 = 0.149 vs SVD's 0.167 on the validation split. Contrary to the implicit-feedback literature, plain SVD beat ALS at every confidence level, and higher α made ALS worse. My interpretation: OULAD click distributions have heavy positive per-user tails (browsing behaviour), so strong confidence overfits within-course structure and destroys held-out generalisation. This is a real empirical finding to report.

**Hybrid weight tuning drops CF entirely.** The simplex grid selected `(α=0.0, β=0.8, γ=0.2)` for both SVD- and ALS-backed configurations; the ablation's `hybrid_no_cf` beats `hybrid_full` at fixed weights. Kept as a documented finding, not papered over.

**Gated hybrid introduced after cold-start analysis.** Cold users (<10 training clicks) turned out to be a distinct regime where Popularity beat every personalised model. A Burke-switching hybrid (Burke, 2002) — Popularity for cold, tuned Hybrid for warm — recovers the cold-user regime without hurting warm performance, and is the default deployed recommender.

## Two-Stage Reranking with LambdaMART

I also implemented a **LambdaMART two-stage reranker** (Burges et al., 2010), the industry-standard learning-to-rank layer used by Bing, YouTube, Amazon, and LinkedIn. Stage 1 (retrieval) uses the tuned GatedHybrid to fetch top-N candidates; Stage 2 (rerank) uses a LightGBM `LGBMRanker` with `objective='lambdarank'` to reorder them by directly optimising NDCG.

**Features** (`src/reranker_features.py`) — 35 per (user, item) pair: 3 Stage-1 signals (min-max normalised CF, content, outcome), 25 item features (reusing `ItemFeatures.matrix`), and 7 user features (gender one-hot, IMD/age ordinal, disability, log-clicks, mean past assessment). All train-only.

**Training** (`scripts/train_reranker.py`) uses an honest three-window design: Stage 1 refit on `tune_train`, positive labels from the later `tune_val`, evaluation on `split.test`. Because features are min-max normalised within each user's candidate pool, the trained booster transfers unchanged when Stage 1 is swapped for the persisted full-train Hybrid at inference.

**Result**: the reranker underperforms the linear hybrid on precision/NDCG — a real empirical finding, discussed in the evaluation chapter (§5.11). Implementation is unit-tested; the trained booster is persisted alongside the other model artefacts.

## Evaluation Harness Internals

`metrics.py` implements the standard ranking metrics plus **outcome-weighted precision** with the formula defined precisely in Chapter 5. The implementation caches the training-only outcome baseline $\bar{o}$ once per `evaluate()` call so the per-user loop is O(K), not O(n_items). A parallel `per_user_metrics` function returns the four unaggregated per-user metric arrays used by bootstrap CIs and paired-*t* tests.

`evaluation.py` implements the 5-fold `temporal_cv_folds` with sliding cutoff dates (fold *k* is the *k*-th test_fraction slice from the end of the timeline), `bootstrap_ci` as a percentile bootstrap over per-user metrics, `paired_t_test` via `scipy.stats.ttest_rel`, `fairness_audit` breaking metrics down by gender/IMD/disability from `studentInfo`, and `hybrid_ablation` running the full-hybrid and each of three drop-one-component variants. `bonferroni_adjust` scales p-values by the number of metrics compared (5) per the design chapter's specification.

`tuning.py::build_tuning_split` re-slices the training window into `tune_train` (older) and `tune_val` (newer) so hyperparameter selection sees no test-set information. `tune_svd`, `tune_als`, and `tune_hybrid_weights` all evaluate on `tune_val` and return the best configuration by NDCG@10.

## Reproducibility and Testing

Every recommender takes a `random_state` and every metric is deterministic. The test suite covers all seven recommenders, all metrics including outcome-weighted precision (with cases where OWP reduces to precision under uniform outcomes, up-weights hits with above-average outcomes, and down-weights hits with below-average outcomes), the cold-start masks, Bonferroni adjustment, and every edge case that surfaced during development. `.venv/bin/python -m pytest tests/ -q` reports **88 passed** on a clean checkout. Dependencies are pinned in `requirements.txt`.

## Pipeline CLI and Outputs

`python -m src.pipeline` runs the three-pass pipeline end-to-end; CLI flags expose the interesting knobs — `--rebuild-split`, `--k 5 10 20`, `--no-course-scoping` (reproduces the prelim's Pop@10 = 0.00003 diagnostic), `--no-cv`, `--no-tuning`, `--cold-start-threshold`, `--hybrid-grid-step`, `--decay-rate`, `--persist-models` (writes to `data/processed/models/` for the FastAPI service). The LambdaMART reranker is trained via `python scripts/train_reranker.py`. The pipeline writes 11 CSVs to `evaluation/` (`baseline_results`, `metrics_with_ci`, `popularity_bias`, `fairness_audit`, `hybrid_ablation`, `paired_t_tests`, `cold_start_results`, `tuning_{svd,als,hybrid}`, `cv_results`) each directly cited from Chapter 5.

## Selected Results Preview

To close the implementation chapter with the sort of "visual representation of results" the report guidelines call for, Table 4.1 previews the current headline P@10 on the primary temporal split (decay = 0.01, tuned hyperparameters). Full metric tables with confidence intervals, cross-validation means, ablation deltas, and cold-start stratification are the subject of [@sec:evaluation].

| Model | P@10 | NDCG@10 | HR@10 | OWP@10 |
|---|---:|---:|---:|---:|
| Random | 0.069 | 0.069 | 0.446 | 0.071 |
| Popularity | 0.250 | 0.272 | 0.759 | 0.257 |
| SVD | 0.219 | 0.246 | 0.799 | 0.223 |
| ALS | 0.218 | 0.229 | 0.767 | 0.222 |
| Content | 0.269 | 0.293 | 0.790 | 0.279 |
| Hybrid | 0.275 | 0.303 | 0.784 | 0.287 |
| **GatedHybrid** | **0.276** | **0.303** | 0.784 | **0.287** |

*Table 4.1 — Primary-split metrics at K=10 (decay=0.01, tuned weights). GatedHybrid leads on P@10 and OWP@10, tying Hybrid on NDCG@10 and HR@10, while covering the cold-user regime that pure Hybrid loses to Popularity. Full analysis in [@sec:evaluation].*

The single-split total is an 18% relative gain in P@10 for GatedHybrid over the pre-tuning fixed-weight hybrid (0.233 in the preliminary report) — the concrete pay-off of the design deviations documented above. Figure 3.3 renders the per-candidate score decomposition for the top-*K* list; the Streamlit dashboard (§4.9) surfaces the same decomposition interactively for any student in the test set.

## Service Layer: FastAPI, Streamlit, Docker {#sec:impl-service}

The offline pipeline is packaged behind a service layer so a marker can query the recommender interactively without re-running any experiment. Three components: a **FastAPI** JSON service, a **Streamlit** transparency dashboard, and a **two-stage Docker image** that installs the Cython `implicit` dependency deterministically.

**Persistence (`src/persistence.py`).** `python -m src.pipeline --persist-models` writes three artefacts to `data/processed/models/`: `models.joblib` (dict of the seven fitted recommenders + the LambdaMART booster), `serving_context.joblib` (student/item index, per-user course-scoped candidate arrays, outcome vector, cutoff date), and a JSON `manifest.json` recording tuned hyperparameters, git commit, and shape metadata so the marker can verify what they loaded.

**FastAPI (`src/api.py`).** Three read-only endpoints, loading the persisted artefacts once at lifespan startup so request cost is a small numpy operation:

```
GET /health                            → loaded roster, n_students, n_items, cutoff_date
GET /recommend/{student_id}?k=10&model=Hybrid
                                       → top-K item_ids with rank; Hybrid/GatedHybrid
                                         responses additionally include (cf_weighted,
                                         content_weighted, outcome_weighted, total).
GET /decompose/{student_id}/{item_id}  → un-normalised (cf_raw, content_raw,
                                         outcome_raw) plus the weighted decomposition
                                         within the user's candidate pool.
```

An example round-trip from `curl` against the running image:

```json
$ curl -s http://localhost:8000/recommend/6516?k=3
{"student_id":6516,"model":"Hybrid","k":3,
 "items":[
   {"item_id":546652,"rank":1,"cf_weighted":0.00,"content_weighted":0.79,
    "outcome_weighted":0.15,"total":0.94},
   {"item_id":546614,"rank":2,"cf_weighted":0.00,"content_weighted":0.76,
    "outcome_weighted":0.14,"total":0.90},
   {"item_id":546643,"rank":3,"cf_weighted":0.00,"content_weighted":0.72,
    "outcome_weighted":0.16,"total":0.88}]}
```

Course-scoping is enforced server-side: `/recommend` intersects the model's output with `serving_context.user_candidates[row]` before returning, so the JSON response can never leak items from presentations the student is not enrolled in. Twelve unit tests (`tests/test_api.py` + `tests/test_persistence.py`) cover 404 paths for unknown students/items, the uninitialised-service `/health` degraded response, decomposition math parity with `hybrid.score_breakdown`, and the round-trip of every fitted recommender through joblib.

**Streamlit (`dashboard/app.py`).** Three pages sitting on top of the FastAPI service:

1. *Recommendation Explorer* — student ID input, K slider, model selector; renders the top-*K* table with a stacked bar chart of `(cf, content, outcome)` weighted contributions and a per-item audit view that calls `/decompose`.
2. *Fairness View* — grouped bar charts of Precision@10, NDCG@10, and OWP@10 broken down by gender / IMD band / disability, driven by `evaluation/fairness_audit.csv`, with the warm/cold cold-start breakdown underneath.
3. *Ablation Comparison* — side-by-side top-*K* lists for Content, Hybrid, and GatedHybrid on the same student, highlighting rank divergences that indicate the switching branch has activated.

The dashboard is deliberately a *transparency* artefact: it exists so a marker can query any student in the test set and see both the recommendations *and* the reasons behind them. Screenshots of the three pages appear as Figures 4.1–4.3.

![Figure 4.1 — Recommendation Explorer for student 6516: the top-10 GatedHybrid output with a stacked-bar decomposition (CF blue / Content orange / Outcome green) and the per-item audit metric row.](figures/fig_4_1_dashboard_recommendation_explorer.png){width=95%}

![Figure 4.2 — Fairness View: per-attribute Precision@10 breakdown for GatedHybrid across gender, IMD band, and disability, with warm/cold cold-start table below.](figures/fig_4_2_dashboard_fairness_view.png){width=95%}

![Figure 4.3 — Ablation Comparison: side-by-side top-10 lists for Content, Hybrid, and GatedHybrid on the same student, exposing rank divergences where the switching branch is active.](figures/fig_4_3_dashboard_ablation.png){width=95%}

**Docker (`Dockerfile`).** Two-stage build. Stage 1 (`build`) installs `build-essential`, `libopenblas-dev`, `liblapack-dev`, and `gfortran` on `python:3.12-slim-bookworm` and compiles `implicit` + every other requirement into a venv. Stage 2 (`runtime`) copies the venv into a fresh slim image with only `libopenblas0` and `libgomp1`, serves the FastAPI app on port 8000 via uvicorn, and health-checks `/health`. One `docker build ... && docker run` reproduces the running service on any Docker host — the concrete answer to the `implicit` install risk flagged in the design chapter and the artefact used for the clean-environment verification (§3.8).
