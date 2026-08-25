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

`oulad.py` loads the seven OULAD CSVs into a frozen dataclass with type coercion; missing values encoded as `"?"` (notably `imd_band` and un-marked assessment scores) are converted to NaN so numeric columns stay numeric.

`preprocess.py::build_split` aggregates `sum_click` per `(id_student, id_site)` inside each of the two temporal windows, produces `scipy.sparse.csr_matrix` matrices of shape (26,074 × 6,268), and records the per-user course-scoped candidate index. Two design elements from Chapter 3 are enforced here.

**Course-scoping** is a hard constraint: each `id_site` belongs to exactly one `(code_module, code_presentation)`, and a per-user candidate array is precomputed from `studentRegistration` intersected with the user's presentations, so every recommender's candidate pool is restricted to items the user could actually access.

**Time-decay** weights the training-window clicks by `exp(-λ × (cutoff − date))` before the CSR matrix is built:

```python
if apply_decay and decay_rate > 0:
    weights = df["sum_click"].to_numpy(np.float32) * np.exp(
        -decay_rate * (cutoff - df["date"].to_numpy(np.float32))
    )
```

The test-window matrix is *not* decayed — the label is a binary "did this user click item *i* in the future window?" regardless of magnitude. Empirically the best λ was 0.01 (Chapter 5).

## Item Features

`features.py::build_item_features` returns a dense `(n_items, 25)` float32 matrix aligned with the sparse-matrix column order. Features are:

- `activity_type` one-hot (20 categories from `vle`)
- `week_from_norm`, `week_to_norm` (min-max), plus a `week_known` flag for NaN cases
- `log_access_count` (log1p, min-max normalised)
- `mean_score_of_accessers`: mean assessment score of training-window users who clicked the item, in [0, 100] / 100

The last feature also serves as the outcome vector consumed by the outcome-weighted precision metric — but crucially the *raw* score (in [0, 100]) is exposed separately as `ItemFeatures.outcome_score` so the metric baseline is on the natural score scale, not the normalised one. Every quantity is computed strictly from `train` and `data.student_assessment`, so no test-window signal leaks into features (Ricci et al., 2015).

## The Seven Recommenders

**Random** samples uniformly from the candidate pool with a `numpy.random.default_rng` seeded from `random_state`. Deterministic reproducibility.

**Popularity** precomputes `item_totals = np.asarray(train.sum(axis=0)).ravel()` at fit time, then per-user ranks the intersection of `item_totals` with the user's candidate pool. Surprisingly hard to beat in education because course-central resources dominate the click distribution.

**SVDRecommender** applies `scipy.sparse.linalg.svds` to `log1p(train)`, returning `user_factors[u] · S · V^T[:, i]` as the score. Rank *k* is a tuned hyperparameter. The design chapter mentioned an L2 λ term following Koren, Bell & Volinsky (2009); truncated SVD has no explicit λ — the truncation itself provides regularisation via low-rank approximation — so the implementation only tunes *k*. This is documented as a design deviation below.

**ALSRecommender** wraps `implicit.als.AlternatingLeastSquares`. Confidence is `c(u, i) = 1 + α · clicks(u, i)`; the algorithm alternates least-squares updates over user and item factors until convergence. `n_factors`, `regularization`, and `alpha` are all tuned. This is the canonical implicit-feedback CF baseline (Hu, Koren & Volinsky, 2008), added at implementation time because the design chapter did not specify it and empirical results (below) motivated including it as a first-class recommender.

**ContentRecommender** L2-normalises the item feature matrix once at fit time and computes each user's profile as the click-weighted mean of accessed-item features:

```python
raw_profiles = train @ item_features_norm
user_profiles = raw_profiles / np.linalg.norm(raw_profiles, axis=1, keepdims=True)
```

The score for candidate *i* is then the dot product of the L2-normalised item feature with the L2-normalised user profile — equivalent to cosine similarity.

**HybridRecommender** takes CF and Content sub-models (typically pre-fitted for cheap tuning) and combines their per-candidate score vectors with an outcome vector: `alpha · minmax(cf) + beta · minmax(content) + gamma · minmax(outcome)`. Min-max is applied *within the candidate pool* so weights remain interpretable regardless of course-scoping. A `set_weights` method updates weights without refitting, which is what makes the 21-point simplex grid tuning cheap: fit the components once, sweep the weights.

**GatedHybridRecommender** routes each user to Popularity if their training clicks are below a threshold (default 10) and to the tuned Hybrid otherwise. The gate is a single comparison at recommend time; both sub-models are fit at construction time.

## Design Deviations Discovered During Implementation

The four changes below are not isolated defects; they form an iteration narrative in which each empirical result unlocked the next design decision. The starting point was the preliminary design (SVD + content + outcome hybrid with fixed weights). Running the pipeline surfaced that SVD had no λ to tune, which motivated adding ALS. Tuning ALS in turn revealed it underperformed SVD on OULAD's implicit-feedback distribution, which motivated dropping CF from the hybrid weight simplex. The ablation confirming CF's negative contribution motivated stratifying by warm/cold users, which surfaced that Popularity beats every personalised model on cold users — motivating the switching (gated) hybrid. Time-decay was added last as a data-preprocessing question raised by the "recency matters in learning" pedagogical prior in §3.2. Each deviation is documented below with its trigger and its consequence, in the order the code was written. Documenting them here is what the marker asks for under "critical evaluation of the project so far".

**Truncated SVD has no λ.** The design chapter listed `λ ∈ {0.01, 0.05, 0.1}` as a tuning grid, but `scipy.sparse.linalg.svds` is unregularised beyond the truncation itself. Regularised MF would have required a different library (`implicit`, or a hand-written ALS). The workaround was to add ALS as a first-class recommender rather than force regularisation onto SVD; this converted a design gap into a clean empirical comparison between the two CF families.

**ALS underperforms SVD on OULAD.** The tuning grid explored `n_factors ∈ {32, 64, 128} × regularisation ∈ {0.01, 0.1} × alpha ∈ {1, 10, 40}` — 18 configurations — and the best ALS achieved NDCG@10 = 0.149 on the validation split vs. SVD's 0.167. Contrary to expectation from the implicit-feedback literature, plain SVD beat ALS at every confidence level tried, and higher alphas made ALS worse. My interpretation: OULAD click distributions have a heavy positive tail per user (browsing behaviour), so treating each click as strongly confident overfits the within-course structure and destroys generalisation to held-out clicks. This is a real empirical finding to report, not a bug — and it motivates the gated hybrid design decision that follows.

**Hybrid weight tuning drops the CF signal entirely.** Grid search over `(α, β, γ)` on the simplex consistently selected `(α=0.0, β=0.8, γ=0.2)`. In other words, both SVD- and ALS-backed hybrid configurations concluded that adding CF to content + outcome hurts NDCG. The single-split ablation confirms this: the `hybrid_no_cf` variant beats `hybrid_full` at fixed weights. Kept as a documented empirical finding rather than papered over.

**Gated hybrid was introduced after the cold-start analysis.** Splitting users into warm (≥10 training clicks) and cold cohorts revealed that Popularity outperforms every personalised model on cold users. A Burke-switching hybrid (Burke, 2002) — Popularity for cold, tuned Hybrid for warm — recovers the cold-user regime without hurting warm performance and is now the default deployed recommender in the pipeline.

## Evaluation Harness Internals

`metrics.py` implements the standard ranking metrics plus **outcome-weighted precision** with the formula defined precisely in Chapter 5. The implementation caches the training-only outcome baseline $\bar{o}$ once per `evaluate()` call so the per-user loop is O(K), not O(n_items). A parallel `per_user_metrics` function returns the four unaggregated per-user metric arrays used by bootstrap CIs and paired-*t* tests.

`evaluation.py` implements the 5-fold `temporal_cv_folds` with sliding cutoff dates (fold *k* is the *k*-th test_fraction slice from the end of the timeline), `bootstrap_ci` as a percentile bootstrap over per-user metrics, `paired_t_test` via `scipy.stats.ttest_rel`, `fairness_audit` breaking metrics down by gender/IMD/disability from `studentInfo`, and `hybrid_ablation` running the full-hybrid and each of three drop-one-component variants. `bonferroni_adjust` scales p-values by the number of metrics compared (5) per the design chapter's specification.

`tuning.py::build_tuning_split` re-slices the training window into `tune_train` (older) and `tune_val` (newer) so hyperparameter selection sees no test-set information. `tune_svd`, `tune_als`, and `tune_hybrid_weights` all evaluate on `tune_val` and return the best configuration by NDCG@10.

## Reproducibility and Testing

Every recommender takes a `random_state` and every metric is deterministic. The test suite covers all seven recommenders, all metrics including outcome-weighted precision (with cases where OWP reduces to precision under uniform outcomes, up-weights hits with above-average outcomes, and down-weights hits with below-average outcomes), the cold-start masks, Bonferroni adjustment, and every edge case that surfaced during development. `.venv/bin/python -m pytest tests/ -q` reports **88 passed** on a clean checkout. Dependencies are pinned in `requirements.txt`.

## Pipeline CLI and Outputs

`python -m src.pipeline` runs the three-pass pipeline end-to-end. CLI flags expose the interesting knobs:

```
--rebuild-split          Force rebuild of cached split
--k 5 10 20              Evaluate at multiple K
--no-course-scoping      Diagnostic: reproduce the prelim's Pop@10 = 0.00003
--no-cv                  Skip 5-fold CV (fast dev-loop)
--no-tuning              Skip hyperparameter tuning
--cold-start-threshold N Threshold for warm/cold split (default 10)
--hybrid-grid-step S     Simplex step for hybrid weights (default 0.2)
--decay-rate R           Time-decay rate on training clicks (default 0.0)
```

The pipeline writes 11 CSV files to `evaluation/`, each one directly citable from Chapter 5: `baseline_results`, `metrics_with_ci`, `popularity_bias`, `fairness_audit`, `hybrid_ablation`, `paired_t_tests` (with the Bonferroni column), `cold_start_results`, `tuning_svd`, `tuning_als`, `tuning_hybrid`, and `cv_results`. Chapter 5 reads directly from these artefacts.

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

The single-split total is an 18% relative gain in P@10 for GatedHybrid over the pre-tuning fixed-weight hybrid (0.233 in the preliminary report) — the concrete pay-off of the design deviations documented above. Figure 3.3 in the design chapter renders the per-candidate score decomposition that produced the top-*K* list; the Streamlit dashboard (Section 3.5) will surface the same decomposition interactively for any student in the test set.
