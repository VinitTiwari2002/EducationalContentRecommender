# Feature Prototype {#sec:prototype}

## What the Prototype Implements

The prototype is a working slice of the system described in [@sec:design], exercising every stage of the pipeline end-to-end on the full OULAD dataset (10.6 million interactions, 32,593 students, 6,364 VLE items). It contains four components:

1. A **data loader** that ingests the seven OULAD CSV tables and exposes them through a single typed container.
2. A **preprocessing module** that builds a sparse user-item interaction matrix from `studentVle`, performs a temporal 80/20 train/test split on the `date` column, and computes a per-user *course-scoped candidate set* from `studentRegistration` so recommenders only propose items from presentations the user is enrolled in.
3. An **evaluation framework** implementing Precision@K, Recall@K, NDCG@K, and Hit Rate@K with a harness that returns mean per-user scores.
4. Two **baseline recommenders** — Random and Popularity — behind a common `fit / recommend` interface that accepts a candidate-pool argument. A pipeline runner ties everything together: load $\rightarrow$ split $\rightarrow$ fit $\rightarrow$ evaluate at K in {5, 10, 20} $\rightarrow$ persist to `evaluation/baseline_results.csv`.

Every component the final hybrid system needs is now in place. The SVD, content-based, and hybrid recommenders will implement the same `Recommender` interface and be evaluated by the same harness on the same split, so model comparisons in the final report will be controlled and reproducible. Twenty-five unit tests cover the metrics, both baselines, candidate-pool restriction, edge cases, and recommender determinism. All pass on a fresh checkout; the full pipeline runs against OULAD in approximately three minutes on a laptop.

## Design Decisions and Justification

Four design decisions in the prototype deserve explicit justification, since each is informed either by the literature reviewed in [@sec:lit-review] or by structural properties of OULAD discovered during the prototype build.

**Sparse representation throughout.** OULAD's interaction matrix is 26,074 × 6,268 (after restricting to users with at least one VLE interaction) with 2.24 million non-zero entries across the train and test splits — a density of roughly 1.4%. Dense storage would require ~650 MB of float32, while CSR storage requires roughly 30 MB. More importantly, every operation downstream (matrix factorisation, item-item similarity) has efficient sparse variants. The pipeline therefore commits to CSR matrices end-to-end and never densifies. This is standard practice in the recommender-system literature (Koren et al., 2009).

**Temporal — not random — split.** The cutoff is the 80th percentile of `date` in `studentVle`, which corresponds to day 172 from course start. Training interactions are those at or before the cutoff; test interactions are those after. This is the temporal protocol argued for by Ricci, Rokach & Shapira (2015): random sampling would allow future information to leak into training and overestimate performance, particularly for a system whose intended use is sequential (recommending what to study *next*). The cost of temporal splitting is that the test set is biased towards late-course interactions, which the final evaluation chapter will discuss.

**Course-scoped candidate sets.** This decision emerged from the first end-to-end pipeline run and is the most consequential structural property of OULAD for a recommender. Each `id_site` belongs to exactly one `(code_module, code_presentation)`; the catalogue is partitioned by presentation rather than shared. An average student is enrolled in only 1.1 presentations and so has access to a median of 324 items (~5% of the 6,268 total). A naïve global Popularity recommender mostly suggests items from presentations the user is not enrolled in, which is meaningless regardless of metric. The prototype therefore computes per-user candidate sets from `studentRegistration`, and recommenders restrict their pool to that set. The first run *without* this restriction gave Popularity@10 = 0.00003 — an order of magnitude worse than Random. Adding course-scoping moved Popularity@10 to 0.204; this is itself a non-trivial methodological finding from the prototype phase.

**Common interface across recommenders.** Both baselines implement `fit(train)` and `recommend(user_row, k, candidates, exclude_seen)`, the standard implicit-feedback formulation (Rendle et al., 2012). Adopting this interface now means SVD, content-based, and hybrid models will drop into the same harness in later weeks without changes to the evaluation code.

## Results

The pipeline runs end-to-end against the full OULAD dataset and produces the results in Table 4.1. Evaluation covers 17,562 users — every user with at least one interaction in the held-out test window. All numbers are mean per-user scores; the harness writes them to `evaluation/baseline_results.csv`.

> **Table 4.1 — Baseline results on OULAD with course-scoped candidates.** Per-user means over 17,562 users. Higher is better for all four metrics.

| Model | K | Precision@K | Recall@K | NDCG@K | Hit Rate@K |
|---|---:|---:|---:|---:|---:|
| Random | 5 | 0.0712 | 0.0131 | 0.0719 | 0.2872 |
| Random | 10 | 0.0689 | 0.0253 | 0.0693 | 0.4470 |
| Random | 20 | 0.0689 | 0.0502 | 0.0731 | 0.6319 |
| **Popularity** | **5** | **0.2298** | **0.0398** | **0.2342** | **0.5446** |
| **Popularity** | **10** | **0.2035** | **0.0724** | **0.2171** | **0.6942** |
| **Popularity** | **20** | **0.1710** | **0.1228** | **0.1998** | **0.8198** |

Three observations from these numbers. First, Popularity outperforms Random by roughly a factor of three on Precision and NDCG and by a factor of 1.5–2 on Hit Rate. This direction is the expected one (Ricci et al., 2015) and confirms that the harness produces consistent, comparable numbers across models. Second, Hit Rate@20 reaches 0.82 for Popularity, meaning that for four out of five users at least one of their held-out items appears in the top-20 most popular items within their presentation; this gives an early sense of the upper bound on what raw popularity can achieve before any personalisation. Third, Precision declines with K for both models — fewer of the top-K items are relevant on average as K grows — while Recall and Hit Rate increase, which is the standard ranking-metric behaviour and a useful sanity check.

These numbers establish the bar that the SVD, content-based, and hybrid models will need to clear in the final report.

## Evaluation of the Prototype

Following Rendle et al. (2012) and Ricci et al. (2015), the harness reports Precision@K, Recall@K, NDCG@K, and Hit Rate@K — the standard set for implicit-feedback top-K recommendation — because no single metric captures recommender quality: Precision can hide poor coverage, Recall can hide poor ranking. The outcome-weighted Precision metric described in [@sec:design] is *not* yet in the harness; it depends on the feature-engineering step and is the principal honest gap between prototype and final system.

Two baselines are deliberately chosen. Random fixes the floor; any model failing to beat it has learnt nothing. Popularity is the strong baseline a useful recommender must clear (Thai-Nghe et al., 2010; Bousbahi & Chorfi, 2015) — most-accessed items in a course tend to be its central resources. Reproducibility is enforced: Random is seeded and deterministic; the split and candidate sets are persisted to disk; one command runs the full pipeline from a fresh checkout.

## Limitations and Planned Improvements

The prototype has three principal limitations, all of which are deliberate scoping decisions for the preliminary submission rather than oversights.

First, **only baselines are implemented.** SVD, content-based, and hybrid models are not in this submission. They will use the same interfaces and harness, so they will integrate without architectural changes.

Second, **outcome-aware evaluation is not yet in the harness.** Outcome-weighted Precision requires per-item mean assessment scores from the training portion of `studentAssessment`, which in turn requires the feature engineering step. The schema is sketched in the design chapter.

Third, **the temporal split is single-fold.** Five-fold temporal cross-validation, as argued in [@sec:design], is the protocol for the final evaluation. A single fold is sufficient to establish that the harness produces sane numbers; the multi-fold harness is a small extension and is scheduled for future submission.

The path from prototype to final system is short and well-defined. The remaining work falls into these blocks: SVD collaborative filtering, feature-based content filtering, hybrid ensemble with ablation, FastAPI service and Streamlit dashboard, final cross-validation and significance testing. The plan is feasible because the technically novel steps each use library implementations with established performance characteristics — the project's contribution is in their combination and evaluation, not in implementing new algorithms from scratch.
