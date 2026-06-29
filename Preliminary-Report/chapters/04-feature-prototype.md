# Feature Prototype {#sec:prototype}

## What the Prototype Implements

The prototype I submit with this preliminary report is a working slice of the system described in [@sec:design]. It exercises every stage of the pipeline end-to-end on the full OULAD dataset (10.6 million interactions, 32,593 students, 6,364 VLE items). It has four components.

1. A **data loader** that ingests the seven OULAD CSV tables and exposes them through a single typed container.
2. A **preprocessing module** that builds a sparse user-item interaction matrix from VLE, performs a temporal 80/20 train/test split on the `date` column, and computes a per-user *course-scoped candidate set* from `studentRegistration`, so that recommenders only propose items from presentations the user is enrolled in.
3. An **evaluation framework** implementing Precision@K, Recall@K, NDCG@K, and Hit Rate@K, with a harness that returns mean per-user scores.
4. Two **baseline recommenders** — Random and Popularity — behind a common `fit / recommend` interface that accepts a candidate-pool argument. A pipeline runner ties everything together: load $\rightarrow$ split $\rightarrow$ fit $\rightarrow$ evaluate at K in {5, 10, 20} $\rightarrow$ persist to `evaluation/baseline_results.csv`.

Every component the final hybrid system needs is in place. The SVD, content-based, and hybrid recommenders I plan to add will implement the same `Recommender` interface and be evaluated by the same harness on the same split, so the comparisons I make in the final report will be controlled and reproducible. I have written twenty-seven unit tests that cover the metrics, both baselines, candidate-pool restriction, edge cases, and recommender determinism. All pass on a fresh checkout, and the full pipeline runs against OULAD in about three minutes on a laptop.

## Design Decisions and Justification

Four design decisions deserve explicit justification. Each is informed either by the literature I reviewed in [@sec:lit-review] or by a structural property of OULAD I discovered during the prototype build.

**Sparse representation throughout.** The OULAD interaction matrix is 26,074 × 6,268 (after restricting to users with at least one VLE interaction) with 2.24 million non-zero entries across train and test — a density of roughly 1.4%. Dense storage would take around 650 MB of float32; CSR storage takes roughly 30 MB. More importantly, every operation downstream (matrix factorisation, item-item similarity) has an efficient sparse variant. I therefore commit to CSR matrices end-to-end and never densify. This is standard practice in the recommender-system literature (Koren et al., 2009).

**Temporal — not random — split.** I cut at the 80th percentile of `date` in `studentVle`, which corresponds to day 172 from course start. Training interactions are those at or before the cutoff; test interactions are those after. This is the temporal protocol argued for by Ricci, Rokach & Shapira (2015): random sampling would let future information leak into training and overestimate performance, particularly for a system whose intended use is sequential (recommending what to study *next*). The cost of temporal splitting is that the test set is biased towards late-course interactions, which I will return to in the final evaluation.

**Course-scoped candidate sets.** This decision emerged from the first end-to-end pipeline run and turned out to be the most consequential structural property of OULAD for a recommender. Each `id_site` belongs to exactly one `(code_module, code_presentation)`; the catalogue is partitioned by presentation rather than shared. An average student is enrolled in only 1.1 presentations and therefore has access to a median of 324 items (~5% of the 6,268 total). A naïve global Popularity recommender will mostly suggest items from presentations the user is not enrolled in, which is meaningless whatever metric I report. So I compute per-user candidate sets from `studentRegistration` and have the recommenders restrict their pool to that set. My first run *without* this restriction gave Popularity@10 = 0.00003 — an order of magnitude worse than Random. Adding course-scoping moved Popularity@10 to 0.204. The pipeline retains a `--no-course-scoping` diagnostic flag so the finding remains reproducible on demand. This is a non-trivial methodological finding in its own right, and it only surfaced because I had the pipeline running end-to-end.

**Common interface across recommenders.** Both baselines implement `fit(train)` and `recommend(user_row, k, candidates, exclude_seen)` — the standard implicit-feedback formulation (Rendle et al., 2012). I adopted this interface now so that the SVD, content-based, and hybrid models I add later drop into the same harness without changes to the evaluation code.

## Results

The pipeline runs end-to-end against the full OULAD dataset and produces the results in Table 4.1. The evaluation covers 17,562 users — every user with at least one interaction in the held-out test window. All numbers are mean per-user scores; the harness writes them to `evaluation/baseline_results.csv`.

> **Table 4.1 — Baseline results on OULAD with course-scoped candidates.** Per-user means over 17,562 users. Higher is better for all four metrics.

| Model | K | Precision@K | Recall@K | NDCG@K | Hit Rate@K |
|---|---:|---:|---:|---:|---:|
| Random | 5 | 0.0712 | 0.0131 | 0.0719 | 0.2872 |
| Random | 10 | 0.0689 | 0.0253 | 0.0693 | 0.4470 |
| Random | 20 | 0.0689 | 0.0502 | 0.0731 | 0.6319 |
| **Popularity** | **5** | **0.2298** | **0.0398** | **0.2342** | **0.5446** |
| **Popularity** | **10** | **0.2035** | **0.0724** | **0.2171** | **0.6942** |
| **Popularity** | **20** | **0.1710** | **0.1228** | **0.1998** | **0.8198** |

I draw three observations from these numbers. First, Popularity beats Random by roughly a factor of three on Precision and NDCG and by 1.5–2× on Hit Rate. This is the direction I expected (Ricci et al., 2015), and it confirms that the harness produces consistent, comparable numbers across models. Second, Hit Rate@20 reaches 0.82 for Popularity, meaning that for four out of five users at least one of their held-out items appears in the top-20 most popular items within their presentation. That gives me an early sense of how high raw popularity can climb before any personalisation. Third, Precision falls with K for both models — fewer of the top-K items are relevant on average as K grows — while Recall and Hit Rate rise. This is standard ranking-metric behaviour and a useful sanity check.

These numbers set the bar the SVD, content-based, and hybrid models need to clear in the final report.

## Evaluation of the Prototype

Following Rendle et al. (2012) and Ricci et al. (2015), the harness reports Precision@K, Recall@K, NDCG@K, and Hit Rate@K — the standard set for implicit-feedback top-K recommendation. I report all four because no single metric captures recommender quality: Precision can hide poor coverage, and Recall can hide poor ranking. The outcome-weighted Precision metric described in [@sec:design] is *not* yet in the harness. It depends on a feature-engineering step I have not yet built, and it is the principal honest gap between this prototype and the final system.

I chose the two baselines deliberately. Random fixes the floor: any model that fails to beat Random has not learnt anything. Popularity is the strong baseline a useful recommender has to clear (Thai-Nghe et al., 2010; Bousbahi & Chorfi, 2015) — the most-accessed items in a course tend to be its central resources. I enforce reproducibility throughout: Random is seeded and deterministic; the split and candidate sets are persisted to disk; twenty-seven unit tests guard the metric and baseline code; and one command runs the full pipeline from a fresh checkout.

## Limitations and Planned Improvements

The prototype has three principal limitations, all of them deliberate scoping decisions for this submission rather than oversights.

First, **only the baselines are implemented.** The SVD, content-based, and hybrid models are not in this submission. They will use the same interfaces and the same harness, so they will integrate without any architectural change.

Second, **outcome-aware evaluation is not yet in the harness.** Outcome-weighted Precision needs per-item mean assessment scores from the training portion of `studentAssessment`, which in turn needs the feature engineering step I have not built. I have sketched the schema in the design chapter.

Third, **the temporal split is single-fold.** Five-fold temporal cross-validation, as I argued in [@sec:design], is the protocol for the final evaluation. A single fold is enough to show the harness produces sensible numbers; the multi-fold harness is a small extension and is scheduled for the next submission.

The path from prototype to final system is short and well-defined. The remaining work falls into these blocks: SVD collaborative filtering, feature-based content filtering, hybrid ensemble with ablation, FastAPI service and Streamlit dashboard, and final cross-validation and significance testing. The plan is feasible because the technically novel steps each use library implementations with well-known performance characteristics — my contribution is in the combination and evaluation, not in implementing new algorithms from scratch.
