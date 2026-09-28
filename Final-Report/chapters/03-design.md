# Design {#sec:design}

## System Overview

The system is a five-stage offline data pipeline — ingestion, preprocessing, feature engineering, modelling, evaluation — fronted by a FastAPI service and a Streamlit dashboard. Each stage consumes the previous stage's artefact and persists its own for downstream reuse. Figure 3.1 shows the flow.

![Figure 3.1 — Pipeline architecture: from raw OULAD CSVs through preprocessing, modelling, and evaluation.](figures/fig_3_1_pipeline.png){width=95%}

Immutable artefacts between stages ensure like-for-like model comparison and single-command reproducibility.

## Data Design

I use OULAD (Kuzilek, Hlosta & Zdrahal, 2017), CC-BY licensed, obtainable from UCI. It is the only public dataset I am aware of that links *click-level* interaction with *graded assessment outcomes* and *demographic* metadata at learner level. Table 3.1 summarises the seven CSV tables.

| Table | Records | Role in pipeline |
|---|---:|---|
| `studentVle` | ~10.6 M | User-item interactions (sum_click weights) |
| `studentAssessment` | ~173 K | Outcome signal (score per student per task) |
| `studentInfo` | ~32 K | Demographics; fairness-audit cohorts |
| `studentRegistration` | ~32 K | Course registration windows; defines active period |
| `vle` | ~6.4 K | Items (catalogue) with `activity_type` taxonomy |
| `assessments` | 206 | Assessment metadata (type, weight, due date) |
| `courses` | 22 | Course/presentation identifiers |

*Table 3.1 — OULAD tables and their role in the pipeline.*

The unit of recommendation is `id_site` (a VLE learning resource). The user-item matrix aggregates `sum_click` per `(id_student, id_site)` within the training window into a sparse ~32K × 6.4K CSR. Item features are drawn from `vle` (`activity_type` one-hot; normalised `week_from`, `week_to`) and enriched with two behavioural features from `studentVle` and `studentAssessment`: mean assessment score of training-window accessers, and log-scaled global access count. All derived features are computed strictly from the training portion (Ricci et al., 2015).

**Course-scoping** restricts each user's candidate pool to items from presentations they registered for — each `id_site` maps to one `(code_module, code_presentation)`, and a student registers for ~1.1 presentations on average. The preliminary report documents the pathological Popularity@10 = 0.00003 result before this constraint was added; it is a methodological finding, not merely an engineering fix.

**Temporal decay** weights training clicks by `exp(-λ × (cutoff − date))` — a pedagogical prior that a click three weeks before assessment is more informative than one three months before. Empirical evaluation motivated adoption; λ = 0 remains a diagnostic baseline.

## Recommendation Process — Worked Example

Figure 3.2 walks through the end-to-end transformation for a single learner.

![Figure 3.2 — Recommendation process for a real OULAD student: input profile $\rightarrow$ feature extraction $\rightarrow$ model scoring $\rightarrow$ top-10 ranked output.](figures/fig_3_2_recommendation_example.png){width=98%}

The recommender scores each `(student, candidate_item)` pair as $s(u, i) = \alpha \cdot s_{\text{CF}}(u,i) + \beta \cdot s_{\text{content}}(u,i) + \gamma \cdot s_{\text{outcome}}(i)$: the CF reconstruction (SVD or ALS), cosine similarity between the item's feature vector and the learner's click-weighted profile, and the item's training-window mean outcome score. Each component is min-max normalised within the candidate pool so weights are directly interpretable, tuned on a temporal validation sub-split, and the top-$K$ un-accessed items are returned. Figure 3.3 shows a per-candidate decomposition.

![Figure 3.3 — Hybrid score decomposition: per-candidate contributions from collaborative, content, and outcome components, with the total score determining the recommendation rank.](figures/fig_3_3_score_decomposition.png){width=95%}

The dashboard surfaces this decomposition so the recommendations are auditable: a user can see *why* each item was recommended, not just *that* it was.

## Model Design

Seven recommenders sit behind a common `Recommender` interface — `fit`, `recommend(user, k, candidates)`, `score(user)`. The progression is deliberate: each step adds one concept, so the ablation can isolate its contribution.

1. **Random** — uniform sample; establishes the floor.
2. **Popularity** — top-*K* by training click totals; strong because course-central VLE items dominate.
3. **SVD** — truncated SVD on `log1p(train)` following Koren, Bell & Volinsky (2009); rank tuned.
4. **ALS** — implicit-feedback ALS with confidence $c(u,i) = 1 + \alpha \cdot \text{clicks}$ (Hu, Koren & Volinsky, 2008); `n_factors × regularisation × α` tuned.
5. **Content** — cosine similarity over item features (Bousbahi & Chorfi, 2015); profile = click-weighted mean of accessed-item features.
6. **Hybrid** — Burke-weighted ensemble (Burke, 2002) of CF + content + outcome; weights tuned on a temporal validation sub-split.
7. **GatedHybrid** — Burke-switching hybrid: cold users to Popularity, warm to Hybrid. Introduced after the ablation surfaced Popularity's cold-user lead.

An eighth model, the LambdaMART two-stage reranker (§4.6), sits *on top of* the Hybrid rather than in the progression: Stage 1 retrieves the top-N from GatedHybrid, Stage 2 rescores with a LightGBM ranker trained under an honest three-window design.

## Service Design

The service layer exposes the recommender as a queryable artefact. **FastAPI** serves three read-only endpoints (`/health`, `/recommend/{student_id}`, `/decompose/{student_id}/{item_id}`), loading persisted model artefacts at startup so per-request cost is O(1). **Streamlit** layers a three-page transparency dashboard on top of the API: (1) *Recommendation Explorer* — per-student top-*K* with a bar-chart decomposition of the hybrid score and a per-item audit view; (2) *Fairness View* — per-attribute metric breakdown across gender, IMD band, and disability, driven by the fairness-audit CSV; (3) *Ablation Comparison* — side-by-side top-*K* for Content vs Hybrid vs GatedHybrid on the same student. Chapter 4 documents the delivered implementation (§4.9) and the two-stage Docker image that packages both services for a marker's clean-environment run.

## Evaluation Strategy

Six metrics per model at $K = 10$: Precision, Recall, NDCG, Hit-Rate, outcome-weighted Precision ([@sec:evaluation]), plus catalogue coverage and Gini for popularity bias. Statistical confidence comes from five-fold temporal cross-validation with sliding cutoff dates, 95% bootstrap CIs on per-user metric means, paired-*t* with Bonferroni adjustment across the five metrics, an ablation disabling each hybrid component in turn, cold-start stratification below/above a training-click threshold, and a fairness audit across gender, IMD band, and disability. Every split — primary, tuning sub-split, and each CV fold — is temporal (chronological cutoff), user rows are shared across `train`/`test` matrices, and course-scoping restricts candidates identically in every window so no split leaks across presentations.

## Ethics, DEI, and Outcome-Signal Bias

OULAD contains sensitive demographic attributes (gender, age band, region, IMD band, disability flag). Recommendations are not conditioned on these directly — the model sees only behavioural and content features. A fairness audit in the evaluation reports per-metric performance broken down by gender, IMD band, and disability, so any group the recommender systematically underperforms on is visible. Demographic data is not shared beyond aggregate metrics.

**Outcome-signal bias.** The `mean_score_of_accessers` feature is observational: it captures the score of learners who *chose* to access an item. Course-central resources therefore benefit twice — from popularity (high click count) and from self-selection (their accessers skew toward higher-scoring cohorts). I treat this as a design trade-off rather than a bug: the evaluation reports catalogue coverage and recommendation-Gini at every K (§5.8) so the trade-off is inspectable, and the conclusion earmarks a fairness-aware re-ranking layer that would use the score decomposition already emitted by `/decompose` to penalise items whose ranking is dominated by the outcome component.

## Workplan and Risk Register

The project runs across twenty calendar weeks, from **10 May 2026** (project start) to **28 September 2026** (CM3070 final-report submission deadline), with the interim **19 August 2026** draft-report submission at the end of Week 15. Figure 3.4 shows the Gantt across the whole span; completed phases are hatched, remaining phases solid, and the red dashed line marks the final deadline. Table 3.2a records the completed milestones; Table 3.2b records the remaining ones with dependencies, primary risks, and contingencies. Retrospective effort is quoted as *spent*; forward effort as *budgeted*.

![Figure 3.4 — Workplan Gantt: 10 May 2026 — 28 September 2026 (twenty weeks). Hatched bars are completed; the red dashed line marks the 28 Sep final submission. The Draft-report submission (19 Aug) sits at the boundary between the completed retrospective and the remaining forward work.](figures/fig_3_4_workplan_gantt.png){width=98%}

| Weeks (dates) | Milestone | Deliverable | Effort |
|---|---|---|---:|
| 1–3 (10 May – 30 May) | Project setup + proposal | Repository scaffold; proposal PDF; proposal video | ~35 h |
| 4–7 (31 May – 27 Jun) | Data ingestion + EDA + baselines | OULAD loader; sparse matrix + temporal split; EDA notebook; Random + Popularity | ~45 h |
| 8 (28 Jun – 4 Jul) | Preliminary report submission | Prelim PDF (4 chapters, ~4,500 w) + prototype video | ~25 h |
| 9–12 (5 Jul – 1 Aug) | Post-feedback core implementation | SVD, Content, Hybrid recommenders + 25 unit tests; outcome-weighted precision v1 | ~50 h |
| 13–14 (2 Aug – 17 Aug) | Advanced experiments | ALS; simplex-grid hybrid tuning; GatedHybrid; time-decay; 5-fold temporal CV; bootstrap CIs; paired-*t* + Bonferroni; ablation; cold-start; fairness; popularity bias | ~50 h |
| 15 (18 Aug – 19 Aug) | Draft report submission | 6-chapter draft PDF + Gantt figure | ~15 h |
| 16 (20 Aug – 26 Aug) | FastAPI service | `/health`, `/recommend`, `/decompose` + persistence layer, 12 unit tests | ~15 h |
| 17 (27 Aug – 2 Sep) | Streamlit dashboard + Docker | 3-page dashboard on the FastAPI service; two-stage Docker image | ~15 h |
| 18 (3 Sep – 9 Sep) | LambdaMART reranker + robustness | 8th recommender (§4.6), three-seed robustness sweep, real-hybrid Figs 3.2–3.3 | ~15 h |

*Table 3.2a — Completed milestones (Weeks 1–18, 10 May – 9 Sep). Total retrospective effort ≈ 265 hours.*

| Weeks (dates) | Milestone | Deliverable | Effort | Depends on | Risk & contingency |
|---|---|---|---:|---|---|
| 19 (10 Sep – 18 Sep) | Draft-feedback incorporation + final-report writing | All 6 chapters finalised against marker feedback; word-count trim to ≤ 10,500 w | ~25 h | Draft feedback received | Late feedback — start immediately, hold Week 20 pure for video |
| 20 (19 Sep – 25 Sep) | Video (3–5 min MP4) + PDF polish | Demo video; final PDF ≤ 15 MB; screenshots refreshed | ~15 h | All chapters green | Video overrun — script *before* recording |
| — (26 Sep – 27 Sep) | Buffer + submission dry-run | Clean-env Docker rebuild verification; end-to-end submission dry-run | ~4 h | Video ready | Portal issues — submit ≥ 24 h early |
| — (28 Sep) | **Submit final report** | Coursera upload | ~2 h | All above complete | — |

*Table 3.2b — Remaining milestones (Weeks 19–20, 10 – 28 Sep). Total forward effort ≈ 45 hours across two weeks.*

**Slack allocation.** Each weekly bar covers seven days but is designed for five weekdays of effort, giving a two-day per-week rolling buffer that absorbs slip without cascading.

**Cut-line policy — retrospective.** Two items originally deferred to future work moved *back* into the delivered scope during Weeks 16–18: the FastAPI service + dashboard (previously stretch) and the LambdaMART two-stage reranker (previously listed as post-project). The knowledge-tracing component and the grade-prediction diagnostic remain in [@sec:conclusion] as future work.

**Top risks for the remaining two weeks.** (i) Video overrun of the 5-minute cap — cited in prelim feedback — mitigated by scripting *before* recording; (ii) `implicit` install failure on a marker's machine — mitigated by pinned deps and the two-stage Docker image (§4.9), whose runtime stage is verified against a clean container in the pre-submission dry-run.
