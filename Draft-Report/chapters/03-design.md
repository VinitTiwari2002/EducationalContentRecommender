# Design {#sec:design}

## System Overview

The system is a five-stage offline data pipeline whose output is a ranked recommendation list and an evaluation report, fronted by a FastAPI service for interactive querying and a Streamlit dashboard for inspection. The five stages — ingestion, preprocessing, feature engineering, modelling, and evaluation — are organised as independent components, each consuming the output of the previous stage and persisting its own artefacts for downstream reuse. Figure 3.1 shows the flow.

![Figure 3.1 — Pipeline architecture: from raw OULAD CSVs through preprocessing, modelling, and evaluation.](figures/fig_3_1_pipeline.png){width=95%}

The architectural choices are deliberate: separating ingestion, preprocessing, modelling, and evaluation means each stage produces an immutable artefact the next consumes, every model is compared like-for-like, and a marker can pull the repository, run one command, and reproduce every number reported here.

## Data Design

I use the Open University Learning Analytics Dataset (OULAD; Kuzilek, Hlosta & Zdrahal, 2017), distributed under a CC-BY licence and obtainable from the UCI ML Repository. OULAD is, so far as I am aware, the only public dataset that links *click-level* interaction with *graded assessment outcomes* and *demographic* metadata at the individual learner level, which is what makes it suitable for my outcome-aware evaluation strategy. Table 3.1 summarises the seven CSV tables I use.

| Table | Records | Role in pipeline |
|---|---:|---|
| `studentVle` | ~10.6 M | User-item interactions (sum_click weights) |
| `studentAssessment` | ~173 K | Outcome signal (assessment score per student per task) |
| `studentInfo` | ~32 K | Demographics; fairness-audit cohorts |
| `studentRegistration` | ~32 K | Course registration windows; defines active period |
| `vle` | ~6.4 K | Items (catalogue) with `activity_type` taxonomy |
| `assessments` | 206 | Assessment metadata (type, weight, due date) |
| `courses` | 22 | Course/presentation identifiers |

*Table 3.1 — OULAD tables and their role in the pipeline.*

The unit of recommendation is the VLE `id_site` (an individual learning resource). I build the user-item interaction matrix by aggregating `sum_click` per `(id_student, id_site)` within the training window, producing a sparse ~32K × 6.4K matrix. Item-level features are drawn from `vle` (`activity_type` one-hot; `week_from`, `week_to` normalised) and enriched from `studentVle` and `studentAssessment` with two behavioural features: the mean assessment score of training-set students who accessed each item, and the log-scaled global access count. All derived features are computed strictly from the *training* portion of the temporal split, so no future information leaks into features (Ricci et al., 2015).

An explicit *course-scoping* constraint restricts each user's candidate pool to items from presentations they are enrolled in — each `id_site` belongs to exactly one `(code_module, code_presentation)`, and a student is enrolled in ~1.1 presentations on average. The preliminary report documents the pathological Popularity@10 = 0.00003 result that appeared before this constraint was added; it is a methodological finding, not just an engineering fix.

A second design choice is *temporal decay*: training-window clicks are weighted by `exp(-λ × (cutoff − date))`, giving recent clicks weight ≈1 and older clicks weight $\rightarrow 0$. This encodes the pedagogical prior that a click three weeks before assessment is more informative than one three months before. Empirical evaluation ([@sec:evaluation]) motivated adopting it; the raw-click case (λ = 0) is retained as a diagnostic baseline.

## Recommendation Process — Worked Example

Figure 3.2 walks through the end-to-end transformation for a single learner.

![Figure 3.2 — Recommendation process for a real OULAD student: input profile $\rightarrow$ feature extraction $\rightarrow$ model scoring $\rightarrow$ top-10 ranked output.](figures/fig_3_2_recommendation_example.png){width=98%}

Concretely, the recommender produces a score for each `(student, candidate_item)` pair as $s(u, i) = \alpha \cdot s_{\text{CF}}(u,i) + \beta \cdot s_{\text{content}}(u,i) + \gamma \cdot s_{\text{outcome}}(i)$. The three components are the CF reconstruction (SVD or ALS), the cosine similarity between the item's feature vector and the learner's click-weighted mean of accessed-item features, and the item's training-window mean outcome score. Each component is min-max normalised across the candidate pool before combination so weights $\alpha, \beta, \gamma$ are directly interpretable as relative importances. The weights are tuned on a temporal validation sub-split of the training window, and the top-$K$ items the learner has not already accessed are returned. Figure 3.3 shows a per-candidate decomposition.

![Figure 3.3 — Hybrid score decomposition: per-candidate contributions from collaborative, content, and outcome components, with the total score determining the recommendation rank.](figures/fig_3_3_score_decomposition.png){width=95%}

The dashboard surfaces this decomposition so the recommendations are auditable: a user can see *why* each item was recommended, not just *that* it was.

## Model Design

Seven recommenders sit behind a common `Recommender` interface — `fit`, `recommend(user, k, candidates)`, `score(user)` — so every model is evaluated by the same harness on the same split.

1. **Random.** Samples $K$ items uniformly from the candidate pool; establishes the floor.
2. **Popularity.** Returns the $K$ most-accessed items (training only); surprisingly strong because course-central VLE items dominate the click distribution.
3. **Collaborative — SVD.** Truncated SVD on the log-transformed click matrix, following Koren, Bell & Volinsky (2009); n_factors tuned in $\{20, 50, 100\}$.
4. **Collaborative — ALS.** Alternating Least Squares with confidence weighting $c(u, i) = 1 + \alpha \cdot \text{clicks}(u, i)$ (Hu, Koren & Volinsky, 2008), via the `implicit` library; tuned over n_factors × regularisation × α.
5. **Content-Based.** Cosine similarity over the item feature vectors of Section 3.2 (Bousbahi & Chorfi, 2015); each learner's profile is the click-weighted mean of accessed-item features. Included for cold-start.
6. **Hybrid.** Burke-weighted ensemble (Burke, 2002) of CF, content, and outcome components. Weights tuned by simplex grid search on a validation sub-split.
7. **GatedHybrid.** Burke-switching hybrid: users with fewer than *N* training clicks are routed to Popularity, otherwise to the tuned Hybrid. Introduced after ablation showed Popularity outperforms Hybrid on cold users.

The progression is deliberate: each step adds a single concept (popularity $\rightarrow$ collaborative personalisation $\rightarrow$ content awareness $\rightarrow$ outcome awareness $\rightarrow$ cold-start handling), so I can isolate the contribution of each component in the ablation.

## Service Design

The service layer exposes the recommender as a queryable artefact for markers. It has two components on top of the pipeline.

**FastAPI service.** Three read-only endpoints.

- `GET /recommend/{student_id}?k=10&model=hybrid` — top-*K* item IDs for a student with per-item activity_type, mean outcome, and hybrid decomposition. `model` accepts any of the seven registered recommenders.
- `GET /decompose/{student_id}/{item_id}` — the (CF, content, outcome) triple for a single pair; used by the dashboard's audit view.
- `GET /health` — currently-loaded roster + split cutoff date.

The service loads persisted model artefacts at startup, so per-request cost is O(1). Responses are JSON; no authentication — this is a coursework demonstration on aggregated data.

**Streamlit dashboard.** Three pages layered on the FastAPI service.

1. *Recommendation Explorer.* Student selector (dropdown over the test set), *K* slider (1–20), model selector, and a top-*K* table showing each item with its activity type, its aggregate mean outcome, and a bar-chart decomposition of its hybrid score. Clicking a row opens the score-decomposition view.
2. *Fairness View.* Per-attribute metric breakdown across gender, IMD band and disability, plotted as a grouped bar chart with the fairness-audit CSV underneath. Toggles switch between metrics.
3. *Ablation Comparison.* Side-by-side comparison of Hybrid, Content, and GatedHybrid on the currently-selected student, with each variant's top-*K* list and where they diverge.

The dashboard is a *transparency* artefact: it lets a marker inspect recommendations and understand why the system produced them.

## Evaluation Strategy

**What I measure.** Six metrics per model at $K = 10$: Precision@K, Recall@K, NDCG@K, Hit-Rate@K, outcome-weighted Precision@K (defined precisely in [@sec:evaluation]), and catalogue coverage + Gini for popularity bias. The outcome-weighted precision is my principal contribution and operationalises "learning benefit" as far as offline evaluation allows.

**Against what.** Every model is compared against Random and Popularity, and against each preceding model in the progression.

**With what statistical confidence.** Five-fold temporal cross-validation with sliding cutoff dates; 95% bootstrap CIs on per-user metric means; paired *t*-tests with Bonferroni adjustment across the five metrics; an ablation study disabling each hybrid component in turn; cold-start stratification below/above a training-clicks threshold; and a fairness audit across gender, IMD band, and disability.

## Ethics, DEI, and Risks

OULAD contains sensitive demographic attributes (gender, age band, region, IMD band, disability flag). Recommendations are not conditioned on these directly — the model sees only behavioural and content features. A fairness audit in the evaluation reports per-metric performance broken down by gender, IMD band, and disability, so any group for which the recommender systematically underperforms is visible. Demographic data is not shared beyond aggregate metrics.

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

*Table 3.2a — Completed milestones (Weeks 1–14, 10 May – 17 Aug). Total retrospective effort ≈ 205 hours.*

| Weeks (dates) | Milestone | Deliverable | Effort | Depends on | Risk & contingency |
|---|---|---|---:|---|---|
| 15 (18 Aug – 19 Aug) | **Draft report submission** | This 6-chapter PDF + Gantt figure | ~15 h | All experiments captured in `evaluation/*.csv` | Word overshoot — rely on +10% total allowance |
| 16 (20 Aug – 26 Aug) | FastAPI service scaffold | `/recommend`, `/decompose`, `/health` + persisted model artefacts | ~15 h | Draft submitted | `implicit` install failure — pin deps; provide Docker image |
| 17 (27 Aug – 2 Sep) | Streamlit dashboard MVP | 3-page dashboard on FastAPI | ~15 h | Service scaffold live | CORS friction — serve dashboard on `/dashboard` route |
| 18 (3 Sep – 9 Sep) | Figures + robustness re-runs | Real-hybrid Figs 3.2–3.3; pipeline re-runs | ~15 h | Dashboard MVP complete | Numerical drift — fix seeds; report medians over 3 runs |
| 19 (10 Sep – 18 Sep) | Final report writing + peer review | All 6 chapters finalised; peer feedback incorporated | ~20 h | Robustness runs done; draft feedback received | Late feedback — start revisions when it lands, not at end of Week 19 |
| 20 (19 Sep – 25 Sep) | Video (3–5 min MP4) + PDF polish | Demo video; final PDF ≤ 15 MB | ~15 h | All chapters green | Video overrun — script *before* recording |
| — (26 Sep – 27 Sep) | Buffer + submission dry-run | End-to-end submission dry-run on a marker-like environment | ~4 h | Video ready | Portal issues — submit ≥ 24 h early |
| — (28 Sep) | **Submit final report** | Coursera upload | ~2 h | All above complete | — |

*Table 3.2b — Remaining milestones (Weeks 15–20, 18 Aug – 28 Sep). Total forward effort ≈ 100 hours over six weeks (~17 h/week), consistent with a part-time student's capacity.*

**Slack allocation.** Each weekly bar covers seven days but is designed for five weekdays of effort, giving a two-day per-week rolling buffer that absorbs slip without cascading. The earliest single-milestone slip that would threaten the deadline requires losing more than two full weeks — implausible given the effort estimates and the retrospective evidence that most weeks landed within budget.

**Cut-line policy.** Three items originally floated as stretch work (LambdaMART re-rank; grade-prediction diagnostic; knowledge-tracing component) were pushed to future work in the conclusion once the six-week forward budget was clear. The remaining plan protects marker-facing artefacts (service, dashboard, evaluation completeness) at the cost of stretch technical work.

**Top risks for the remaining six weeks.** (i) Video overrun of the 5-minute cap — cited in prelim feedback — mitigated by scripting *before* recording; (ii) draft-report feedback returning in the final 48 hours before the final submission, mitigated by starting Week 19's revision pass immediately when feedback lands rather than waiting for the peer round; (iii) `implicit` install failure on a marker's machine, mitigated by pinned deps and a Docker image.
