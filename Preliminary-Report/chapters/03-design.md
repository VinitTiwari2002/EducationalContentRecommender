# Design {#sec:design}

## System Overview

The system is a five-stage offline data pipeline whose output is a ranked recommendation list and an evaluation report, fronted by a FastAPI service for interactive querying and a Streamlit dashboard for inspection. The five stages — ingestion, preprocessing, feature engineering, modelling, and evaluation — are organised as independent components, each consuming the output of the previous stage and persisting its own artefacts for downstream reuse. Figure 3.1 shows the high-level flow.

![Figure 3.1 — Pipeline architecture: from raw OULAD CSVs through preprocessing, modelling, and evaluation.](figures/fig_3_1_pipeline.png){width=95%}

The architectural choices that follow from this layout are deliberate. Separation of ingestion from preprocessing means the raw CSVs are treated as immutable inputs; all derived artefacts are reproducible from a clean checkout. Separation of preprocessing from modelling means every model consumes the same matrices and is therefore directly comparable. Separation of modelling from evaluation means new models can be added without touching the evaluation harness. The intent is that a marker can pull the repository, run a single command, and reproduce every number reported in this submission.

## Data Design

The project uses the Open University Learning Analytics Dataset (OULAD; Kuzilek, Hlosta & Zdrahal, 2017), distributed under a CC-BY licence and obtainable from the UCI ML Repository or Kaggle without registration. OULAD is the only public dataset that links *click-level* interaction with *graded assessment outcomes* and *demographic* metadata at the individual learner level, which makes it uniquely suited to the project's outcome-aware evaluation strategy. Table 3.1 summarises the seven CSV tables used.

| Table | Records | Role in pipeline |
|---|---:|---|
| `studentVle` | ~10.6 M | User-item interactions (sum_click weights) |
| `studentAssessment` | ~173 K | Outcome signal (assessment score per student per task) |
| `studentInfo` | ~32 K | Demographics; DEI analysis cohorts |
| `studentRegistration` | ~32 K | Course registration windows; defines active period |
| `vle` | ~6.4 K | Items (catalogue) with `activity_type` taxonomy |
| `assessments` | 206 | Assessment metadata (type, weight, due date) |
| `courses` | 22 | Course/presentation identifiers |

*Table 3.1 — OULAD tables and their role in the pipeline.*

The unit of recommendation is the VLE `id_site` (an individual learning resource). The user-item interaction is constructed from `studentVle` by summing `sum_click` per `(id_student, id_site, code_module, code_presentation)`, producing a sparse matrix of approximately 32 K × 6.4 K. Item-level features are constructed from `vle` (`activity_type` one-hot encoded; `week_from`, `week_to` normalised) and enriched from `studentVle` and `studentAssessment` with two behavioural features: mean assessment-score of the students who interacted with each item, and the global access count. These two derived features are computed *strictly from the training portion of the temporal split* to avoid leakage into the test set — a methodological point that, following Ricci et al. (2015), is critical for honest offline evaluation.

## Recommendation Process — Worked Example

A central piece of feedback on the proposal was a request for clearer visual examples of how the recommendation process works. Figure 3.2 illustrates the end-to-end transformation for a single learner.

![Figure 3.2 — Recommendation process for a real OULAD student: input profile $\rightarrow$ feature extraction $\rightarrow$ model scoring $\rightarrow$ top-10 ranked output. Generated from the prototype against OULAD; the top-10 panel is illustrative since the hybrid model has not yet been trained.](figures/fig_3_2_recommendation_example.png){width=98%}

Concretely, the recommender produces, for each `(student, candidate_item)` pair, a score $s(u, i) = \alpha \cdot s_{\text{CF}}(u,i) + \beta \cdot s_{\text{content}}(u,i) + \gamma \cdot s_{\text{outcome}}(i)$, where the three components are the SVD reconstruction, the cosine similarity between the item's feature vector and the learner's average accessed-item profile, and the item's training-set outcome correlation. Weights $\alpha, \beta, \gamma$ are tuned on a validation fold. The top-$K$ items not previously accessed by the learner are returned. Figure 3.3 shows the model decomposition.

![Figure 3.3 — Hybrid score decomposition: per-candidate contributions from collaborative, content, and outcome components, with the total score determining the recommendation rank.](figures/fig_3_3_score_decomposition.png){width=95%}

The dashboard surfaces these decompositions so the recommendations are auditable: a marker can see *why* each item was recommended, not just *that* it was.

## Model Design

The system implements five recommenders behind a common `Recommender` interface. The interface is shared deliberately so that every model is evaluated by the same harness on the same split.

1. **Random.** Samples *K* items uniformly from the catalogue not in the student's training history. Establishes the floor.
2. **Popularity.** Returns the *K* most-accessed items globally (training portion only). A surprisingly strong educational baseline because the most popular VLE items in OULAD tend to be the course's central resources.
3. **Collaborative (SVD).** Truncated SVD on the centred interaction matrix. Hyperparameters: number of latent factors $k \in \{20, 50, 100\}$ and regularisation $\lambda \in \{0.01, 0.05, 0.1\}$, tuned by 5-fold CV. Based on Koren, Bell & Volinsky (2009).
4. **Content-Based.** Cosine similarity over item feature vectors. Each learner's profile is the mean of the feature vectors of items they accessed in training. Items are ranked by similarity to the profile. Justified by Bousbahi & Chorfi (2015), and chosen specifically to handle the cold-start case where SVD has insufficient interaction signal.
5. **Hybrid.** Weighted ensemble of SVD and content scores, with an optional outcome-weighting term. Weights are tuned by grid search on a validation fold.

The progression is deliberate: each step adds a single concept (popularity $\rightarrow$ personalisation $\rightarrow$ content awareness $\rightarrow$ outcome awareness) so that the contribution of each component is isolatable in the final ablation study.

## Evaluation Strategy

The marker feedback on the proposal explicitly asked for clearer explanation of how recommendation accuracy and learning improvement will be evaluated. This section addresses that directly and is structured around three questions: what is measured, against what, and with what statistical confidence.

**What is measured.** Five metrics are computed for every model.

| Metric | Definition | Why included |
|---|---|---|
| Precision@10 | Fraction of top-10 recommendations that appear in the held-out interaction set | Standard recommender quality |
| Recall@10 | Fraction of held-out interactions captured in top-10 | Coverage |
| NDCG@10 | Discounted-cumulative-gain at 10, normalised to ideal ranking | Ranking quality (primary; follows Rendle et al., 2012) |
| Hit Rate@10 | Fraction of users for whom at least one held-out item appears in top-10 | User-level success |
| Outcome-weighted Precision@10 | Precision@10 weighted by the mean assessment score of training-set users who accessed each recommended item | Outcome-awareness — the project's principal contribution |

The first four are standard and benchmarked against the prior literature. The fifth is the project's outcome-aware contribution and operationalises "learning improvement" in the only way available offline: by measuring whether recommended items are correlated with higher assessment scores in held-out training data.

**Against what.** Every model is compared against (i) the Random and Popularity baselines, which establish floor and naive-strong performance respectively, and (ii) each preceding model in the progression. The hybrid's improvement over its components is the principal hypothesis test of the project.

**With what statistical confidence.** Evaluation is performed under 5-fold temporal cross-validation. For each fold the data is split chronologically — *not* randomly — into 80% training and 20% test, with the cutoff date advanced across folds. Statistical significance of model A over model B is assessed by paired *t*-test of per-fold metrics with Bonferroni correction across the five metrics. An ablation study disables each component of the hybrid in turn and reports the resulting metric degradation.

The link between this evaluation strategy and the literature reviewed in [@sec:lit-review] is explicit. Temporal splitting follows Ricci et al. (2015). NDCG as the primary metric follows the implicit-feedback argument of Rendle et al. (2012). The choice to evaluate ranking quality rather than rating prediction is informed by the critique of pure-prediction evaluation that Thai-Nghe et al. (2010) leave unaddressed, and by the methodological caution that Wilson et al.'s critique of DKT (2016) imposes on any model claiming improvement.

## Work Plan

The project follows a 24-week plan, of which weeks 1–10 are complete or in progress at the time of this submission. Table 3.2 summarises the schedule.

| Wk | Phase | Deliverable | Status |
|---:|---|---|---|
| 1–4 | Proposal | Proposal video + plan | Done — Submitted |
| 5–6 | EDA + repo setup | Notebook with key OULAD findings | Done — In this submission |
| 7–8 | Preprocessing | Interaction matrix, feature vectors, temporal split | Done — In this submission |
| 9–10 | Baselines + Prelim report | Random + Popularity + evaluation harness | Done — This submission |
| 11–12 | Collaborative filtering | SVD with hyperparameter tuning | Planned |
| 13–14 | Content-based | Feature-similarity recommender | Planned |
| 15–16 | Hybrid + ablation | Weighted ensemble + significance tests | Planned |
| 17–18 | API + dashboard + draft report | FastAPI + Streamlit + draft submission | Planned |
| 19–20 | Final evaluation | Cross-validation, all figures, ablation | Planned |
| 21–22 | Report writing + exam | Final report drafts; exam preparation | Planned |
| 23–24 | Video + final submission | MP4 demo; submission-ready code | Planned |

*Table 3.2 — 24-week project plan. Total committed effort 150 hours, of which approximately 28 are spent at the time of this submission; remaining work is paced at ~7.5 h/week with a 14-hour reserve.*

The plan is feasible because the technically novel steps (SVD, content similarity, hybrid weighting) each use library implementations with established performance characteristics; the project's contribution is in their combination and evaluation, not in implementing new algorithms from scratch.

## Ethics, DEI, and Risks

OULAD contains sensitive demographic attributes — gender, age band, region, IMD (Index of Multiple Deprivation) band, disability flag — for every student. The project handles these as follows. First, recommendations are never conditioned on demographic attributes directly; the model sees only behavioural and content features. Second, the evaluation report includes a *fairness audit*: per-metric performance is broken down by gender, IMD band, and disability flag, to surface any group for whom the recommender systematically underperforms. This is consistent with the recommendation-system fairness literature and addresses the project's DEI learning objective in a concrete and measurable way. Third, no demographic data is published or shared beyond aggregate metrics in the report; the raw OULAD CSVs are listed in `.gitignore` and are not committed to the repository.

The principal risks to delivery are (i) the hybrid model failing to improve significantly over SVD alone, in which case the project's evaluation chapter will document this as a valid empirical finding rather than a failure of method; (ii) sparsity producing unstable CF metrics for cold-start users, which is the explicit motivation for the content-based component and is mitigated by reporting cold-start results separately.
