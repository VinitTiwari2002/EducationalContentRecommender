# Design {#sec:design}

## System Overview

I have designed the system as a five-stage offline data pipeline. Its output is a ranked recommendation list and an evaluation report, fronted by a FastAPI service for interactive querying and a Streamlit dashboard for inspection. The five stages — ingestion, preprocessing, feature engineering, modelling, and evaluation — are organised as independent components, each consuming the output of the previous stage and persisting its own artefacts for downstream reuse. Figure 3.1 shows the high-level flow.

![Figure 3.1 — Pipeline architecture: from raw OULAD CSVs through preprocessing, modelling, and evaluation.](figures/fig_3_1_pipeline.png){width=95%}

The architectural choices behind this layout are deliberate. Separating ingestion from preprocessing means I treat the raw CSVs as immutable inputs; every derived artefact can be reproduced from a clean checkout. Separating preprocessing from modelling means every model consumes the same matrices and can therefore be compared like-for-like. Separating modelling from evaluation means I can add a new model without touching the evaluation harness. My intent is that a marker can pull the repository, run a single command, and reproduce every number reported in this submission.

## Data Design

I use the Open University Learning Analytics Dataset (OULAD; Kuzilek, Hlosta & Zdrahal, 2017), distributed under a CC-BY licence and obtainable from the UCI ML Repository. OULAD is, as far as I am aware, the only public dataset that links *click-level* interaction with *graded assessment outcomes* and *demographic* metadata at the individual learner level, which is what makes it suitable for my outcome-aware evaluation strategy. Table 3.1 summarises the seven CSV tables I use.

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

My unit of recommendation is the VLE `id_site` (an individual learning resource). I build the user-item interaction by summing `sum_click` per `(id_student, id_site, code_module, code_presentation)`, which produces a sparse matrix of roughly 32K × 6.4K. I build item-level features from `vle` (`activity_type` one-hot encoded; `week_from`, `week_to` normalised) and enrich them from `studentVle` and `studentAssessment` with two behavioural features: the mean assessment score of students who interacted with each item, and the global access count. I compute both of these strictly from the training portion of the temporal split, to avoid leaking future information into the test set — following Ricci et al. (2015), this is essential for honest offline evaluation.

## Recommendation Process — Worked Example

Figure 3.2 walks through the end-to-end transformation for a single learner.

![Figure 3.2 — Recommendation process for a real OULAD student: input profile $\rightarrow$ feature extraction $\rightarrow$ model scoring $\rightarrow$ top-10 ranked output. Generated from the prototype against OULAD; the top-10 panel is illustrative since the hybrid model has not yet been trained.](figures/fig_3_2_recommendation_example.png){width=98%}

Concretely, the recommender produces a score for each `(student, candidate_item)` pair as $s(u, i) = \alpha \cdot s_{\text{CF}}(u,i) + \beta \cdot s_{\text{content}}(u,i) + \gamma \cdot s_{\text{outcome}}(i)$. The three components are the SVD reconstruction, the cosine similarity between the item's feature vector and the learner's average accessed-item profile, and the item's training-set outcome correlation. I tune the weights $\alpha, \beta, \gamma$ on a validation fold and return the top-$K$ items the learner has not already accessed. Figure 3.3 shows the decomposition.

![Figure 3.3 — Hybrid score decomposition: per-candidate contributions from collaborative, content, and outcome components, with the total score determining the recommendation rank.](figures/fig_3_3_score_decomposition.png){width=95%}

I surface these decompositions in the dashboard so the recommendations are auditable: we can see *why* each item was recommended, not just *that* it was.

## Model Design

I implement five recommenders behind a common `Recommender` interface, so that every model is evaluated by the same harness on the same split.

1. **Random.** Samples *K* items uniformly from the catalogue, not from the student's training history. Establishes the floor.
2. **Popularity.** Returns the *K* most-accessed items globally (training portion only). A surprisingly strong baseline in education, because the most popular VLE items in OULAD tend to be the course's central resources.
3. **Collaborative (SVD).** Truncated SVD on the centred interaction matrix. I plan to tune the number of latent factors $k \in \{20, 50, 100\}$ and regularisation $\lambda \in \{0.01, 0.05, 0.1\}$ by 5-fold CV, following Koren, Bell & Volinsky (2009).
4. **Content-Based.** Cosine similarity over item feature vectors. Each learner's profile is the mean of the feature vectors of items they accessed in training. Items are ranked by similarity to that profile. I include this following Bousbahi & Chorfi (2015), specifically for the cold-start case where SVD has too little interaction signal.
5. **Hybrid.** Weighted ensemble of SVD and content scores, with an optional outcome-weighting term. I will tune the weights by grid search on a validation fold.

The progression is deliberate: each step adds a single concept (popularity $\rightarrow$ personalisation $\rightarrow$ content awareness $\rightarrow$ outcome awareness), so that I can isolate the contribution of each component in the final ablation study.

## Evaluation Strategy

I structure this section around three questions: what I measure, what I compare against, and with what statistical confidence.

**What I measure.** I compute five metrics for every model.

| Metric | Definition | Why included |
|---|---|---|
| Precision@10 | Fraction of top-10 recommendations that appear in the held-out interaction set | Standard recommender quality |
| Recall@10 | Fraction of held-out interactions captured in top-10 | Coverage |
| NDCG@10 | Discounted-cumulative-gain at 10, normalised to ideal ranking | Ranking quality (primary; follows Rendle et al., 2012) |
| Hit Rate@10 | Fraction of users for whom at least one held-out item appears in top-10 | User-level success |
| Outcome-weighted Precision@10 | Precision@10 weighted by the mean assessment score of training-set users who accessed each recommended item | Outcome-awareness — my principal contribution |

The first four are standard and benchmarked against the prior literature. The fifth is my outcome-aware contribution. It operationalises "learning improvement" in the only way I can offline: by measuring whether recommended items are correlated with higher assessment scores in held-out training data.

**Against what.** I compare every model against the Random and Popularity baselines, which establish floor and naive-strong performance, and against each preceding model in the progression. The principal hypothesis test of the project is whether the hybrid beats its components.

**With what statistical confidence.** I evaluate under 5-fold temporal cross-validation. For each fold I split the data chronologically — not randomly — into 80% training and 20% test, advancing the cutoff date across folds. I assess statistical significance of one model over another by paired *t*-test of per-fold metrics with Bonferroni correction across the five metrics. The ablation study disables each component of the hybrid in turn and reports the resulting drop.

The link between this evaluation strategy and the literature I reviewed in [@sec:lit-review] is explicit. Temporal splitting follows Ricci et al. (2015). NDCG as the primary metric follows the implicit-feedback argument of Rendle et al. (2012). Evaluating ranking quality rather than rating prediction is informed by the critique of pure-prediction evaluation that Thai-Nghe et al. (2010) leave unaddressed, and by Wilson et al.'s critique of DKT (2016), which urges caution on any model claiming improvement.

## Ethics, DEI, and Risks

OULAD contains sensitive demographic attributes for every student: gender, age band, region, IMD (Index of Multiple Deprivation) band, and disability flag. I handle these as follows. First, I do not condition recommendations on demographic attributes directly; the model sees only behavioural and content features. Second, I include a *fairness audit* in the evaluation report, breaking per-metric performance down by gender, IMD band, and disability flag, so any group for which the recommender systematically underperforms is visible. This is consistent with the recommendation-system fairness literature and is the way I address the DEI learning objective in concrete, measurable terms. Third, I do not publish or share demographic data beyond aggregate metrics in the report.

I foresee two principal risks going forward. First, the hybrid model might not significantly improve on SVD alone — if so, I will document this as a valid empirical finding rather than a failure of method. Second, sparsity may produce unstable CF metrics for cold-start users, which is the explicit reason I include a content-based component and report cold-start results separately.
