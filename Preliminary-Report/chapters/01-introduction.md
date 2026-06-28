# Introduction {#sec:intro}

## Project Concept

This project, undertaken under the **CM3005 Data Science**, develops a hybrid recommendation engine that suggests learning resources to individual learners based on three complementary signals: how similar learners have engaged with content (collaborative filtering), the structural and pedagogical characteristics of the resources themselves (feature-based content filtering), and the assessment outcomes that follow that engagement. The system is implemented as a Python pipeline with a FastAPI service exposing recommendation endpoints and a Streamlit dashboard for inspection. It is evaluated offline on the Open University Learning Analytics Dataset (OULAD), which contains 32,593 students, over 10 million Virtual Learning Environment (VLE) interactions, and linked assessment results across seven distance-learning courses.

## Motivation

Online learning has expanded the supply of educational content by orders of magnitude, but learners have no corresponding expansion in their ability to evaluate which resource to study next. This is the classical paradox of choice, and it has measurable consequences for learning. The dominant approach in commercial recommendation — popularity-weighted or engagement-optimised ranking — is well suited to retail and media domains where the implicit goal is to maximise consumption. Education has a different objective function: the goal is not that the learner engages with a resource but that they *learn* from it, and the two are not the same. A video that holds attention is not necessarily a video that closes a knowledge gap.

The OULAD dataset makes this gap visible. An exploratory analysis of the 23,326 students for whom both VLE interaction logs and assessment results exist yields a Pearson correlation of r = 0.274 between total click volume and mean assessment score — a moderate positive association consistent with the broader educational-analytics literature and large enough to be useful as a signal in a recommender. Students who access certain resource types at particular points in a course show measurably higher assessment scores than students who do not. The opportunity, therefore, is to recommend not for engagement but for *outcome*: to surface the resources that have historically helped learners with similar profiles improve their assessment performance. This shifts the recommendation problem from a popularity-and-similarity question to an outcome-conditioned one, and it is the central contribution this project aims to make.

## Aims and Objectives

The project has one principal aim and four supporting objectives.

**Aim.** To design, implement, and rigorously evaluate a hybrid educational content recommendation engine that combines collaborative filtering, feature-based content filtering, and learning-outcome signals, and to demonstrate that the hybrid outperforms its individual components on the OULAD dataset.

**Objectives.**

1. **Data foundation.** Acquire, clean, and characterise the OULAD dataset, producing a reproducible preprocessing pipeline that yields a user-item interaction matrix, item-level content feature vectors, and a temporal train/test split that prevents data leakage.
2. **Model progression.** Implement a graded series of recommenders — Random, Popularity, Collaborative (truncated SVD), Content-Based (cosine similarity over item features), and Hybrid (weighted ensemble) — under a common interface so that direct comparison is fair.
3. **Outcome-aware evaluation.** Evaluate every model using both standard ranking metrics (Precision@10, Recall@10, NDCG@10) and an outcome-weighted variant that rewards recommendations correlating with higher assessment scores, with cross-validation, ablation, and paired statistical testing.
4. **Working artefact.** Deliver a FastAPI endpoint and Streamlit dashboard that allow a marker to query the recommender for any student in the test set and inspect both the recommendations and their justifications.

## Scope and Non-Goals

The project is offline and observational. It does *not* deploy to live learners, run online A/B tests, or perform any intervention study. The outcome signal is therefore a proxy: the project measures whether recommendations are *associated with* better outcomes in held-out historical data, not whether issuing those recommendations *causes* better outcomes. This limitation is acknowledged throughout and the evaluation chapter discusses it explicitly.

The project also does not attempt full deep knowledge tracing in the sense of Piech et al. (2015), which would require an entire neural-modelling sub-project of its own. Instead, the project incorporates the *insight* from knowledge tracing — that prior assessment performance is informative about current need — by using assessment scores as a feature in the hybrid model, rather than as the model itself.

Finally, the project is single-dataset. OULAD is rich enough to demonstrate every technique, and using a second dataset would dilute the depth of evaluation rather than strengthen it. Cross-dataset generalisation can be tracked as future extension.

## Contribution

Existing research in educational recommendation has tended to specialise: collaborative methods (Thai-Nghe et al., 2010) ignore content features; content-based methods (Bousbahi & Chorfi, 2015) ignore collaborative signal; knowledge-tracing approaches (Corbett & Anderson, 1995; Piech et al., 2015) model knowledge state but stop short of resource-level recommendation. No public study combines all three signals — collaborative, content, and outcome — and evaluates the combination on a public dataset with an outcome-aware metric. This project occupies that gap. The contribution is not the invention of new algorithms but the systematic, reproducible combination and outcome-aware evaluation of established techniques on a publicly available educational dataset.

## Report Structure

The remainder of this report is organised as follows. **[@sec:lit-review]** reviews the four most relevant prior works and the underlying technical literature on matrix factorisation and recommender evaluation, identifying the gap that the project addresses. **[@sec:design]** presents the system design: data design, model design, evaluation strategy, work plan, and the ethical and DEI considerations specific to educational data. **[@sec:prototype]** describes the feature prototype implemented at the time of submission — a working data-loading and preprocessing pipeline with two baseline recommenders (Random and Popularity) and the full evaluation framework — together with results, an evaluation of those results, and the planned path from prototype to final system.
