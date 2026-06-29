# Introduction {#sec:intro}

## Project Concept

For my final project under the **CM3005 Data Science**, I am building a hybrid recommendation engine that suggests learning resources to individual learners. The engine combines three signals: how similar learners engaged with content (collaborative filtering), the characteristics of the resources themselves (feature-based content filtering), and the assessment outcomes that followed the engagement. I implement the system as a Python pipeline with a FastAPI service for recommendation queries and a Streamlit dashboard for inspection. I evaluate it offline on the Open University Learning Analytics Dataset (OULAD), which contains 32,593 students, over 10 million Virtual Learning Environment (VLE) interactions, and linked assessment results across seven distance-learning courses.

## Motivation

Online learning has expanded the supply of educational content by orders of magnitude, but learners still have no real way to decide which resource to study next. This is the classical paradox of choice, and in education it has measurable consequences. Most commercial recommendation systems optimise for engagement, which makes sense in retail and media where consumption is the goal. Education is different: I do not want a learner to engage with a resource, I want them to learn from it. A video that holds attention is not the same as a video that closes a knowledge gap.

OULAD makes this gap visible. When I looked at the 23,326 students for whom both VLE logs and assessment results exist, I found a Pearson correlation of r = 0.274 between total clicks and mean assessment score. This is a moderate positive association, consistent with the wider educational-analytics literature, and large enough to be useful as a signal. Students who access certain resource types at certain points in a course perform measurably better on assessments. The opportunity, then, is to recommend for *outcome* rather than engagement: to surface the resources that have historically helped learners with similar profiles improve their scores. This turns the recommendation problem from a popularity-and-similarity question into an outcome-conditioned one, and it is the central contribution I intend to make.

## Aims and Objectives

**Aim.** To design, build, and rigorously evaluate a hybrid educational content recommendation engine that combines collaborative filtering, feature-based content filtering, and learning-outcome signals, and to show that the hybrid outperforms its individual components on OULAD.

**Objectives.**

1. **Data foundation.** Acquire, clean, and characterise OULAD, producing a reproducible preprocessing pipeline that yields a user-item interaction matrix, item-level feature vectors, and a temporal train/test split with no leakage.
2. **Model progression.** Build a graded series of recommenders — Random, Popularity, Collaborative (truncated SVD), Content-Based (cosine similarity over item features), and Hybrid (weighted ensemble) — behind a common interface, so that comparisons across them are like-for-like.
3. **Outcome-aware evaluation.** Score every model with standard ranking metrics (Precision@10, Recall@10, NDCG@10) and an outcome-weighted variant that rewards recommendations correlated with higher assessment scores. Use cross-validation, an ablation study, and paired statistical testing.
4. **Working artefact.** Deliver a FastAPI endpoint and Streamlit dashboard that allow a marker to query the recommender for any student in the test set and inspect both the recommendations and the reasons behind them.

## Scope and Non-Goals

My project is offline and observational. I do not deploy to live learners, run online A/B tests, or perform any intervention study. My outcome signal is therefore a proxy: I measure whether recommendations are *associated with* better outcomes in held-out historical data, not whether issuing those recommendations *causes* better outcomes. I acknowledge this limitation throughout, and the evaluation chapter discusses it explicitly.

I also do not attempt full deep knowledge tracing in the sense of Piech et al. (2015) — that is a sub-project in its own right. Instead, I take the *insight* from knowledge tracing — that prior assessment performance is informative about current need — and use assessment scores as a feature in my hybrid model.

I work on a single dataset. OULAD is rich enough to demonstrate every technique, and adding a second dataset would dilute the depth of evaluation rather than strengthen it. Cross-dataset generalisation is a future extension.

## Contribution

Existing research in educational recommendation tends to specialise. Collaborative methods (Thai-Nghe et al., 2010) ignore content features. Content-based methods (Bousbahi & Chorfi, 2015) ignore collaborative signal. Knowledge-tracing approaches (Corbett & Anderson, 1995; Piech et al., 2015) model knowledge state but stop short of recommending resources. To my knowledge, no published study combines all three — collaborative, content, and outcome — and evaluates the combination on a public dataset with an outcome-aware metric. That is the gap I want to occupy. I am not inventing a new algorithm; my contribution is the systematic, reproducible combination and outcome-aware evaluation of established techniques on a publicly available educational dataset.

## Report Structure

The rest of this report is organised as follows. **[@sec:lit-review]** reviews the most relevant prior work and the technical literature on matrix factorisation and recommender evaluation, and pins down the gap I am addressing. **[@sec:design]** sets out the system design: data, models, evaluation strategy, and the ethical and DEI considerations specific to educational data. **[@sec:prototype]** describes what I have built by the time of this submission — a working data-loading and preprocessing pipeline with two baseline recommenders and a full evaluation framework — together with results, an evaluation of those results, and the planned path from prototype to final system.
