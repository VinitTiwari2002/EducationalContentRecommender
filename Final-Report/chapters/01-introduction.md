# Introduction {#sec:intro}

## Project Concept and Template

For this CM3070 final project, I built a hybrid recommendation engine that suggests learning resources to individual learners, combining three signals — how similar learners engaged with content (collaborative filtering), the resources' own characteristics (content filtering), and the assessment outcomes that followed engagement. I evaluate it offline on the Open University Learning Analytics Dataset (OULAD; Kuzilek, Hlosta & Zdrahal, 2017): 32,593 students, 10.6 M VLE interactions, and linked assessment results across seven distance-learning courses.

The project uses the **CM3005 Data Science** template, which asks for a data-driven investigation of a specific question against a real dataset, delivered as a reproducible pipeline with quantitative evaluation. The *investigation question* is whether outcome-aware hybrid recommendation beats engagement-only baselines on educational data; the *dataset* is OULAD; the *methods* are matrix factorisation, cosine-similarity content filtering, their weighted ensemble, and a LambdaMART reranker; and the *evaluation* is a full offline harness with cross-validation, statistical testing, popularity-bias analysis, and stratified cold-start reporting.

## Motivation

Online learning has expanded content supply by orders of magnitude, but learners still lack a reliable way to decide which resource to study next. In OU's own analyses of OULAD (Kuzilek et al., 2017; Hlosta et al., 2017), roughly a quarter of distance-learning students withdraw before assessment; the modal cause in follow-up surveys is not academic difficulty but disengagement with pacing and content selection. Recommender-system research has largely addressed this problem in retail and media where the goal is consumption. Education is different: I do not want a learner to *engage with* a resource, I want them to *learn from* it — a video that holds attention is not the same as a video that closes a knowledge gap.

OULAD makes that gap directly visible. Across the 23,326 students for whom both VLE logs and assessment results exist, the Pearson correlation between total training-window clicks and mean assessment score is $r = 0.274$ — a moderate positive association, consistent with the wider educational-analytics literature (Hlosta et al., 2017), and large enough to serve as a *signal for recommendation* rather than merely descriptive summary. In evaluations at the Open University (Kuzilek et al., 2017; Rienties et al., 2016), navigational and study-path guidance are repeatedly named as high-value VLE features for distance learners. The opportunity is to recommend for outcome rather than engagement — to surface resources that have historically helped similar learners improve their scores — and that stated learner need is what my central contribution addresses.

## Aims and Objectives

**Aim.** To design, build, and evaluate a hybrid educational-content recommender combining collaborative filtering, content filtering, and outcome signals, and to show that the hybrid outperforms its individual components on OULAD under statistically-honest offline evaluation.

**Objectives.**

1. **Reproducible data foundation** — OULAD ingestion, sparse user-item matrix, train-only item features, and a temporal train/test split with course-scoping and optional time-decay (measured by §5.10 sensitivity sweeps).
2. **Model progression** — Random, Popularity, SVD, ALS, Content, Hybrid, GatedHybrid, and a LambdaMART reranker behind a common `recommend(user, k, candidates)` interface (§5.3, §5.4, §5.11).
3. **Outcome-aware evaluation** — Precision/Recall/NDCG/Hit-Rate@K plus a novel outcome-weighted Precision@K, with 95% bootstrap CIs, paired-*t* with Bonferroni adjustment, catalogue coverage + Gini, cold-start stratification, and a demographic fairness audit (§5.2, §5.5–§5.9).
4. **Service artefact** — three FastAPI endpoints (`/health`, `/recommend`, `/decompose`), a three-page Streamlit transparency dashboard, and a two-stage Docker image, all persisting the fitted recommenders for O(1) request-time cost (§4.9).

## Scope and Non-Goals

The project is offline and observational. I do not deploy to live learners, run online A/B tests, or perform any intervention study; the outcome signal is a proxy for *association*, not causation, and the evaluation chapter treats that distinction as its own methodological question. I do not attempt deep knowledge tracing (Piech et al., 2015); instead I take its insight — that prior assessment performance is informative about current need — and use assessment scores as a feature. I work on a single dataset: OULAD is rich enough to demonstrate every technique, and a second dataset would dilute evaluation depth rather than strengthen it.

## Contribution

Existing educational-recommendation research tends to specialise: collaborative methods (Thai-Nghe et al., 2010) ignore content features, content-based methods (Bousbahi & Chorfi, 2015) ignore collaborative signal, and knowledge-tracing approaches (Corbett & Anderson, 1995; Piech et al., 2015) model knowledge state but stop short of recommending resources. A systematic search across the ACM Digital Library, IEEE Xplore, and Google Scholar ([@sec:lit-review]) surfaces pairwise combinations but no study I am aware of that combines all three signals and evaluates against an outcome-aware metric on a public educational dataset.

Five specific contributions are load-bearing:

1. **Outcome-weighted Precision@K** — a novel offline metric precisely defined (§5.2), reducing to standard Precision@K under a uniform-outcome null, computed strictly from training data.
2. **A per-user switching (gated) hybrid** grounded in a cold-start ablation — cold users route to Popularity, warm users to the tuned Hybrid (§4.5, §5.7).
3. **A two-stage LambdaMART reranker with an honest three-window training design** — Stage 1 refit on `tune_train`, labels from `tune_val`, evaluation on the primary test window; pool-normalised features let the booster transfer to a full-train Stage 1 at inference (§4.6, §5.11).
4. **A full serving stack** — FastAPI service, Streamlit transparency dashboard, and a two-stage Docker image, letting a marker query any student and inspect the per-item score decomposition (§4.9).
5. **A reproducible statistical harness** — five-fold temporal CV, bootstrap CIs, paired-*t* with Bonferroni adjustment, ablation, cold-start, fairness audit, and a three-seed robustness sweep.

## Report Structure

[@sec:lit-review] surveys prior work and the gap. [@sec:design] sets out data, models, evaluation strategy, and service layer. [@sec:implementation] describes the algorithms, code, and delivered service artefacts. [@sec:evaluation] presents metrics with CIs, cross-validation, ablation, cold-start stratification, fairness audit, hyperparameter sensitivity, and the LambdaMART reranker experiment. [@sec:conclusion] summarises contributions, limitations, and future work.
