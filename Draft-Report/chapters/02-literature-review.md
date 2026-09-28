# Literature Review {#sec:lit-review}

## Scope and Research Questions

I focus this review on the intersection of recommender systems and educational technology, and on four issues my project must address: whether collaborative filtering techniques developed for commercial domains carry over to education, how content features can substitute for or complement collaborative signal when interaction data is sparse, how prior work has used learning outcomes as a recommendation signal, and what evaluation methodology fits an offline educational recommender on a public dataset. Three research questions structure the review:

> **1.** What modelling approaches have been shown to work for recommendation in educational settings, and what are their limitations?
>
> **2.** How have prior systems incorporated — or failed to incorporate — learning outcomes, either in the model or its evaluation?
>
> **3.** What evaluation methodology is appropriate for an offline educational recommender on a public dataset, and which metrics are defensible?

Section 2.2 documents the systematic search that produced the corpus; Section 2.3 sharpens the methodological distinction between *predicting* outcomes and *correlating resources with* outcomes, which is central to my contribution; Sections 2.4–2.7 review the primary works; Section 2.8 covers the technical-foundations references; Section 2.9 synthesises the gap.

## Search Methodology

I searched the ACM Digital Library, IEEE Xplore, and Google Scholar for 2005–2025 papers with the query `(recommender OR recommendation) AND (education OR MOOC OR "learning analytics" OR "e-learning") AND (hybrid OR collaborative OR "content-based")`. Screening 342 candidate titles against three inclusion criteria (recommendation of educational resources as the primary artefact; evaluation on a real dataset; English-language full text available) yielded 61 full-text-eligible papers; a fourth criterion (outcomes used in either model or evaluation) narrowed this to eight primary works. Three TEL surveys (Manouselis et al., 2011; Drachsler et al., 2015; Zawacki-Richter et al., 2019) supplement these with landscape coverage of ~200 further systems. The gap-in-literature claim rests on all three sources: primary works, systematic search, and survey corroboration.

## Predicting Outcomes vs. Correlating Resources with Outcomes

There are two very different ways to use assessment scores in an educational recommender.

**Predicting outcomes.** The recommender's *target* is the assessment score itself. Given a student and a task, the system predicts what score the student would obtain. This is what Thai-Nghe et al. (2010) do with matrix factorisation, and what Piech et al. (2015) do with DKT. The prediction is directly evaluable — RMSE or AUC on held-out scores — and directly useful for intervention targeting (which students to help), but does not answer the recommendation question of *which resource to suggest next*.

**Correlating resources with outcomes.** The recommender's *goal* is to rank resources by their empirical association with better outcomes among historically similar learners. The recommender does not predict a specific score for a specific student; it prefers items that, in aggregate training-window data, correlated with high assessment performance. This is what my outcome-weighted precision metric captures. Because the correlation is observational, no causal claim follows: I measure *association*, not *treatment effect*. That is a genuine limitation but a defensible offline proxy — much of applied recommender-system literature uses precisely this pattern.

The two uses are complementary rather than substitutes: an outcome-prediction model could be a *component* of a recommender, and an outcome-correlated recommender could be *evaluated* by a downstream prediction task. But conflating them — as some prior surveys do — obscures both what any system is claiming and how it should be tested. I remain in the second regime throughout, and the evaluation chapter reports both metrics that make sense there (ranking quality on held-out clicks; outcome-weighted precision) and a separate diagnostic that connects to the first (per-model correlation with final assessment result).

## Collaborative Filtering in Education: Thai-Nghe et al. (2010)

Thai-Nghe et al. (2010) apply matrix factorisation to educational data by reframing student performance prediction as a recommender-system problem — students are users, tasks are items, scores are implicit ratings. MF beats regression on three datasets (KDD Cup 2010 Algebra and Bridge to Algebra among them). The lesson is that CF's algebra transfers to education: latent-factor models capture student and task abilities without textual metadata.

The limitations bear on my project directly. The system *predicts* held-out scores — it does not *recommend* unseen resources. Every task is an opaque identifier (no content), so new tasks are unhandled. Evaluation is RMSE, not learning improvement. Thai-Nghe et al. therefore justify including SVD and ALS in my progression while also showing why pure CF is not enough.

## Modelling Knowledge State: Corbett & Anderson (1995); Piech et al. (2015)

A parallel strand treats education as knowledge-state estimation rather than recommendation. Corbett & Anderson's (1995) BKT models learner knowledge as a hidden state (prior, transition, slip, guess) and underpins the Cognitive Tutor family. Piech et al. (2015) introduced Deep Knowledge Tracing (DKT), replacing the Markov model with an LSTM that learns its own latent state and beats BKT on standard benchmarks by capturing cross-skill transfer.

Two takeaways. First, prior assessment performance is informative about learner state — justifying assessment scores as a feature. Second, both papers stop at knowledge *estimation*; neither says what to recommend, and both are content-blind (no representation of the resource itself). Wilson et al. (2016) further show simpler models match DKT under fairer protocols — a caution against evaluating deep-learning components in isolation.

## Content-Based Recommendation for MOOCs: Bousbahi & Chorfi (2015)

Bousbahi & Chorfi (2015) take the opposite stance to Thai-Nghe et al.: their MOOC-Rec represents each course as a metadata vector (topic, level, language, prerequisites, duration) and recommends nearest-neighbour matches to a learner's stated profile. The approach is case-based and auditable — the matching dimensions can be surfaced to the learner. Both properties matter in education, where unexplained recommendations erode trust and new courses/learners arrive frequently.

The limitations are real: no collaborative component (cannot learn what worked for similar learners); evaluation is IR-style on an expert-curated test set (no learning-gain measurement); features are textual metadata only (no behavioural or outcome signals). Bousbahi & Chorfi therefore justify my content-based component — particularly for cold-start — while showing why it is not enough alone.

## Surveys of TEL Recommender Systems

Two field-level surveys give the landscape view that the four primary works cannot on their own. Manouselis et al. (2011) reviewed 82 TEL recommenders published between 2000 and 2010, classifying them by algorithm family, data source, and evaluation methodology. Their central finding is that TEL recommenders skew heavily to content-based and knowledge-based paradigms because interaction data has historically been unavailable outside the largest MOOC providers, and that outcome-aware evaluation is rare — only three of the 82 systems attempted any post-recommendation learning-gain measurement, and none did so on a public dataset.

Drachsler et al. (2015) revisited the field in a taxonomy paper covering 91 additional systems from 2011–2015. Their taxonomy has seven axes (task, resource, personalisation, algorithm, data, evaluation, and stakeholder); crossing "algorithm = hybrid" with "data = outcome-aware" yields zero systems evaluated on a public dataset. Both surveys corroborate the gap claim my project makes without requiring per-paper deep dives.

Zawacki-Richter et al. (2019) provide the most recent systematic review of AI in higher education (146 papers, 2007–2018). Only 6% of reviewed systems perform any form of *personalised content recommendation*, and of those, none combine collaborative, content, and outcome signals under an outcome-aware evaluation protocol on public data. Rienties et al. (2016) directly report the learner-need evidence at the Open University: their Analytics4Action evaluation across OU distance-learning cohorts finds that learners repeatedly request help navigating the VLE and choosing what to study next, and that early-warning + guidance interventions correlate with improved completion — direct empirical support for the domain-user justification. The gap is therefore both *architectural* (no comparable published combination on public data across these surveys) and *user-motivated* (a stated learner need).

## Technical Foundations

The methodological choices in my project draw on five technical references.

**Koren, Bell & Volinsky (2009)** set out the algebra and optimisation of SVD-based recommendation, particularly the biased matrix factorisation formulation with user and item biases and L2 regularisation. I use truncated SVD as one of my collaborative-filtering baselines following this treatment.

**Hu, Koren & Volinsky (2008)** address the specifically implicit-feedback case that OULAD gives me. Absence of a click is not the same as active dislike, so treating the zero cells of the interaction matrix as negatives is the well-known trap of implicit-feedback CF. Their alternating-least-squares (ALS) formulation with confidence weighting — c(u, i) = 1 + α × clicks(u, i) — is the canonical mitigation and the reference implementation for the `implicit` library I use.

**Rendle et al. (2012)** extend the implicit-feedback story by arguing that ranking quality is the correct evaluation target, not rating prediction. My evaluation follows their lead in treating NDCG@K as the primary success criterion.

**Burke (2002)** provides the taxonomy of hybrid recommenders that structures my modelling chapter: weighted, switching, mixed, feature-combination, cascade, feature-augmentation, and meta-level. My weighted hybrid is a Burke-weighted hybrid; my gated hybrid is a Burke-switching hybrid — the terminology comes from this reference.

**Ricci, Rokach & Shapira (2015)** cover evaluation methodology in depth. The most important point I take from them is that temporal splits (train on past, test on future) are essential for any recommender that will be deployed sequentially; random splits systematically overestimate performance because they let future information leak into training. Their treatment of confidence intervals and paired testing across folds underpins my evaluation strategy in [@sec:design].

**Burges et al. (2010)** provide the LambdaMART formulation of learning-to-rank via gradient-boosted decision trees. The pairwise-preference framing lets a ranker directly optimise a listwise metric like NDCG rather than a pointwise loss, and is the standard second-stage industrial reranker (Bing, YouTube, Amazon). I use LightGBM's `LGBMRanker` with `objective='lambdarank'` for the two-stage reranker (§4.6).

**Ekstrand et al. (2018)** and **Zehlike et al. (2022)** ground the fairness-in-recommendation critique of §5.9 and the fairness-aware re-ranking layer earmarked in the conclusion. Ekstrand et al. show that recommenders can systematically underperform on demographic subgroups (their MovieLens results parallel the OULAD gender gap I report); Zehlike et al. survey mitigation strategies (post-hoc re-ranking with exposure or parity penalties, in-training constraints), which is the specific route I propose for future work.

## Critical Synthesis and Identified Gap

Four themes cut through the primary works. **Outcomes as target vs. signal:** Thai-Nghe et al. and Piech et al. treat outcomes as a *prediction target*; Corbett & Anderson use them as *observations of latent state*; Bousbahi & Chorfi ignore them. No primary work uses outcomes as a *quality signal for recommendation evaluation*, which is my position. **Content vs. behavioural signal:** the CF/KT strand ignores content features; the content-based strand ignores behavioural signal; no primary work combines them with a third outcome signal. **Evaluation methodology:** RMSE, next-answer AUC, and IR metrics on curated test sets are three incompatible evaluations for nominally the same task; Rendle et al. and Ricci et al. argue for temporal-split ranking metrics, but the surveys confirm how rarely this is done. **Cold-start and sparsity:** the CF/KT strand degrades under sparsity without stratified reporting; the content-based strand handles cold-start by construction but underperforms on warm users. None of the primary works stratify by user warmth. My design addresses each theme: a hybrid combining all three signals, a ranking-metric temporal evaluation, and a per-user gated recommender with explicit cold-start stratification.

| Work | Approach | Content | Outcomes | Evaluation |
|------|----------|:---:|----------|------------|
| Thai-Nghe et al. (2010) | Collaborative (MF) | — | as target | RMSE |
| Corbett & Anderson (1995) | Knowledge tracing (HMM) | — | as target | Knowledge estimation |
| Piech et al. (2015) | Knowledge tracing (RNN) | — | as target | Next-answer AUC |
| Bousbahi & Chorfi (2015) | Content-based | yes | — | IR metrics |
| Manouselis et al. (2011) | Survey (82 systems) | mixed | rarely | mixed |
| Drachsler et al. (2015) | Taxonomy (91 systems) | mixed | rarely | mixed |
| **This project** | **Hybrid (CF + content + outcome)** | **yes** | **as evaluation signal** | **Ranking + OWP** |

Two specific gaps follow. **Architectural:** I am not aware of a published system that combines CF + content + outcome signals in a single recommender evaluated on a public educational dataset; the three surveys (Manouselis, Drachsler, Zawacki-Richter) corroborate this at scale. **Methodological:** where outcomes are used at all they are the prediction *target*, not a quality signal for evaluation; predicting failure is not the same as recommending content to reduce failure — only the latter is actionable for the learner.

Wilson et al.'s critique of DKT and Ricci et al.'s temporal-split argument close the loop: any system claiming improvement has to be evaluated against carefully chosen baselines under a leak-free protocol. I address this with a graded eight-model progression under a temporal split, five-fold temporal CV, paired-*t* with Bonferroni adjustment, an ablation study, and a three-seed robustness sweep.

*All references for this report are consolidated in the References section at the end of the document.*
