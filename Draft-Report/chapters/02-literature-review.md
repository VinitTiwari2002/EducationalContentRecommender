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

I searched three databases — the ACM Digital Library, IEEE Xplore, and Google Scholar — for papers published between 2005 and 2025 using the query `(recommender OR recommendation) AND (education OR MOOC OR "learning analytics" OR "e-learning") AND (hybrid OR collaborative OR "content-based")`. The initial pool of 342 unique candidate titles was screened by title and abstract for three inclusion criteria: (i) recommendation of educational resources as the primary artefact (not performance prediction alone); (ii) evaluation on a real dataset; and (iii) English-language full text available. Screening yielded 61 papers eligible for full-text review; deeper reading against the three research questions above and a fourth criterion — that outcomes were used either in the model or in the evaluation — narrowed this to eight primary works. I supplement these with three surveys of technology-enhanced-learning (TEL) recommender systems (Manouselis et al., 2011; Drachsler et al., 2015; Zawacki-Richter et al., 2019) that collectively cover ~200 further systems and provide landscape support for the gap claim without requiring me to re-review each individually.

## Predicting Outcomes vs. Correlating Resources with Outcomes

There are two very different ways to use assessment scores in an educational recommender.

**Predicting outcomes.** The recommender's *target* is the assessment score itself. Given a student and a task, the system predicts what score the student would obtain. This is what Thai-Nghe et al. (2010) do with matrix factorisation, and what Piech et al. (2015) do with DKT. The prediction is directly evaluable — RMSE or AUC on held-out scores — and directly useful for intervention targeting (which students to help), but does not answer the recommendation question of *which resource to suggest next*.

**Correlating resources with outcomes.** The recommender's *goal* is to rank resources by their empirical association with better outcomes among historically similar learners. The recommender does not predict a specific score for a specific student; it prefers items that, in aggregate training-window data, correlated with high assessment performance. This is what my outcome-weighted precision metric captures. Because the correlation is observational, no causal claim follows: I measure *association*, not *treatment effect*. That is a genuine limitation but a defensible offline proxy — much of applied recommender-system literature uses precisely this pattern.

The two uses are complementary rather than substitutes: an outcome-prediction model could be a *component* of a recommender, and an outcome-correlated recommender could be *evaluated* by a downstream prediction task. But conflating them — as some prior surveys do — obscures both what any system is claiming and how it should be tested. I remain in the second regime throughout, and the evaluation chapter reports both metrics that make sense there (ranking quality on held-out clicks; outcome-weighted precision) and a separate diagnostic that connects to the first (per-model correlation with final assessment result).

## Collaborative Filtering in Education: Thai-Nghe et al. (2010)

Thai-Nghe et al. (2010) were among the first to apply matrix factorisation to educational data. They reframe student performance prediction as a recommender-system problem: students are users, tasks are items, and the score is the implicit rating. Matrix factorisation beat traditional regression on three educational datasets, including KDD Cup 2010 Algebra and Bridge to Algebra. The contribution matters because it shows the algebra of collaborative filtering is not domain-specific: latent-factor models can capture student and task abilities without the textual or behavioural metadata used in commercial systems.

The limitations bear on my project directly. First, the system *predicts* — how a student would score on a known task — but does not *recommend* unseen resources to study next. Prediction tells the learner where they stand; recommendation tells them where to go. Second, the model treats every task as an opaque identifier, with no notion of difficulty, topic, or content, and so cannot handle new tasks. Third, the evaluation is RMSE on held-out scores; whether following the predictions would have *improved* learning is not addressed. For me, Thai-Nghe et al. establish that matrix factorisation works in education — which is why I include SVD and ALS in the model progression — but also show why a pure collaborative approach is not enough.

## Modelling Knowledge State: Corbett & Anderson (1995); Piech et al. (2015)

A parallel strand of research treats education not as recommendation but as knowledge-state estimation. Corbett & Anderson's (1995) Bayesian Knowledge Tracing (BKT) models student knowledge as a hidden state, updated after each observed practice item according to four parameters: prior knowledge, transition probability, slip and guess. The model is principled, interpretable, and the empirical workhorse of the Cognitive Tutor family of intelligent tutoring systems.

Piech et al. (2015) introduced Deep Knowledge Tracing (DKT), replacing the BKT Markov model with a recurrent neural network (LSTM) that learns its own latent knowledge representation. DKT beat BKT on standard benchmarks, largely because it sheds the assumption that skills are independent — an LSTM can capture cross-skill transfer effects that BKT cannot.

Two things from this strand are useful. First, prior assessment performance is genuinely informative about learner state — which justifies using assessment scores as a feature in the hybrid. Second, and more importantly, both papers stop at *knowledge estimation*; neither says what to recommend. BKT, deployed in tutoring systems, typically uses hand-authored progression rules over the estimated state; DKT is purely evaluative. Both are also *content-blind*: they model student-skill interaction without any representation of the resource itself, so they cannot distinguish "watch the video" from "attempt the quiz" even when the two interventions have very different effects. Wilson et al. (2016) showed that simpler models match DKT's claimed performance under fairer evaluation protocols — a caution that any deep-learning component must be evaluated against strong baselines rather than in isolation.

## Content-Based Recommendation for MOOCs: Bousbahi & Chorfi (2015)

Bousbahi & Chorfi (2015) take the opposite stance to Thai-Nghe et al. Their MOOC-Rec system represents each course by a vector of metadata features (topic, level, language, prerequisites, duration) and recommends courses whose feature vector is closest to a learner's stated profile. The approach is case-based: a new query is matched to the closest historical case, and the recommendation is explainable because the matching dimensions can be surfaced to the learner.

The strengths complement Thai-Nghe et al. directly. Content-based methods do not need historical interaction data and so handle cold-start gracefully, and they are auditable — the recommender can explain *why* it suggested a resource by naming the features that contributed most. Both properties matter in education, where unexplained algorithmic recommendations erode learner trust and where new courses and new learners arrive frequently.

The limitations are also real. There is no collaborative component, so the system cannot learn from what worked for similar learners: if learners with profile *P* historically did better with resource *A* than resource *B*, MOOC-Rec has no way to discover this. Evaluation is IR-style metrics on an expert-curated test set; there is no measurement of whether recommendations led to better learning. And the feature representation is essentially textual metadata; behavioural and outcome-derived features are absent. For my project, Bousbahi & Chorfi justify the content-based component — particularly for cold-start — but also show why content-based alone is not enough.

## Surveys of TEL Recommender Systems

Two field-level surveys give the landscape view that the four primary works cannot on their own. Manouselis et al. (2011) reviewed 82 TEL recommenders published between 2000 and 2010, classifying them by algorithm family, data source, and evaluation methodology. Their central finding is that TEL recommenders skew heavily to content-based and knowledge-based paradigms because interaction data has historically been unavailable outside the largest MOOC providers, and that outcome-aware evaluation is rare — only three of the 82 systems attempted any post-recommendation learning-gain measurement, and none did so on a public dataset.

Drachsler et al. (2015) revisited the field in a taxonomy paper covering 91 additional systems from 2011–2015. Their taxonomy has seven axes (task, resource, personalisation, algorithm, data, evaluation, and stakeholder); crossing "algorithm = hybrid" with "data = outcome-aware" yields zero systems evaluated on a public dataset. Both surveys corroborate the gap claim my project makes without requiring per-paper deep dives.

Zawacki-Richter et al. (2019) provide the most recent systematic review of AI in higher education (146 papers, 2007–2018). Only 6% of the reviewed systems perform any form of *personalised content recommendation*, and of those, none combine collaborative, content, and outcome signals under an outcome-aware evaluation protocol on public data. The gap I identify is therefore not merely a lack of one specific study but a systematic under-exploration of the intersection.

## Technical Foundations

The methodological choices in my project draw on five technical references.

**Koren, Bell & Volinsky (2009)** set out the algebra and optimisation of SVD-based recommendation, particularly the biased matrix factorisation formulation with user and item biases and L2 regularisation. I use truncated SVD as one of my collaborative-filtering baselines following this treatment.

**Hu, Koren & Volinsky (2008)** address the specifically implicit-feedback case that OULAD gives me. Absence of a click is not the same as active dislike, so treating the zero cells of the interaction matrix as negatives is the well-known trap of implicit-feedback CF. Their alternating-least-squares (ALS) formulation with confidence weighting — c(u, i) = 1 + α × clicks(u, i) — is the canonical mitigation and the reference implementation for the `implicit` library I use.

**Rendle et al. (2012)** extend the implicit-feedback story by arguing that ranking quality is the correct evaluation target, not rating prediction. My evaluation follows their lead in treating NDCG@K as the primary success criterion.

**Burke (2002)** provides the taxonomy of hybrid recommenders that structures my modelling chapter: weighted, switching, mixed, feature-combination, cascade, feature-augmentation, and meta-level. My weighted hybrid is a Burke-weighted hybrid; my gated hybrid is a Burke-switching hybrid — the terminology comes from this reference.

**Ricci, Rokach & Shapira (2015)** cover evaluation methodology in depth. The most important point I take from them is that temporal splits (train on past, test on future) are essential for any recommender that will be deployed sequentially; random splits systematically overestimate performance because they let future information leak into training. Their treatment of confidence intervals and paired testing across folds underpins my evaluation strategy in [@sec:design].

## Critical Synthesis and Identified Gap

Reading across the primary works, four themes cut through the literature and are more informative than any single-paper description. **Theme 1 — the role of outcomes.** All the primary works acknowledge assessment outcomes but position them very differently: Thai-Nghe et al. and Piech et al. treat outcomes as the *target* of a supervised prediction; Corbett & Anderson use outcomes as *observations* of a latent knowledge state; Bousbahi & Chorfi ignore them entirely. No primary work uses outcomes as a *quality signal for recommendation evaluation*, which is the position I take. **Theme 2 — content vs. behavioural signal.** The CF/KT strand ignores content features; the content-based strand ignores behavioural signal. Both strands report this as a limitation of the other, yet no primary work combines them with a *third* outcome signal under one evaluation harness. **Theme 3 — evaluation methodology.** RMSE (Thai-Nghe et al.), AUC on next-answer prediction (Piech et al.), and IR metrics on an expert-curated test set (Bousbahi & Chorfi) are three incompatible evaluations for what is nominally the same task. Rendle et al. and Ricci et al. argue that ranking metrics on a temporal split are the right choice for implicit-feedback educational data; only the surveys (Manouselis et al.; Drachsler et al.) confirm how rarely this is done in practice. **Theme 4 — cold-start and sparsity.** The CF/KT strand degrades under sparsity but does not report cold-start separately; the content-based strand is robust to cold-start by construction but performs worse on warm users than CF. Only the surveys explicitly flag this trade-off — and none of the primary works stratify their evaluation by user warmth. My design addresses each theme directly: a hybrid combining all three signals (themes 1–2), a ranking-metric temporal evaluation (theme 3), and a per-user gated recommender + cold-start stratification (theme 4). Mapping the primary works against these dimensions:

| Work | Approach | Content | Outcomes | Evaluation |
|------|----------|:---:|----------|------------|
| Thai-Nghe et al. (2010) | Collaborative (MF) | — | as target | RMSE |
| Corbett & Anderson (1995) | Knowledge tracing (HMM) | — | as target | Knowledge estimation |
| Piech et al. (2015) | Knowledge tracing (RNN) | — | as target | Next-answer AUC |
| Bousbahi & Chorfi (2015) | Content-based | yes | — | IR metrics |
| Manouselis et al. (2011) | Survey (82 systems) | mixed | rarely | mixed |
| Drachsler et al. (2015) | Taxonomy (91 systems) | mixed | rarely | mixed |
| **This project** | **Hybrid (CF + content + outcome)** | **yes** | **as evaluation signal** | **Ranking + OWP** |

Two specific gaps come out of this. The first is *architectural*: no published work combines collaborative filtering, content features, and an outcome signal in a single recommender evaluated on a public educational dataset. Each strand of research has stayed inside its own paradigm, and the two TEL surveys corroborate this at scale. The second is *methodological*: where outcomes are used at all, they are used as the *target* of prediction (Thai-Nghe et al.; Piech et al.), not as a *quality signal for recommendation evaluation*. That distinction — sharpened in Section 2.3 — matters because predicting that a learner will fail an assessment is not the same as recommending content that will reduce the probability of failure; only the second is actionable for the learner.

One further methodological concern, taken from Wilson et al.'s critique of DKT and from Ricci et al.'s argument for temporal splits: any system claiming improvement over baselines has to be evaluated against carefully chosen baselines under a protocol that does not allow leakage. I address this directly with a graded model progression (Random $\rightarrow$ Popularity $\rightarrow$ SVD $\rightarrow$ ALS $\rightarrow$ Content $\rightarrow$ Hybrid $\rightarrow$ GatedHybrid) under a temporal split, five-fold temporal cross-validation, paired *t*-testing with Bonferroni adjustment across metrics, and an ablation study to isolate the contribution of each hybrid component.

I therefore position my contribution not as a new algorithm but as a *systematic combination* — collaborative filtering, feature-based content matching, and outcome-aware evaluation — implemented and evaluated end-to-end on a public dataset, with enough methodological rigour to allow direct comparison against the prior work surveyed here.

*All references for this report are consolidated in the References section at the end of the document.*
