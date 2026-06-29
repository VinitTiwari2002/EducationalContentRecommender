# Literature Review {#sec:lit-review}

## Scope and Research Questions

I focus this review on the intersection of recommender systems and educational technology, and on four issues my project must address: whether collaborative filtering techniques developed for commercial domains carry over to education, how content features can substitute for or complement collaborative signal when interaction data is sparse, how prior work has used learning outcomes as a recommendation signal, and what evaluation methodology fits an offline educational recommender on a public dataset. Three research questions structure the review:

> **1.** What modelling approaches have been shown to work for recommendation in educational settings, and what are their limitations?
>
> **2.** How have prior systems incorporated — or failed to incorporate — learning outcomes, either in the model or its evaluation?
>
> **3.** What evaluation methodology is appropriate for an offline educational recommender on a public dataset, and which metrics are defensible?

I examine four primary works in detail, each addressing one or more of these questions, and then draw methodological inferences from three supporting technical references on matrix factorisation, evaluation, and ranking. The synthesis at the end identifies the gap that motivates my project.

## Collaborative Filtering in Education: Thai-Nghe et al. (2010)

Thai-Nghe et al. (2010) were among the first to apply matrix factorisation to educational data. Their study reframes student performance prediction as a recommender-system problem: students are the users, tasks are the items, and the score a student achieves on a task is the implicit rating. They show that matrix factorisation beats traditional regression on three educational datasets, including KDD Cup 2010 Algebra and Bridge to Algebra.

The contribution matters because it shows that the algebra of collaborative filtering is not domain-specific. Latent-factor models can capture student and task abilities without the textual or behavioural metadata used in commercial systems.

The limitations bear directly on my project. First, the system *predicts* — it estimates how a student would score on a known task — but it does not *recommend* unseen resources for the student to study next. Prediction tells the learner where they stand; recommendation tells them where to go. Second, the model treats every task as an opaque identifier, with no notion of difficulty, topic, or content. The result is a purely collaborative model that suffers the classical cold-start problem for new tasks and new students. Third, the evaluation is RMSE on held-out scores; whether following the predictions would have *improved* the student's learning outcome is not addressed and, given the data, could not have been. For my project, Thai-Nghe et al. establish that matrix factorisation works in education — which is why I include an SVD model in my progression — but they also show why a pure collaborative approach is not enough.

## Modelling Knowledge State: Corbett & Anderson (1995); Piech et al. (2015)

A parallel strand of research treats education not as a recommendation problem at all, but as a knowledge-state estimation problem. The seminal contribution is Corbett & Anderson's (1995) Bayesian Knowledge Tracing (BKT). BKT models student knowledge as a hidden state, updated after each observed practice item according to four parameters: the prior probability of knowing a skill, the probability of moving from "not known" to "known" given a learning opportunity, and the slip and guess probabilities relating knowledge state to observed performance. The model is principled, interpretable, and the empirical workhorse of the Cognitive Tutor family of intelligent tutoring systems.

Piech et al. (2015) introduced Deep Knowledge Tracing (DKT). DKT replaces the BKT Markov model with a recurrent neural network (LSTM) that learns its own latent knowledge representation from raw sequences of student responses. DKT beats BKT on standard benchmarks, largely because it sheds the assumption that skills are independent: an LSTM can capture cross-skill transfer effects that BKT cannot.

Two things from this strand are useful to me. First, prior assessment performance is genuinely informative about learner state and should not be discarded — which justifies my use of assessment scores as a feature in the hybrid model. Second, and more importantly, both papers stop at *knowledge estimation*; neither says what to recommend next. BKT, deployed in tutoring systems, typically uses hand-authored progression rules over the estimated state; DKT is purely evaluative. Both are also *content-blind*: they model student-skill interaction without any representation of the resource the student engaged with, so they cannot distinguish "watch the video" from "attempt the quiz" even though the two interventions can have very different effects. DKT has also been repeatedly criticised on evaluation — Wilson et al. (2016) showed that simpler models match its claimed performance under fairer evaluation protocols. The lesson I take from this is that any deep-learning component must be evaluated against simpler baselines, not in isolation.

## Content-Based Recommendation for MOOCs: Bousbahi & Chorfi (2015)

Bousbahi & Chorfi (2015) take the opposite stance to Thai-Nghe et al. Their MOOC-Rec system represents each course by a vector of metadata features (topic, level, language, prerequisites, duration) and recommends courses whose feature vector is closest to a learner's stated profile. The approach is case-based: a new query is matched to the closest historical case, and the recommendation is explainable because the matching dimensions can be surfaced to the learner.

The strengths are real and complement Thai-Nghe et al. directly. Content-based methods do not need historical interaction data and so handle cold-start gracefully, and they are auditable: a recommender can explain *why* it suggested a resource by naming the features that contributed most. In education, where unexplained algorithmic recommendations can erode learner trust and where new courses and new learners arrive frequently, both properties matter.

The limitations are also real. First, there is no collaborative component, so the system cannot learn from what worked for similar learners. If learners with profile *P* historically did better with resource *A* than resource *B*, MOOC-Rec has no way to discover this. Second, the evaluation is standard information-retrieval metrics on a small expert-curated test set; there is no measurement of whether the recommendations led to better learning. Third, the feature representation is essentially textual metadata and ignores behavioural features such as historical engagement patterns or the outcome correlations of the resource. For my project, Bousbahi & Chorfi justify the content-based component of the hybrid — particularly for cold-start — but also show why content-based alone is not enough.

## Technical Foundations

Koren, Bell & Volinsky (2009) set out the algebra and optimisation of SVD-based recommendation. I plan to use truncated SVD, following their formulations, with regularisation and biased baseline terms as they recommend.

Ricci, Rokach & Shapira (2015) cover evaluation methodology for recommender systems in depth. Their treatment of offline evaluation — choice of metric, construction of train/test splits, trade-offs between precision-style and recall-style metrics — underpins my evaluation strategy in [@sec:design] of this report. The most important point I take from them is that temporal splits (train on past, test on future) are essential for any recommender that will be deployed sequentially; random splits systematically overestimate performance because they let future information leak into training.

Rendle et al. (2012) address a problem that is central to me: the OULAD data is *implicit* (the learner accessed a resource) rather than *explicit* (the learner rated a resource). Treating absence-of-interaction as negative evidence is the well-known trap of implicit-feedback recommenders, and BPR's pairwise-ranking formulation is the standard mitigation. My evaluation framework follows their lead in treating ranking quality (NDCG) as the primary success criterion rather than rating-prediction accuracy.

## Critical Synthesis and Identified Gap

Mapping the four primary works against the three research questions makes the pattern clear.

| Work | Modelling approach | Uses content features | Uses outcomes | Evaluation focus |
|------|-------------------|----------------------|--------------|------------------|
| Thai-Nghe et al. (2010) | Collaborative (MF) | No | Outcome-as-target | RMSE prediction |
| Corbett & Anderson (1995) | Knowledge tracing (HMM) | No | Outcome-as-target | Knowledge estimation |
| Piech et al. (2015) | Knowledge tracing (RNN) | No | Outcome-as-target | Next-answer prediction |
| Bousbahi & Chorfi (2015) | Content-based | Yes | No | IR metrics |
| **My project** | **Hybrid (CF + content)** | **Yes** | **Outcome-as-evaluation signal** | **Ranking + outcome-weighted** |

Two specific gaps come out of this. The first is *architectural*: no published work combines collaborative filtering, content features, and an outcome signal in a single recommender evaluated on a public educational dataset. Each strand of research has stayed inside its own paradigm. The second is *methodological*: where outcomes are used at all, they are used as the *target* of prediction (Thai-Nghe et al.; Piech et al.), not as a *quality signal for recommendation evaluation*. The distinction matters. Predicting that a learner will fail an assessment is not the same as recommending content that will reduce the probability of failure; only the second is actionable for the learner.

There is one further methodological concern, taken from Wilson et al.'s critique of DKT and from Ricci et al.'s argument for temporal splits: any system claiming improvement over baselines has to be evaluated against carefully chosen baselines under a protocol that does not allow leakage. I address this directly with a graded model progression (Random $\rightarrow$ Popularity $\rightarrow$ SVD $\rightarrow$ Content $\rightarrow$ Hybrid) under a temporal 80/20 split, paired statistical testing across folds, and an ablation study to isolate the contribution of each component.

I am therefore positioning my contribution not as a new algorithm but as a *systematic combination* — collaborative filtering, feature-based content matching, and outcome-aware evaluation — implemented and evaluated end-to-end on a public dataset, with enough methodological rigour to allow direct comparison against the prior work surveyed here.

*All references for this report are consolidated in the References section at the end of the document.*
