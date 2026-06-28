# Literature Review {#sec:lit-review}

## Scope and Research Questions

This review focuses on the intersection of recommender systems and educational technology, with particular attention to four issues that the proposed project must address: (i) whether collaborative filtering techniques developed for commercial domains transfer to education, (ii) how content features can substitute for or complement collaborative signal when interaction data is sparse, (iii) how learning outcomes have been used as a recommendation signal, and (iv) how educational recommenders have been evaluated. Three research questions structure the review:

> **1.** What modelling approaches have been shown to work for recommendation in educational settings, and what are their respective limitations?
>
> **2.** How have prior systems incorporated — or failed to incorporate — learning outcomes into either the model or its evaluation?
>
> **3.** What evaluation methodology is appropriate for an offline educational recommender on a public dataset, and what metrics are defensible?

The review proceeds by examining four primary works in detail, each addressing one or more of these questions, before drawing methodological inferences from three supporting technical references on matrix factorisation, evaluation, and ranking. A short synthesis at the end identifies the gap that motivates the project.

## Collaborative Filtering in Education: Thai-Nghe et al. (2010)

Thai-Nghe et al. (2010) were among the first to apply matrix factorisation. Their study reframes student performance prediction as a recommender-system problem: students are users, tasks are items, and the score a student achieves on a task is the implicit rating. They show that matrix factorisation outperforms traditional regression methods on three educational datasets, including the well-known KDD Cup 2010 Algebra and Bridge to Algebra datasets.

The contribution is significant: it demonstrates that the algebraic machinery of collaborative filtering is not domain-specific. Latent-factor models can capture student-level and task-level abilities even without the textual or behavioural metadata used in commercial systems.

The paper's limitations, however, bear directly on the present project. First, the system *predicts* — it estimates how a student would score on a known task — but it does not *recommend* unseen resources for the student to study next. The two problems are related but distinct: prediction tells the learner where they stand, while recommendation tells them where to go. Second, the model treats every task as an opaque identifier; it has no notion of task difficulty, topic, or content. This makes the model purely collaborative in the strictest sense, with the consequence that new tasks (or new students) suffer the classical cold-start problem. Third, evaluation is restricted to RMSE on held-out scores; whether following the model's predictions would have *improved* the student's learning outcomes is not addressed and, given the dataset, could not have been. For this project, Thai-Nghe et al. establish the empirical viability of matrix factorisation in education — justifying the inclusion of an SVD model in the model progression — but they also illustrate exactly why a pure collaborative approach is insufficient.

## Modelling Knowledge State: Corbett & Anderson (1995); Piech et al. (2015)

A parallel strand of research treats education not as a recommendation problem at all but as a knowledge-state estimation problem. The seminal contribution is Corbett & Anderson's (1995) Bayesian Knowledge Tracing (BKT). BKT models student knowledge as a hidden state, updated after each observed practice item according to four parameters: the prior probability of knowing a skill, the probability of transitioning from "not known" to "known" given a learning opportunity, and the slip and guess probabilities that determine the relationship between knowledge state and observed performance. The model is principled, interpretable, and the empirical workhorse of the Cognitive Tutor family of intelligent tutoring systems.

Piech et al. (2015) introduced Deep Knowledge Tracing (DKT). DKT replaces the BKT Markov model with a recurrent neural network (LSTM) that learns its own latent knowledge representation from raw sequences of student responses. DKT substantially outperforms BKT on standard benchmarks, in large part because it sheds the assumption that skills are independent: an LSTM can capture cross-skill transfer effects that BKT cannot represent.

These papers establish two important points for the present project. First, prior assessment performance is genuinely informative about learner state and should not be discarded. This justifies the project's intention to use assessment scores as a feature in the hybrid model. Second, and more critically, both papers stop at *knowledge estimation*; neither addresses what to recommend next. BKT's deployment in tutoring systems typically uses hand-authored progression rules over the estimated knowledge state; DKT is purely evaluative. Furthermore, both methods are *content-blind*: they model student-skill interaction without any representation of the resource the student engaged with, which means they cannot distinguish "watch the video" from "attempt the quiz" even if those interventions have very different effects. Finally, DKT has been repeatedly criticised for evaluation methodology — Wilson et al. (2016) showed that simpler models match its claimed performance under fairer evaluation protocols. The present project takes the lesson that any deep-learning component must be evaluated against simpler baselines, not in isolation.

## Content-Based Recommendation for MOOCs: Bousbahi & Chorfi (2015)

Bousbahi & Chorfi (2015) take the opposite stance to Thai-Nghe et al. It represents each course by a vector of metadata features (topic, level, language, prerequisites, duration) and recommends courses whose feature vector is closest to a learner's stated profile. The approach is case-based — a new query is matched to the closest historical case — and produces explainable recommendations, since the matching dimensions can be surfaced to the learner.

The strengths of this approach are real and complementary to those of Thai-Nghe et al. Content-based methods do not require historical interaction data and therefore handle cold-start gracefully. The matching process is auditable: a recommender can explain *why* it suggested a resource by naming the features that contributed most. These properties are valuable in education, where unexplained algorithmic recommendations may erode learner trust and where new courses (and indeed new learners) are added frequently.

The limitations are equally real. First, the system has no collaborative component, which means it cannot learn from the wisdom of similar learners — if learners with profile *P* historically did better with resource *A* than resource *B*, there is no machinery to discover this. Second, evaluation is restricted to standard information-retrieval metrics on a small expert-curated test set; there is no measurement of whether the recommendations led to better learning outcomes. Third, the feature representation is essentially textual metadata and does not include behavioural features such as historical engagement patterns or outcome correlations of the resource. For this project, Bousbahi & Chorfi justify the use of a content-based component in the hybrid model — particularly for handling cold-start — but they also show why content-based alone is insufficient.

## Technical Foundations

Three supporting references inform the methodological choices of the project.

Koren, Bell & Volinsky (2009) lay out the algebraic and optimisation foundations of SVD-based recommendation. The project uses truncated SVD via the `surprise` library or `scipy.sparse.linalg.svds` following the formulations described in this article, with regularisation and biased baseline terms as recommended.

Ricci, Rokach & Shapira (2015) provide a comprehensive treatment of evaluation methodology for recommender systems. Their treatment of offline evaluation — including the choice of metric, the construction of train/test splits, and the trade-offs between precision-style and recall-style metrics — directly underpins the evaluation strategy presented in [@sec:design] of this report. Particularly important is their argument that temporal splits (train on past, test on future) are essential for any recommender that will be deployed sequentially; randomly sampled splits systematically overestimate performance because they allow future information to leak into the training set.

Rendle et al. (2012) address a problem that is central to educational recommendation: the data is *implicit* (the learner accessed a resource) rather than *explicit* (the learner rated the resource). Treating absence-of-interaction as negative evidence is the well-known trap of implicit-feedback recommenders, and BPR's pairwise-ranking formulation is the standard mitigation. The project's evaluation framework follows BPR's lead in treating ranking quality (NDCG) as the primary success criterion rather than rating-prediction accuracy.

## Critical Synthesis and Identified Gap

Mapping the four primary works against the three research questions reveals a clear pattern.

| Work | Modelling approach | Uses content features | Uses outcomes | Evaluation focus |
|------|-------------------|----------------------|--------------|------------------|
| Thai-Nghe et al. (2010) | Collaborative (MF) | No | Outcome-as-target | RMSE prediction |
| Corbett & Anderson (1995) | Knowledge tracing (HMM) | No | Outcome-as-target | Knowledge estimation |
| Piech et al. (2015) | Knowledge tracing (RNN) | No | Outcome-as-target | Next-answer prediction |
| Bousbahi & Chorfi (2015) | Content-based | Yes | No | IR metrics |
| **This project** | **Hybrid (CF + content)** | **Yes** | **Outcome-as-evaluation signal** | **Ranking + outcome-weighted** |

Two specific gaps emerge from this synthesis. The first is *architectural*: no published work combines collaborative filtering, content features, and outcome signal in a single recommender evaluated on a public educational dataset. Each strand of research has stayed within its own paradigm. The second is *methodological*: where outcomes are used at all, they are used as the *target* of prediction (Thai-Nghe et al.; Piech et al.), not as a *quality signal for recommendation evaluation*. The distinction matters. Predicting that a learner will fail an assessment is not the same as recommending content that will reduce the probability of failure; only the latter is genuinely actionable for the learner.

A further methodological concern, drawn from Wilson et al.'s critique of DKT and from the temporal-split argument in Ricci et al., is that any system claiming improvement over baselines must be evaluated against carefully chosen baselines using a protocol that does not allow leakage. The project addresses this directly with a graded model progression (Random $\rightarrow$ Popularity $\rightarrow$ SVD $\rightarrow$ Content $\rightarrow$ Hybrid) under a temporal 80/20 split, paired statistical testing across folds, and an ablation study to isolate the contribution of each component.

The project's contribution is therefore positioned not as a new algorithm but as a *systematic combination* — collaborative filtering, feature-based content matching, and outcome-aware evaluation — implemented and evaluated end-to-end on a publicly available dataset, with sufficient methodological rigour to allow direct comparison against the prior work surveyed here.
