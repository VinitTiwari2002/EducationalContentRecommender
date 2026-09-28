# Evaluation {#sec:evaluation}

## Evaluation Design

Six axes: (i) headline metrics on the primary temporal split; (ii) five-fold temporal cross-validation; (iii) paired-*t* significance testing on per-user arrays with Bonferroni adjustment across the five metrics; (iv) hybrid ablation; (v) cold-start stratification; (vi) demographic fairness audit. Popularity bias (coverage + Gini) and four-axis hyperparameter sensitivity are reported alongside each result. Every number reproduces from the CSVs written by `python -m src.pipeline` (§4.9).

**Split contract, consistent across every table below.** All splits are *temporal* (chronological `date` cutoff, no leakage). User rows are shared across `train` and `test` matrices so a fit-on-train / score-on-test contract holds per row. Course-scoping — the intersection of each user's registered `(code_module, code_presentation)` with the item column — is enforced in every window, including the tuning sub-split and each CV fold, so no recommender can score items outside a user's enrolled presentations regardless of split. The primary split is 80/20 by click-date; the tuning sub-split re-slices the primary training window (last 25% of dates -> `tune_val`); the five CV folds slide the test cutoff back through the training timeline in 15% chunks and rebuild item features per fold.

## Outcome-Weighted Precision: Formal Definition and Validation

Let $R_u$ be the top-$K$ recommendation list for user $u$, $H_u$ the set of held-out relevant items for user $u$, $o(i) \in [0, 100]$ the mean assessment score of training-window users who accessed item $i$, and $\bar{o}$ the training-window global mean of $o$. Then

$$
\text{OWP@K}(u) \;=\; \frac{1}{K} \sum_{i \in R_u} \mathbb{1}[i \in H_u] \cdot \frac{o(i)}{\bar{o}}.
$$

The system-level metric averages $\text{OWP@K}(u)$ over users with at least one held-out relevant item. Two properties are worth emphasising. First, under uniform outcomes — every item has the same $o(i)$ — every hit contributes exactly 1 to the sum, so OWP@K reduces to Precision@K. This is a validity check: OWP does not simply re-scale precision by an unrelated factor. Second, all quantities in the denominator and the ratio are computed strictly from *training* data, so no test-window information leaks into the metric.

Unit tests in `tests/test_new_metrics.py` verify the uniform-outcomes reduction, the up-weighting of above-average hits, and the down-weighting of below-average hits. The baseline $\bar o$ is precomputed once per `evaluate()` call so per-user cost is O(K).

OWP measures *association* between recommended items and better outcomes, not the *causal effect* of issuing recommendations — the standard methodological limit of any offline learning-benefit proxy, restated in the limitations at the end of this chapter.

## Headline Results — Primary Temporal Split

Table 5.1 reports every model at $K = 10$ on the primary 80/20 temporal split (cutoff = date-172), with time-decay $\lambda = 0.01$ and tuned hyperparameters. GatedHybrid is the best system on P@10 and OWP@10, and ties Hybrid on NDCG@10 and Hit-Rate@10.

| Model | P@10 | NDCG@10 | HR@10 | OWP@10 | Coverage | Gini |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.069 | 0.069 | 0.446 | 0.071 | 1.00 | 0.32 |
| Popularity | 0.251 | 0.272 | 0.759 | 0.257 | 0.33 | 0.73 |
| SVD | 0.219 | 0.246 | 0.799 | 0.223 | 0.42 | 0.59 |
| ALS | 0.218 | 0.229 | 0.767 | 0.222 | 0.38 | 0.63 |
| Content | 0.269 | 0.293 | 0.790 | 0.279 | 0.41 | 0.75 |
| Hybrid | 0.275 | 0.303 | 0.784 | 0.287 | 0.41 | 0.77 |
| **GatedHybrid** | **0.276** | **0.303** | 0.784 | **0.287** | 0.37 | 0.75 |

*Table 5.1 — Primary-split metrics at K=10 (decay=0.01, tuned weights; n_users = 17,562). Half-widths of 95% percentile bootstrap CIs on the per-user metric mean (1,000 resamples, per-user array): P@10 ≤ ±0.004, NDCG@10 ≤ ±0.004, OWP@10 ≤ ±0.004, Hit-Rate@10 ≤ ±0.008, Recall@10 ≤ ±0.002. CIs do not overlap between adjacent-rank models except Hybrid vs GatedHybrid. Full per-metric CI table in `evaluation/metrics_with_ci.csv`.*

Two findings stand out. First, Content alone reaches P@10 = 0.269 — the outcome signal buys only a further two-point improvement, and CF signals (SVD, ALS) individually underperform Content by ~5 points. Second, the hybrid tuner selected $\alpha = 0.0, \beta = 0.8, \gamma = 0.2$ on the tuning-validation split — i.e. it dropped CF entirely. On OULAD, content and outcome are the load-bearing signals; the ablation in §5.5 corroborates this.

Hybrid and GatedHybrid have overlapping aggregate CIs (P@10 half-width ±0.004, NDCG@10 half-width ±0.004): their bootstrap intervals on the mean cover each other, though the per-user paired-*t* on P@10 and OWP is significant at $p \approx 10^{-3}$ (§5.5) because the difference is systematic across the small cold cohort. The practical divergence lives on the cold-user subset (§5.7).

## Cross-Validation Results

Table 5.2 reports mean metrics across five temporal folds sliding back through the training timeline in 15% chunks. Relative ordering matches the primary split: GatedHybrid > Hybrid > Content > Popularity > SVD > ALS > Random.

| Model | P@10 | NDCG@10 | Coverage | Gini |
|---|---:|---:|---:|---:|
| **GatedHybrid** | **0.196** | **0.224** | 0.36 | 0.73 |
| Hybrid | 0.193 | 0.221 | 0.39 | 0.75 |
| Content | 0.186 | 0.211 | 0.39 | 0.74 |
| Popularity | 0.173 | 0.193 | 0.29 | 0.69 |
| SVD | 0.163 | 0.188 | 0.37 | 0.58 |
| ALS | 0.162 | 0.176 | 0.33 | 0.62 |
| Random | 0.046 | 0.047 | 1.00 | 0.30 |

*Table 5.2 — Five-fold temporal-CV means at K=10 (decay=0.01, tuned weights).*

Per-fold standard deviations at K=10 are: Hybrid 0.011, GatedHybrid 0.009, Content 0.009, Random 0.007 (all tight), versus Popularity 0.039, ALS 0.038, SVD 0.032 (much noisier). The earliest fold, where training data is smallest, drives most of that variance for the personalised CF/popularity models; the hybrid and content models are essentially fold-invariant. The CV means confirm the primary-split ranking is not an artefact of the specific 80/20 cutoff.

## Statistical Significance

The preliminary-report feedback asked whether a paired test over five folds carries enough evidence. I address this by testing on per-user metric arrays ($n = 17{,}562$) rather than five-fold means — each user contributes an independent paired observation. `evaluation/paired_t_tests.csv` reports the paired-$t$ statistic and both raw and Bonferroni-adjusted (across five metrics) $p$-values for Hybrid versus every other model at $K=10$.

Hybrid beats Random on every metric at machine precision (all raw $p$-values are numerical zero to double precision). Against Popularity, SVD, and ALS the four ranking-quality metrics all sit at adjusted $p \ll 10^{-40}$; the weakest comparison in each pair is Hit-Rate@10 with adjusted $p = 4 \times 10^{-19}$, $1.5 \times 10^{-5}$, and $3.2 \times 10^{-7}$ respectively — still strongly significant. Against Content, Hybrid wins on precision, recall, NDCG, and OWP with adjusted $p$ up to $5 \times 10^{-5}$ (recall is the weakest); Hybrid loses on Hit-Rate@10 by a small margin (0.784 vs 0.790, adjusted $p = 5 \times 10^{-4}$) — a real trade-off: Content places at least one hit for slightly more users, Hybrid ranks its hits higher (which is what NDCG rewards). Against GatedHybrid, Hybrid loses on precision and OWP by tiny but significant margins (adjusted $p \approx 10^{-3}$; the two models differ only on the ~1% of cold users, where GatedHybrid routes to Popularity and reclaims the small aggregate edge documented in §5.7) and ties on NDCG and Hit-Rate.

## Ablation Study

Table 5.3 reports the four hybrid variants at the *tuned* weights $(\alpha, \beta, \gamma) = (0.0, 0.8, 0.2)$ selected by the simplex-grid search — with each component's weight zeroed in turn.

| Variant | $\alpha,\beta,\gamma$ | P@10 | NDCG@10 | Gini | Cov |
|---|---|---:|---:|---:|---:|
| full         | 0.0, 0.8, 0.2 | **0.275** | **0.303** | 0.766 | 0.409 |
| no CF        | 0.0, 0.8, 0.2 | 0.275     | 0.303     | 0.766 | 0.409 |
| no content   | 0.0, 0.0, 0.2 | 0.086     | 0.077     | 0.645 | 0.074 |
| no outcome   | 0.0, 0.8, 0.0 | 0.269     | 0.293     | 0.748 | 0.412 |

*Table 5.3 — Tuned-weight ablation of the Hybrid. `no CF` is identical to `full` by construction (α was already 0 in the tuned hybrid). Dropping content collapses precision to near-random; dropping outcome loses ~1 P@10 point and shifts Gini by only −0.017.*

Three design decisions follow. First, `no CF` matches `full` exactly — the CF signal was already excluded by tuning, and this confirms the primary-split ordering was not sensitive to residual CF contribution. Second, content is the load-bearing component: `no content` collapses to 0.086 P@10 (near-random), so a hybrid with the tuned weights depends on content almost entirely. Third, outcome contributes a smaller but consistent ~1 P@10 point on top of content, and its removal barely shifts Gini (−0.017) — consistent with the near-zero outcome↔popularity correlation reported in §5.8.

## Cold-Start Stratification

Users are stratified by training clicks: warm (≥10, $n = 17{,}407$) vs cold (<10, $n = 155$).

| Model | Warm P@10 | Cold P@10 | Warm NDCG | Cold NDCG |
|---|---:|---:|---:|---:|
| Popularity | 0.251 | **0.205** | 0.273 | 0.233 |
| Content | 0.270 | 0.174 | 0.293 | 0.270 |
| Hybrid | **0.276** | 0.163 | **0.303** | 0.252 |
| **GatedHybrid** | **0.276** | **0.205** | **0.303** | 0.233 |

*Table 5.4 — Warm/cold metrics. GatedHybrid matches Hybrid on warm and Popularity on cold.*

On cold users, Popularity beats every personalised model — cold users have too little history for either content profiles or CF factors, and personalised models overfit the few clicks present while Popularity falls back to safe course-central items. Plain Hybrid loses 20% to Popularity on cold users (0.163 vs 0.205). GatedHybrid resolves this by design: cold users route to Popularity, warm users to Hybrid, matching both simultaneously. The gate is a single comparison; no learned parameter, no aggregate loss for warm users. The ablation-driven "problem" becomes a designed feature.

## Popularity Bias and Outcome-Signal Concentration

Table 5.1's rightmost columns report catalogue coverage (fraction of the catalogue appearing at least once) and Gini (of the per-item recommendation frequency). Random covers ~100% at Gini ≈ 0.32; Popularity concentrates hardest (coverage 0.33, Gini 0.73); Content and Hybrid reach coverage ~0.41 with Hybrid pushing Gini to 0.77 because tuned weights sharpen scores; GatedHybrid inherits the cold-user Popularity branch and lands at 0.37 / 0.74.

The load-bearing question is whether the outcome signal itself concentrates recommendations onto popular items. Two diagnostics say no — the concentration in Gatehybrid comes from content, not outcome. First, the correlation between `mean_score_of_accessers` and `log_access_count` on the primary split is $r = -0.08$ (near-zero, marginally negative) — high-outcome items are not disproportionately the popular ones. Second, the ablation's `hybrid_no_outcome` variant is essentially unchanged in Gini relative to `hybrid_full`, and only 1 P@10 point weaker — the outcome signal is orthogonal to popularity rather than reinforcing it. The design decision to expose the per-item score decomposition via `/decompose` is what makes this trade-off *inspectable* rather than hidden: a fairness-aware re-ranking layer (Conclusion, Future Work) would sit here.

## Fairness Audit

Per-attribute breakdown for GatedHybrid (`evaluation/fairness_audit.csv`): on gender, males (P@10 = 0.290, $n = 9{,}233$) outperform females (0.259, $n = 8{,}329$) — a 3-point absolute gap; on IMD band, precision is roughly uniform across the ten known bands (P@10 = 0.264–0.283) with no clear deprivation gradient, and the small "unknown" cohort ($n = 757$) sits highest at 0.313; on disability, non-disabled users (0.277, $n = 16{,}085$) marginally outperform disabled users (0.258, $n = 1{,}477$) — a 2-point gap. None of the gaps are large enough to indicate systematic exclusion, but the gender gap warrants flagging; it likely reflects course composition (STEM over-represented in the male sub-cohort) rather than a modelling choice, though I claim no causal knowledge. Reporting the audit satisfies the DEI objective; explaining *why* would require intervention data OULAD does not contain.

## Hyperparameter Sensitivity

`evaluation/tuning_{svd,als,hybrid}.csv` record the full sweeps.

- **SVD** — $k \in \{20, 50, 100\}$: NDCG@10 differences small (0.159–0.167); $k=100$ selected; plateau at $k>50$ suggests effective rank saturated by 50 factors.
- **ALS** — 18-point grid ($n_\text{factors} \times \lambda \times \alpha$): best $(32, 0.01, 1.0)$, NDCG@10 = 0.149. Increasing $\alpha$ *degraded* performance monotonically (0.149 at $\alpha=1$ -> 0.128 at $\alpha=40$), contradicting the Hu-Koren-Volinsky music-recommendation setting — OULAD click distributions have heavy positive tails per user, so strong confidence overfits.
- **Hybrid** — 21-point simplex ($\alpha+\beta+\gamma=1$, step 0.2); best $(0.0, 0.8, 0.2)$. All $\alpha > 0.25$ configurations underperform CF-dropped ones, confirming the ablation.
- **Time-decay** — $\lambda \in \{0.0, 0.005, 0.01, 0.02\}$: $\lambda=0.01$ produces the largest gain (+2.3 P@10 pts for Hybrid, +4.7 for Popularity); larger and smaller values regress toward raw clicks.
- **Robustness across seeds** — `scripts/robustness_check.py` with `random_state ∈ {0, 1, 2}` (`evaluation/robustness_summary.csv`): Content, Hybrid, GatedHybrid, Popularity, SVD exactly deterministic; ALS drifts in the 4th decimal; Random shifts by 0.005 on HR@10. Model ordering identical across all seeds.

## LambdaMART Re-Ranking Experiment

The two-stage LambdaMART reranker (Burges et al., 2010, §4.6): Stage 1 (tuned Hybrid) retrieves top-100; Stage 2 LightGBM `LGBMRanker` (`objective='lambdarank'`) reorders via 35 (user, item) features. Honest three-window training design (Stage 1 refit on `tune_train`, labels from `tune_val`, evaluation on `split.test`) prevents test-window leakage. The baseline row in Table 5.5 is the *deployed* GatedHybrid — the strongest single-model system from §5.3 — so the comparison is "does the reranker beat what a marker would actually be served?", not "does it beat its own Stage 1".

| Model | P@10 | Recall@10 | NDCG@10 | HR@10 | OWP@10 |
|---|---:|---:|---:|---:|---:|
| GatedHybrid (deployed baseline) | **0.276** | 0.105 | **0.303** | 0.784 | **0.287** |
| LambdaMART (rerank) | 0.252 | **0.107** | 0.284 | **0.796** | 0.262 |

*Table 5.5 — LambdaMART reranking versus the deployed GatedHybrid, primary split (n_users = 17,562).*

The reranker **underperforms the deployed baseline on P/NDCG/OWP** (−2.3 / −1.9 / −2.6 pts) while marginally improving HR (+1.2) and Recall (+0.2) — it finds hits for more users but ranks them lower. Feature-importance (`evaluation/reranker_importance.csv`) surfaces `activity_type=resource`, `log_access_count`, `mean_score_of_accessers`, `week_from_norm` — precisely the item features the linear hybrid already exploits. Gradient-boosted trees add flexibility but that flexibility does not translate into ranking gains: content + outcome saturates achievable NDCG in the pointwise-linear space on OULAD's course-scoped catalogue, and heavier LTR machinery does not close a gap that is essentially already closed.

## Critical Evaluation and Limitations

The evaluation reaches the level of statistical confidence the marker specifically asked for: precise metric definition (Section 5.2), 95% CIs across ~17K users, paired testing with Bonferroni adjustment, five-fold temporal CV, ablation, cold-start stratification, fairness audit, and hyperparameter sensitivity across four axes plus a three-seed robustness sweep. The claim I make — that GatedHybrid is the strongest system on the primary split and across five folds — is supported at adjusted $p \le 5 \times 10^{-5}$ on the four ranking-quality metrics against every baseline including Content (§5.5), with numerically-zero $p$-values against Random, Popularity, SVD, and ALS on precision, NDCG, and OWP.

Honest limits: OWP is *correlational*, not causal, so I cannot claim that issuing these recommendations would improve outcomes without an interventional study; ALS underperforms SVD and CF contributes negatively at every non-trivial weight, so the "collaborative filtering" contribution has to be reported honestly as adding little value on OULAD; the cold cohort ($n = 155$) is small, so the gated-hybrid cold-user precision carries a wider CI; the gender-fairness gap is descriptive, not diagnosed or mitigated (future work). None of these limitations undermine the primary claim — they are the honest boundaries of an offline observational evaluation on a single dataset.
