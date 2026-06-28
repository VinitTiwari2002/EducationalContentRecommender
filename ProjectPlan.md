# Project Plan: Educational Content Recommender
## CM3005 Data Science — CM3070 Final Project
### Vinit Tiwari | 220174440

---

## KEY DEADLINES

| Milestone | Due (approx.) | Weight | Status |
|-----------|---------------|--------|--------|
| Project Proposal Video | Week 4 (May 18) | 0% formative | ✅ DONE |
| Preliminary Project Report | Week 10 (~June 29) | 10% | ⬜ TODO |
| Draft Report | Week 18 (~Aug 24) | 0% formative | ⬜ TODO |
| Written Exam | Week 22 (~Sep 21) | 20% | ⬜ TODO |
| Final Report + Code + Video | Week 24 (~Oct 5) | 60% + 5% | ⬜ TODO |

---

## PHASE 1: DATA & EXPLORATION (Weeks 5-6 | ~8 hours)
**Goal:** Understand the data completely before building anything.

### Week 5 Tasks:
- [x] Set up Git repository (private, structure per plan) ✅
- [ ] Set up Python virtual environment + requirements.txt
- [ ] Download OULAD dataset from Kaggle
- [ ] Load all 7 CSV files into pandas
- [ ] Explore schema: row counts, column types, nulls, distributions

### Week 6 Tasks:
- [ ] EDA notebook:
  - [ ] How many students per course/presentation?
  - [ ] Distribution of final_result (pass/fail/distinction/withdrawn)
  - [ ] Activity type distribution in VLE (what types of resources exist?)
  - [ ] Interaction frequency over time (when do students access resources?)
  - [ ] Assessment score distributions
  - [ ] Correlation: interaction volume vs. assessment scores vs. final result
- [ ] Document key findings (these go into Prelim Report)
- [ ] **Git commit:** `v0.1-eda`

### Deliverable: EDA notebook with visualisations and key insights.

---

## PHASE 2: PREPROCESSING & FEATURE ENGINEERING (Weeks 7-8 | ~12 hours)
**Goal:** Build the interaction matrix and content feature vectors.

### Week 7 Tasks:
- [ ] Build user-item interaction matrix
  - Rows: students | Columns: VLE resources
  - Values: sum_click (or binary: accessed/not accessed)
  - Decide: use all presentations or subset?
- [ ] Temporal train/test split
  - Train: first 80% of interactions chronologically
  - Test: last 20%
  - Ensure no data leakage (future interactions not in train)
- [ ] Handle sparsity (most students access only a fraction of resources)

### Week 8 Tasks:
- [ ] Build content feature vectors for each VLE item:
  - activity_type (one-hot encoded: resource, quiz, forum, ouwiki, etc.)
  - week_from, week_to (normalised temporal position)
  - Module/presentation context
  - Popularity features (how many students accessed it?)
  - Outcome correlation (average score of students who accessed it)
- [ ] Build evaluation framework (functions for Precision@10, Recall@10, NDCG)
- [ ] **Git commit:** `v0.2-preprocessing`

### Deliverable: Clean interaction matrix + feature vectors + evaluation functions.

---

## PHASE 3: BASELINE MODELS (Weeks 9-10 | ~8 hours)
**Goal:** Establish performance floor that your models must beat.

### Week 9 Tasks:
- [ ] Implement Random baseline
  - Randomly recommend K items from catalogue
  - Measure: Precision@10, Recall@10, NDCG
- [ ] Implement Popularity baseline
  - Recommend the K most-accessed items globally
  - Measure same metrics
- [ ] Compare and document results

### Week 10 Tasks:
- [ ] Write up results for Preliminary Report
- [ ] **Submit Preliminary Project Report**
- [ ] **Git commit:** `v0.3-baselines`

### Deliverable: Baseline results table + Preliminary Report submitted.

---

## PRELIMINARY REPORT — WHAT TO INCLUDE (Week 10 submission)

Based on professor's guidance, this should demonstrate progress:

1. **Introduction & Project Description** (~500 words)
   - What you're building and why
   - Template reference

2. **Literature Review** (~1000 words)
   - 4 papers evaluated (Thai-Nghe, Piech, Corbett, Bousbahi)
   - Gap identification
   - How your project addresses the gap

3. **Data Description** (~500 words)
   - OULAD dataset overview
   - Key statistics from EDA
   - 2-3 visualisations

4. **Methodology** (~500 words)
   - Hybrid approach description
   - Model progression plan
   - Evaluation metrics explained

5. **Progress So Far** (~300 words)
   - EDA completed
   - Preprocessing done
   - Baseline results (random, popularity)
   - What's next

6. **Timeline** (~200 words)
   - Remaining work plan

**Total: ~3000 words + figures + references**

---

## PHASE 4: COLLABORATIVE FILTERING (Weeks 11-12 | ~15 hours)
**Goal:** Implement SVD-based collaborative filtering.

### Week 11 Tasks:
- [ ] Implement SVD (Truncated SVD / Surprise library)
  - Train on user-item interaction matrix
  - Tune: number of latent factors (10, 20, 50, 100)
  - Tune: regularisation parameters
- [ ] Generate top-K recommendations for test users
- [ ] Evaluate: Precision@10, Recall@10, NDCG

### Week 12 Tasks:
- [ ] Compare SVD vs baselines (should clearly outperform)
- [ ] Analyse: which users does it work well for? Poorly for?
- [ ] Document cold-start failures (new users with few interactions)
- [ ] **Git commit:** `v0.4-collaborative`

### Deliverable: Working SVD model with evaluation results.

---

## PHASE 5: FEATURE-BASED CONTENT FILTERING (Weeks 13-14 | ~12 hours)
**Goal:** Recommend items with similar features to what the user has engaged with.

### Week 13 Tasks:
- [ ] Build item-item similarity matrix using content features
  - Cosine similarity on feature vectors (activity_type, week, structure)
- [ ] For each user: profile = aggregated features of items they've accessed
- [ ] Recommend: items most similar to user profile that they haven't seen

### Week 14 Tasks:
- [ ] Evaluate: Precision@10, Recall@10, NDCG
- [ ] Compare vs collaborative (typically: SVD wins on warm users, content wins on cold-start)
- [ ] Analyse strengths/weaknesses of each approach
- [ ] **Git commit:** `v0.5-content-based`

### Deliverable: Working content-based model with comparative evaluation.

---

## PHASE 6: HYBRID MODEL (Weeks 15-16 | ~15 hours)
**Goal:** Combine both approaches and show improvement.

### Week 15 Tasks:
- [ ] Implement weighted hybrid:
  - `hybrid_score = α × collab_score + (1-α) × content_score`
  - Tune α on validation set (grid search: 0.1 to 0.9)
- [ ] Implement switching hybrid:
  - Use collaborative for warm users (>N interactions)
  - Use content-based for cold users (<N interactions)
- [ ] Optionally: incorporate outcome signal
  - Weight recommendations by average assessment score of users who accessed that item

### Week 16 Tasks:
- [ ] Evaluate hybrid vs individual models
- [ ] Ablation study: remove each component, measure drop
- [ ] Statistical significance test (paired t-test across folds)
- [ ] **Git commit:** `v0.6-hybrid`

### Deliverable: Hybrid model outperforming both individual approaches (or documented analysis of why not).

---

## PHASE 7: API & DASHBOARD (Weeks 17-18 | ~15 hours)
**Goal:** Make it demo-able.

### Week 17 Tasks:
- [ ] FastAPI endpoint:
  - `POST /recommend` — takes student_id, returns top-10 recommendations with scores
  - `GET /student/{id}/history` — returns student's interaction history
  - `GET /evaluate` — returns model performance metrics
- [ ] Basic input validation and error handling
- [ ] API documentation (auto-generated by FastAPI/Swagger)

### Week 18 Tasks:
- [ ] Streamlit dashboard:
  - Select a student → see their profile
  - See top-10 recommendations with explanation (why recommended)
  - Compare: what would each model recommend?
  - Performance metrics visualisation
- [ ] **Submit Draft Report (formative)**
- [ ] **Git commit:** `v0.7-api-dashboard`

### Deliverable: Working API + demo dashboard + Draft Report submitted.

---

## PHASE 8: FINAL EVALUATION & ANALYSIS (Weeks 19-20 | ~15 hours)
**Goal:** Rigorous evaluation for the final report.

### Week 19 Tasks:
- [ ] 5-fold cross-validation (all models)
- [ ] Generate final results table
- [ ] Ablation study (final, clean version)
- [ ] Statistical significance testing
- [ ] Generate all plots/visualisations for report:
  - Precision/Recall/NDCG comparison bar chart
  - Model performance progression chart
  - Ablation results
  - Hyperparameter sensitivity plots

### Week 20 Tasks:
- [ ] Analyse failure cases: where does the hybrid still fail?
- [ ] Discuss limitations and future work
- [ ] Prepare all figures for report (publication quality)
- [ ] **Git commit:** `v0.8-evaluation`

### Deliverable: Complete evaluation with all figures ready for report.

---

## PHASE 9: REPORT WRITING (Weeks 17-22 | ~35 hours)
**Goal:** Write the final report. Start early, iterate.

### Structure (based on course requirements):

| Section | Content | ~Words |
|---------|---------|--------|
| Abstract | Summary of project, approach, key results | 250 |
| Introduction | Problem, motivation, project scope | 800 |
| Literature Review | 4+ papers, critical evaluation, gap | 1500 |
| Methodology | Data, preprocessing, models, evaluation design | 2000 |
| Implementation | Architecture, code structure, key decisions | 1500 |
| Results & Evaluation | Metrics tables, comparisons, ablation, significance | 2000 |
| Discussion | What worked, what didn't, limitations, future work | 1000 |
| Conclusion | Summary of contributions | 500 |
| References | All citations (ACM or consistent style) | - |
| Appendix | Additional figures, code snippets | - |
| **TOTAL** | | **~9500** |

### Writing schedule:
- Week 17: Introduction + Literature Review (reuse from Prelim Report, expand)
- Week 18: Methodology + Implementation (write as you build)
- Week 19: Results (fill in as evaluation completes)
- Week 20: Discussion + Conclusion
- Week 21: Full draft review, polish, consistency check
- Week 22: Final edits (light — focus on exam this week)

---

## PHASE 10: EXAM PREP (Week 21-22 | 3-4 hours)
**Goal:** Prepare for the written exam.

- [ ] Review mock exam paper
- [ ] Review past papers (when released)
- [ ] Key topics to prepare:
  - How to differentiate source quality (journals vs blogs)
  - How you carried out your project (process)
  - Evaluation methodology and why you chose it
  - Ethical considerations
  - What you would do differently

---

## PHASE 11: DEMO VIDEO & FINAL SUBMISSION (Weeks 23-24 | ~8 hours)
**Goal:** Record demo video and submit everything.

### Week 23 Tasks:
- [ ] Record 3-5 min demo video showing:
  - The system working (API + dashboard)
  - Key results (model comparison)
  - Your contribution and what you learned
- [ ] Export as MP4
- [ ] Final code cleanup:
  - Remove dead code
  - Ensure requirements.txt is complete
  - README with clear setup instructions
  - Test: can someone else run this from scratch?

### Week 24 Tasks:
- [ ] Final report proofread
- [ ] Make GitHub repo public (for marking)
- [ ] **Submit: Final Report + Code + Video**
- [ ] **Git tag:** `v1.0-final`

---

## WEEKLY HOUR ESTIMATES

| Week | Phase | Hours | Cumulative |
|------|-------|-------|-----------|
| 5 | Setup + Data download | 4h | 4h |
| 6 | EDA | 4h | 8h |
| 7 | Interaction matrix + split | 6h | 14h |
| 8 | Feature engineering + eval framework | 6h | 20h |
| 9 | Baselines (random, popularity) | 4h | 24h |
| 10 | Prelim Report writing | 4h | 28h |
| 11 | SVD implementation + tuning | 8h | 36h |
| 12 | SVD evaluation + analysis | 7h | 43h |
| 13 | Content-based implementation | 6h | 49h |
| 14 | Content-based evaluation | 6h | 55h |
| 15 | Hybrid implementation | 8h | 63h |
| 16 | Hybrid eval + ablation | 7h | 70h |
| 17 | FastAPI + Report (Intro, Lit Review) | 10h | 80h |
| 18 | Streamlit + Report (Methodology) + Draft submission | 10h | 90h |
| 19 | Cross-validation + Report (Results) | 10h | 100h |
| 20 | Final analysis + Report (Discussion) | 10h | 110h |
| 21 | Report polish + Exam prep | 10h | 120h |
| 22 | Exam + light report edits | 6h | 126h |
| 23 | Demo video + code cleanup | 6h | 132h |
| 24 | Final submission | 4h | 136h |
| — | **Buffer** | **14h** | **150h** |

**Average: 7.5 hours/week = ~1.5 hours on weekdays**

---

## GIT MILESTONES

```
v0.1-eda          — EDA complete, key findings documented
v0.2-preprocessing — Interaction matrix + feature vectors + eval functions
v0.3-baselines    — Random + Popularity baselines with results
v0.4-collaborative — SVD model working + evaluated
v0.5-content-based — Feature-based content model working + evaluated
v0.6-hybrid       — Hybrid model + ablation + significance tests
v0.7-api-dashboard — FastAPI + Streamlit working
v0.8-evaluation   — Final evaluation complete, all figures generated
v1.0-final        — Submission-ready
```

---

## RISK MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Hybrid doesn't beat SVD alone | Medium | Low | Document as valid finding. Ablation still shows value. |
| OULAD data issues (sparsity) | Low | Medium | Already checked — 10M interactions is plenty |
| Time crunch at end | Medium | High | Start report writing at Week 17, not 22. Buffer of 14h. |
| Exam + project overlap | Medium | Medium | Report at 90% by Week 21. Week 22 = exam + light edits. |
| Feature engineering underperforms | Low | Low | Even basic activity_type features add signal vs pure collab. |

---

## IMMEDIATE NEXT STEP

**This week (Week 5):** Set up the project repository and download the data.

```bash
# Create repo structure
mkdir -p src data notebooks evaluation docs tests
touch requirements.txt README.md .gitignore

# Key Python packages
# requirements.txt:
# pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
# fastapi, uvicorn, streamlit
# jupyter, notebook
```
