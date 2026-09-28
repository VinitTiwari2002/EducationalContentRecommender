# Data-Driven Personalised Educational Content Recommendation

**CM3070 Final Project — CM3005 Data Science template**
Vinit Tiwari (220174440)

A hybrid recommendation engine for the OULAD dataset, combining collaborative filtering, feature-based content filtering, and outcome-aware evaluation, delivered end-to-end with a FastAPI service, Streamlit transparency dashboard, and a Docker image.

**Public code repository:** <https://github.com/VinitTiwari2002/EducationalContentRecommender>

## What's implemented

**Pipeline + evaluation**
- Data ingestion for all seven OULAD tables
- Sparse user-item interaction matrix with temporal 80/20 split, course-scoping, and optional exponential time-decay weighting
- **Seven recommenders**: Random, Popularity, SVD, ALS, Content, weighted Hybrid, GatedHybrid (Burke-switching cold/warm hybrid)
- **LambdaMART two-stage reranker** on top of Stage-1 GatedHybrid (LightGBM `LGBMRanker` with `objective='lambdarank'`)
- Evaluation harness: Precision@K, Recall@K, NDCG@K, Hit-Rate@K, **outcome-weighted Precision@K**, catalogue coverage, Gini
- Statistical rigour: 5-fold temporal CV, percentile bootstrap CIs (1,000 samples), paired-*t* with Bonferroni adjustment, ablation, cold-start stratification, fairness audit, hyperparameter tuning grids (SVD, ALS, hybrid weights), and a three-seed robustness sweep
- 116 unit tests, all passing

**Serving stack**
- **FastAPI** (`src/api.py`) — three read-only endpoints: `/health`, `/recommend/{student_id}`, `/decompose/{student_id}/{item_id}`. Loads persisted model artefacts once at startup so request cost is O(1). Hybrid and warm-GatedHybrid responses include per-item CF / Content / Outcome decomposition.
- **Streamlit dashboard** (`dashboard/app.py`) — three pages: *Recommendation Explorer* (top-K with stacked-bar score decomposition + per-item audit view), *Fairness View* (per-attribute breakdown by gender / IMD band / disability + warm-vs-cold cold-start table), *Ablation Comparison* (Content vs Hybrid vs GatedHybrid side-by-side).
- **Persistence** (`src/persistence.py`) — joblib-serialised `models.joblib`, `serving_context.joblib`, and a JSON `manifest.json` (tuned hyperparameters, git commit, shape metadata).
- **Docker** (`Dockerfile`) — two-stage build. Stage 1 installs `implicit` (Cython + OpenMP) against OpenBLAS deterministically; stage 2 is a slim runtime that serves the FastAPI app on port 8000 via uvicorn with a `/health` `HEALTHCHECK`.

## Quick start

### Local
```bash
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
pip install -r requirements.txt

# 1. Download OULAD into data/raw/ (manual step; see below)
# 2. Run the pipeline end-to-end (evaluates baselines, tunes, persists models):
python -m src.pipeline --persist-models

# 3. Boot the API
.venv/bin/python -m uvicorn src.api:app --port 8000

# 4. In a second terminal, boot the dashboard against the API
.venv/bin/streamlit run dashboard/app.py

# 5. Open the EDA notebook
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

`GET http://localhost:8000/health` returns the loaded roster + `cutoff_date`; the Streamlit UI defaults to `http://localhost:8501`.

### Querying a different student in the dashboard

The *Recommendation Explorer* and *Ablation Comparison* pages both expose a `Student ID` number input at the top. Type any `id_student` present in the loaded split (or use the ± steppers); the page re-queries the API on the next interaction. IDs not in the split return `404 student_id X not in split`.

To preload a demo ID on startup:

```bash
RECSYS_DEMO_STUDENT=28400 .venv/bin/streamlit run dashboard/app.py --server.port 8501
```

Find valid IDs from the persisted index:

```bash
.venv/bin/python -c "import numpy as np; ix=np.load('data/processed/student_index.npy'); print(ix[:10], '...', ix[-10:])"
```

To demo the GatedHybrid switching behaviour, pick a **cold user** (fewer than 10 training clicks) — the Content, Hybrid, and GatedHybrid columns of the Ablation Comparison page will diverge:

```bash
.venv/bin/python -c "
from scipy import sparse
import numpy as np
tr = sparse.load_npz('data/processed/train.npz')
ix = np.load('data/processed/student_index.npy')
n_clicks = np.asarray((tr!=0).sum(axis=1)).ravel()
print('cold student_ids:', ix[np.where(n_clicks < 10)[0][:5]].tolist())
"
```

Or hit the API directly for any student:

```bash
curl -sS 'http://localhost:8000/recommend/6516?k=10&model=Hybrid'
curl -sS 'http://localhost:8000/decompose/6516/877059'
```

### Runtime configuration

Three environment variables tune the service layer without code changes:

| Variable | Consumer | Default | Purpose |
|---|---|---|---|
| `RECSYS_MODELS_DIR` | `src/api.py`, Dockerfile | `data/processed/models` | Directory to load `models.joblib` / `serving_context.joblib` / `manifest.json` from. Set this if you have multiple persisted runs. |
| `RECSYS_API_URL` | `dashboard/app.py` | `http://localhost:8000` | Base URL the Streamlit dashboard sends `/health`, `/recommend`, `/decompose` requests to. Set when the API is on a different port or host. |
| `RECSYS_DEMO_STUDENT` | `dashboard/app.py` | `6516` | Default student ID pre-filled in the Recommendation Explorer and Ablation Comparison pages. |

### Docker

```bash
docker build -t recsys:latest .
docker run --rm -p 8000:8000 -v $(pwd)/data:/app/data recsys:latest
curl http://localhost:8000/health
```

The `-v $(pwd)/data:/app/data` mount is required — the image intentionally does not bundle raw OULAD CSVs (CC-BY, ~500 MB) nor the `data/processed/models/*.joblib` artefacts (change with every retrain). Persist the models once locally (`python -m src.pipeline --persist-models`), then the container picks them up. Files excluded from the image are listed in `.dockerignore` (venv, caches, reports, data, notebooks, `.git`).

### Downloading OULAD

Download the dataset from the UCI ML Repository
(<https://analyse.kmi.open.ac.uk/open_dataset>) or Kaggle, and extract the seven
CSVs into `data/raw/`:

```
data/raw/
├── assessments.csv
├── courses.csv
├── studentAssessment.csv
├── studentInfo.csv
├── studentRegistration.csv
├── studentVle.csv
└── vle.csv
```

## Pipeline CLI knobs

```
--rebuild-split          Force rebuild of cached split
--k 5 10 20              Evaluate at multiple K
--no-course-scoping      Diagnostic: reproduce the prelim's Pop@10 = 0.00003
--no-cv                  Skip 5-fold CV (fast dev-loop)
--no-tuning              Skip hyperparameter tuning
--cold-start-threshold N Warm/cold split threshold (default 10)
--hybrid-grid-step S     Simplex step for hybrid weights (default 0.2)
--decay-rate R           Exponential time-decay rate on training clicks
--persist-models         Save fitted recommenders + serving context to
                         data/processed/models/ for the FastAPI service.
```

The pipeline writes 11 CSVs under `evaluation/` — `baseline_results`, `metrics_with_ci`, `popularity_bias`, `fairness_audit`, `hybrid_ablation`, `paired_t_tests` (with Bonferroni column), `cold_start_results`, `tuning_{svd,als,hybrid}`, `cv_results`. Each one is cited directly from Chapter 5. The supplementary scripts add `reranker_results.csv`, `reranker_importance.csv`, `robustness.csv`, and `robustness_summary.csv`; `scripts/train_reranker.py` additionally writes the trained LightGBM booster to `data/processed/models/reranker.joblib`.

### One-shot supplementary scripts

```
python scripts/train_reranker.py            # LambdaMART reranker + Table 5.5
python scripts/robustness_check.py          # 3-seed sweep + robustness_summary.csv
python scripts/generate_design_figures.py   # Figs 3.1–3.3 (uses persisted Hybrid if present)
python scripts/generate_workplan_gantt.py   # Fig 3.4
```

## Run the tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite covers every recommender, all metrics (including outcome-weighted precision), the cold-start masks, hyperparameter grid enumeration, Bonferroni adjustment, the pipeline's course-scoping contract, model persistence round-trips, and the FastAPI endpoints.

## Build the report PDF

The final-report sources live in `Final-Report/`. To rebuild the PDF you
need [pandoc](https://pandoc.org/installing.html), a LaTeX engine (xelatex),
and [pandoc-crossref](https://github.com/lierdakil/pandoc-crossref). On macOS:

```bash
brew install pandoc basictex pandoc-crossref
sudo installer -pkg /usr/local/Caskroom/basictex/*/mactex-basictex-*.pkg -target /
eval "$(/usr/libexec/path_helper)"
```

Then:

```bash
cd Final-Report
./build.sh
```

This concatenates the title page, six chapters, and reference list into `final-report.pdf`. Design figures are generated by `scripts/generate_design_figures.py`, the workplan Gantt by `scripts/generate_workplan_gantt.py`, the dashboard figures (4.1, 4.2a/b, 4.3) are captured Streamlit screenshots checked into `Final-Report/figures/`, and EDA figures come from the notebook.

The previous draft-stage PDF and prior submission are preserved in `Preliminary-Report/` for historical reference. A 3–5 min demo video plan is in `Final-Report/final-video-plan.md`.

## Layout

```
src/                # production code
├── oulad.py         # loader
├── preprocess.py    # temporal split + course-scoping + time-decay
├── features.py      # item feature extractor (train-only)
├── baselines.py     # Random, Popularity
├── svd.py           # truncated-SVD CF
├── als.py           # implicit-feedback ALS
├── content.py       # cosine-similarity content-based
├── hybrid.py        # weighted CF + content + outcome ensemble
├── gated_hybrid.py  # per-user switching (cold/warm) hybrid
├── lambdamart.py    # two-stage LambdaMART reranker (Stage 2)
├── reranker_features.py  # per-(user, item) features for Stage 2
├── metrics.py       # ranking metrics + outcome-weighted precision
├── evaluation.py    # CV folds, bootstrap CIs, paired-t, ablation, fairness
├── tuning.py        # grid searches on a temporal sub-split
├── pipeline.py      # end-to-end runner producing evaluation/*.csv
├── persistence.py   # joblib artefacts + serving context + manifest
└── api.py           # FastAPI service (/health, /recommend, /decompose)
tests/              # 116 unit tests for src/
dashboard/          # Streamlit dashboard on top of the API
notebooks/          # exploratory data analysis
scripts/            # one-shot reproducibility scripts
data/               # not committed; raw + processed datasets land here
evaluation/         # results CSVs from the pipeline
Final-Report/       # final report sources, figures, build script, PDF, video plan
Preliminary-Report/ # historical: prior submission sources and PDF
Proposal/           # proposal artefacts (video plan, narration, slides, MP4)
Resources/          # marker feedback, transcripts, guidelines (read-only)
Dockerfile          # two-stage image for the FastAPI service
```

## License

Code: MIT. OULAD data: see CC-BY licence of the original dataset.
