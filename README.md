# Data-Driven Personalised Educational Content Recommendation

**CM3070 Final Project — CM3005 Data Science template**
Vinit Tiwari (220174440)

A hybrid recommendation engine for the OULAD dataset, combining collaborative filtering, feature-based content filtering, and outcome-aware evaluation.

## Status

Preliminary report stage. Implemented:
- Data ingestion pipeline for all seven OULAD tables
- Sparse user-item interaction matrix with temporal 80/20 split
- Random and Popularity baseline recommenders
- Evaluation harness (Precision@K, Recall@K, NDCG@K, Hit Rate@K)
- EDA notebook

Planned: SVD collaborative filtering, content-based filtering, hybrid ensemble, FastAPI service, Streamlit dashboard.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
pip install -r requirements.txt

# 1. Download OULAD into data/raw/ (manual step; see below)
# 2. Run the pipeline end-to-end:
python -m src.pipeline

# 3. Open the EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

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

## Layout

```
src/                # production code
  oulad.py            # data loader
  preprocess.py       # interaction matrix + temporal split
  metrics.py          # evaluation framework
  baselines.py        # Random + Popularity recommenders
  pipeline.py         # end-to-end runner
notebooks/          # EDA + experiment notebooks
data/               # not committed; raw + processed datasets
evaluation/         # results CSVs + figures
tests/              # unit tests
Preliminary-Report/ # report chapters and assembly
```

## License

Code: MIT. OULAD data: see CC-BY licence of the original dataset.
