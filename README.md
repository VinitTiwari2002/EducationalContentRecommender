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
# 2. Run the pipeline end-to-end (caches the split, evaluates baselines):
python -m src.pipeline

# 3. Open the EDA notebook
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
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

## Run the tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The 25 unit tests cover the metric implementations, both baselines, candidate-pool
restriction, and edge cases.

## Build the preliminary report PDF

The report sources live in `Preliminary-Report/`. To rebuild the PDF you need
[pandoc](https://pandoc.org/installing.html), a LaTeX engine (xelatex), and
[pandoc-crossref](https://github.com/lierdakil/pandoc-crossref) installed. On
macOS:

```bash
brew install pandoc basictex pandoc-crossref
sudo installer -pkg /usr/local/Caskroom/basictex/*/mactex-basictex-*.pkg -target /
eval "$(/usr/libexec/path_helper)"
```

Then:

```bash
cd Preliminary-Report
./build.sh
```

This concatenates the title page, four chapters, and reference list into
`prelim-report.pdf`. Design figures are generated separately by
`scripts/generate_design_figures.py`; EDA figures are produced by the notebook.

## Layout

```
src/                # production code (loader, preprocessing, metrics, baselines, pipeline)
tests/              # unit tests for src/
notebooks/          # exploratory data analysis
scripts/            # one-shot reproducibility scripts (e.g. design-figure generator)
data/               # not committed; raw + processed datasets land here
evaluation/         # results CSVs from the pipeline
Preliminary-Report/ # report chapter sources, figures, build script, and PDF
Proposal/           # proposal submission artefacts (video plan, narration, slides, MP4)
Resources/          # marker feedback, transcripts, guidelines (read-only)
```

## License

Code: MIT. OULAD data: see CC-BY licence of the original dataset.
