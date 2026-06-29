# Preliminary Report — CM3070 Final Project

**Student:** Vinit Tiwari (220174440)
**Template:** CM3005 Data Science — Project Idea 1
**Project:** Data-Driven Personalised Educational Content Recommendation

## Submission Components
1. **PDF report** assembled from chapter drafts in `chapters/` — max 6000 words total
   - Chapter 1: Introduction (max 1000 w)
   - Chapter 2: Literature Review (max 2500 w)
   - Chapter 3: Design (max 2000 w)
   - Chapter 4: Feature Prototype (max 1500 w)
2. **3–5 min MP4 video** demonstrating the prototype (recorded after prototype runs)

## Sources
- `../Resources/ProjectProposalFeedback.txt` — marker feedback to address
- `../Resources/PreliminaryReportGuidelines.txt` — submission spec

## Building the PDF
From this directory, run `./build.sh`. The script invokes pandoc with xelatex
and pandoc-crossref, concatenates the title page, four chapters, and the
references file, and writes `prelim-report.pdf`. See the top-level `README.md`
for installation prerequisites.

## Figures
- `figures/fig_eda_*.png` — produced by `../notebooks/01_exploratory_data_analysis.ipynb`
- `figures/fig_3_*.png` — produced by `../scripts/generate_design_figures.py`
