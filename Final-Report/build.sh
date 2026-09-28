#!/usr/bin/env bash
# Build the final report PDF from the chapter markdown files.
#
# Prerequisites (macOS):
#   brew install pandoc basictex pandoc-crossref
#   eval "$(/usr/libexec/path_helper)"   # picks up tlmgr after basictex install
#
# Usage:
#   cd Final-Report && ./build.sh

set -euo pipefail

cd "$(dirname "$0")"

OUT="final-report.pdf"

# Pandoc pulls YAML metadata + section structure from the title page,
# concatenates the six chapters, and appends the reference list.
pandoc \
    00-title-page.md \
    chapters/01-introduction.md \
    chapters/02-literature-review.md \
    chapters/03-design.md \
    chapters/04-implementation.md \
    chapters/05-evaluation.md \
    chapters/06-conclusion.md \
    99-references.md \
    -o "$OUT" \
    --pdf-engine=xelatex \
    --filter pandoc-crossref \
    --metadata secPrefix=Chapter \
    --metadata link-citations=true \
    --resource-path=.:figures \
    --variable papersize=a4 \
    --variable colorlinks=true

echo "Wrote $OUT"
ls -la "$OUT"
