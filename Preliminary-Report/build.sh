#!/usr/bin/env bash
# Build the preliminary report PDF from the chapter markdown files.
#
# Prerequisites:
#   brew install pandoc basictex
#   eval "$(/usr/libexec/path_helper)"   # picks up tlmgr after basictex install
#
# Usage:
#   cd Preliminary-Report && ./build.sh

set -euo pipefail

cd "$(dirname "$0")"

OUT="prelim-report.pdf"

# Pandoc pulls the YAML metadata + section structure from the title page
# and concatenates the four chapter files plus the references file.
pandoc \
    00-title-page.md \
    chapters/01-introduction.md \
    chapters/02-literature-review.md \
    chapters/03-design.md \
    chapters/04-feature-prototype.md \
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
