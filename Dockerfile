# Multi-stage image for the OULAD educational-content recommender.
#
# Stage 1 (`build`): installs Python deps into a virtualenv so the final
# image doesn't carry apt/build tooling. This is the piece that mitigates
# the §3.8 risk of `implicit` (Cython) install failures on a marker's
# machine — a Debian slim base with build-essential + OpenBLAS resolves
# all its transitive C/C++ requirements deterministically.
#
# Stage 2 (`runtime`): copies the venv + project source, then serves the
# FastAPI app on port 8000 via uvicorn. Persisted model artefacts and
# raw OULAD CSVs are expected to be mounted at /app/data at runtime; the
# image does not bundle either (the CSVs are CC-BY but ~500MB, and the
# .joblib artefacts change with every retrain).
#
# Build:
#     docker build -t recsys:latest .
#
# Run (with local data + models mounted):
#     docker run --rm -p 8000:8000 \
#         -v $(pwd)/data:/app/data \
#         recsys:latest
#
# Then hit http://localhost:8000/health.

# --------------------------------------------------------------------------- #
# Stage 1 — build the virtualenv with all Python deps.
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build tooling + BLAS/LAPACK headers so `implicit` (Cython + OpenMP)
# and scipy compile cleanly. Removed in the runtime stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libopenblas-dev \
        liblapack-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# --------------------------------------------------------------------------- #
# Stage 2 — minimal runtime image.
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    RECSYS_MODELS_DIR="/app/data/processed/models"

# Runtime libs only (no build tools).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY src ./src
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY tests ./tests
COPY requirements.txt README.md LICENSE ./

EXPOSE 8000

# Health-check hits `/health` — reports uninitialised if models aren't
# mounted at RECSYS_MODELS_DIR, ready once they are.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health', timeout=3); exit(0 if r.status_code == 200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
