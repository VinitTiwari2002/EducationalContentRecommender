"""Persist and reload fitted recommenders + serving artefacts.

The pipeline can spend several minutes fitting SVD, ALS, and the hybrid;
the FastAPI service and Streamlit dashboard need those artefacts at
startup without re-running the pipeline. This module handles both sides.

What we persist:
    * `models.joblib`        — dict[name, recommender], covering all seven
                                registered recommenders.
    * `serving_context.joblib` — student_index, item_index, per-user
                                candidate arrays, and the outcome vector
                                needed by the /decompose endpoint.
    * `manifest.json`        — cutoff date, tuning hyperparameters, git
                                commit (if available), and a timestamp,
                                so a marker can verify what they loaded.

Joblib is chosen over pickle because it compresses numpy arrays well and
its `mmap_mode` option lets very large factor matrices be memory-mapped
at load time instead of copied into RAM.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "data" / "processed" / "models"


@dataclass
class ServingContext:
    """Everything a serving process needs beyond the model objects.

    Kept separate from the models so that swapping in a new-cutoff split
    doesn't force re-serialising every recommender.
    """

    student_index: np.ndarray  # row → id_student
    item_index: np.ndarray  # col → id_site
    user_candidates: list[np.ndarray]  # course-scoped candidate pool per row
    outcome_score: np.ndarray  # (n_items,), for /decompose baseline
    cutoff_date: int


def _git_commit() -> str | None:
    """Best-effort git commit hash for the manifest; None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def save_models(
    models: dict[str, Any],
    context: ServingContext,
    metadata: dict[str, Any] | None = None,
    out_dir: Path | str = DEFAULT_MODELS_DIR,
) -> Path:
    """Persist fitted recommenders + serving context to `out_dir`.

    Returns the directory the artefacts were written to. Overwrites any
    existing artefacts in the same directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, out_dir / "models.joblib", compress=3)
    joblib.dump(context, out_dir / "serving_context.joblib", compress=3)

    manifest = {
        "n_models": len(models),
        "model_names": sorted(models.keys()),
        "n_students": int(context.student_index.shape[0]),
        "n_items": int(context.item_index.shape[0]),
        "cutoff_date": int(context.cutoff_date),
        "git_commit": _git_commit(),
        "metadata": metadata or {},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out_dir


def load_models(
    in_dir: Path | str = DEFAULT_MODELS_DIR,
) -> tuple[dict[str, Any], ServingContext, dict[str, Any]]:
    """Reload fitted recommenders + serving context + manifest.

    Raises FileNotFoundError if any expected artefact is missing.
    """
    in_dir = Path(in_dir)
    for name in ("models.joblib", "serving_context.joblib", "manifest.json"):
        if not (in_dir / name).exists():
            raise FileNotFoundError(
                f"Missing artefact {name} under {in_dir}. Run the pipeline "
                f"with --persist-models to produce it."
            )
    models = joblib.load(in_dir / "models.joblib")
    context = joblib.load(in_dir / "serving_context.joblib")
    manifest = json.loads((in_dir / "manifest.json").read_text())
    return models, context, manifest


def artefacts_exist(in_dir: Path | str = DEFAULT_MODELS_DIR) -> bool:
    """Cheap check for whether a serving directory is ready."""
    in_dir = Path(in_dir)
    return all(
        (in_dir / name).exists()
        for name in ("models.joblib", "serving_context.joblib", "manifest.json")
    )
