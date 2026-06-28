"""OULAD data loader.

Loads the seven OULAD CSVs from data/raw/ and exposes them as a single
container. No business logic — just IO, dtype enforcement, and presence checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OULAD_TABLES = (
    "assessments",
    "courses",
    "studentAssessment",
    "studentInfo",
    "studentRegistration",
    "studentVle",
    "vle",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class OULAD:
    assessments: pd.DataFrame
    courses: pd.DataFrame
    student_assessment: pd.DataFrame
    student_info: pd.DataFrame
    student_registration: pd.DataFrame
    student_vle: pd.DataFrame
    vle: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = [
            ("assessments", self.assessments),
            ("courses", self.courses),
            ("studentAssessment", self.student_assessment),
            ("studentInfo", self.student_info),
            ("studentRegistration", self.student_registration),
            ("studentVle", self.student_vle),
            ("vle", self.vle),
        ]
        return pd.DataFrame(
            [(name, len(df), df.shape[1], df.isna().sum().sum()) for name, df in rows],
            columns=["table", "rows", "columns", "total_nulls"],
        )


def _read(raw_dir: Path, name: str) -> pd.DataFrame:
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing OULAD file: {path}. Download the dataset and place all "
            f"seven CSVs under {raw_dir}/. See README.md."
        )
    return pd.read_csv(path, na_values=["?"])


def load(raw_dir: Path | str | None = None) -> OULAD:
    """Load all seven OULAD tables from raw_dir (default: data/raw/).

    OULAD encodes missing values as '?' (notably studentAssessment.score for
    un-marked assessments and studentInfo.imd_band for unknown bands); these
    are converted to NaN here so numeric columns stay numeric.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    return OULAD(
        assessments=_read(raw_dir, "assessments"),
        courses=_read(raw_dir, "courses"),
        student_assessment=_read(raw_dir, "studentAssessment"),
        student_info=_read(raw_dir, "studentInfo"),
        student_registration=_read(raw_dir, "studentRegistration"),
        student_vle=_read(raw_dir, "studentVle"),
        vle=_read(raw_dir, "vle"),
    )
