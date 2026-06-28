"""Preprocessing: sparse interaction matrix and temporal train/test split.

Builds a sparse CSR matrix of summed clicks per (student, vle_site) and splits
it chronologically (no leakage) by `date` from studentVle.

Course-scoping. Each id_site belongs to exactly one (code_module,
code_presentation) — the OULAD catalogue is partitioned by course presentation
rather than shared across them, and on average a student is enrolled in only
~1.1 presentations. Recommending an item from a presentation a student is not
enrolled in is therefore nonsensical regardless of metric. The split therefore
records, for every user, the set of item-column indices belonging to that
user's enrolled presentations; recommenders restrict candidates to that set.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .oulad import OULAD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class Split:
    """Train and test interaction matrices with aligned index spaces.

    student_index maps a row index in the matrices back to id_student;
    item_index maps a column index back to id_site. Both matrices share
    the same row/column space so a recommender trained on `train` can be
    scored against `test` row-by-row.

    user_candidates[row] is the sorted array of item-column indices that
    belong to the presentation(s) the user is enrolled in. Recommenders
    use this to restrict candidate items.
    """

    train: sparse.csr_matrix
    test: sparse.csr_matrix
    student_index: np.ndarray
    item_index: np.ndarray
    cutoff_date: int
    user_candidates: list[np.ndarray]

    def save(self, out_dir: Path | str = DEFAULT_PROCESSED_DIR) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(out_dir / "train.npz", self.train)
        sparse.save_npz(out_dir / "test.npz", self.test)
        np.save(out_dir / "student_index.npy", self.student_index)
        np.save(out_dir / "item_index.npy", self.item_index)
        (out_dir / "cutoff_date.txt").write_text(str(self.cutoff_date))
        # Save as object array of variable-length arrays
        np.save(
            out_dir / "user_candidates.npy",
            np.array(self.user_candidates, dtype=object),
            allow_pickle=True,
        )

    @classmethod
    def load(cls, in_dir: Path | str = DEFAULT_PROCESSED_DIR) -> "Split":
        in_dir = Path(in_dir)
        return cls(
            train=sparse.load_npz(in_dir / "train.npz"),
            test=sparse.load_npz(in_dir / "test.npz"),
            student_index=np.load(in_dir / "student_index.npy"),
            item_index=np.load(in_dir / "item_index.npy"),
            cutoff_date=int((in_dir / "cutoff_date.txt").read_text()),
            user_candidates=list(
                np.load(in_dir / "user_candidates.npy", allow_pickle=True)
            ),
        )


def build_split(data: OULAD, test_fraction: float = 0.2) -> Split:
    """Build a temporal train/test split with course-scoped user candidate sets.

    Aggregates clicks per (id_student, id_site) within train/test windows.
    Cutoff is the chronological quantile of `date` such that approximately
    `test_fraction` of interactions fall in the test window.

    The candidate set for each user is derived from `vle` filtered to the
    presentations that user appears in (per `studentRegistration`).
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")

    vle_interactions = data.student_vle[
        ["id_student", "id_site", "date", "sum_click"]
    ].copy()
    cutoff = int(vle_interactions["date"].quantile(1 - test_fraction))

    students = np.sort(vle_interactions["id_student"].unique())
    items = np.sort(vle_interactions["id_site"].unique())
    s_to_row = pd.Series(np.arange(len(students)), index=students)
    i_to_col = pd.Series(np.arange(len(items)), index=items)

    def _matrix(df: pd.DataFrame) -> sparse.csr_matrix:
        agg = df.groupby(["id_student", "id_site"], as_index=False)["sum_click"].sum()
        rows = s_to_row.loc[agg["id_student"]].to_numpy()
        cols = i_to_col.loc[agg["id_site"]].to_numpy()
        vals = agg["sum_click"].to_numpy(dtype=np.float32)
        return sparse.coo_matrix(
            (vals, (rows, cols)), shape=(len(students), len(items))
        ).tocsr()

    train_df = vle_interactions[vle_interactions["date"] <= cutoff]
    test_df = vle_interactions[vle_interactions["date"] > cutoff]

    user_candidates = _build_user_candidates(data, students, items, i_to_col)

    return Split(
        train=_matrix(train_df),
        test=_matrix(test_df),
        student_index=students,
        item_index=items,
        cutoff_date=cutoff,
        user_candidates=user_candidates,
    )


def _build_user_candidates(
    data: OULAD,
    students: np.ndarray,
    items: np.ndarray,
    i_to_col: pd.Series,
) -> list[np.ndarray]:
    """For each student, compute the set of item-column indices that belong
    to a presentation the student is enrolled in.

    Enrolment is derived from studentRegistration (the authoritative source)
    intersected with studentVle (so users without any interaction are
    excluded — they are not in the matrix anyway).
    """
    vle_items = data.vle[["id_site", "code_module", "code_presentation"]]
    item_presentation = (
        vle_items.assign(pres=lambda d: d["code_module"] + "_" + d["code_presentation"])
        .set_index("id_site")["pres"]
    )

    # Restrict to items that appear in studentVle (i.e. are in our `items` array)
    item_presentation = item_presentation.loc[
        item_presentation.index.intersection(items)
    ]
    # Group item-column indices by presentation
    cols_for_item = i_to_col.loc[item_presentation.index].to_numpy()
    by_presentation: dict[str, list[int]] = {}
    for pres, col in zip(item_presentation.to_numpy(), cols_for_item):
        by_presentation.setdefault(pres, []).append(int(col))
    by_presentation_arr = {
        pres: np.sort(np.array(cols, dtype=np.int64))
        for pres, cols in by_presentation.items()
    }

    # For each student, find their presentation(s)
    reg = data.student_registration[
        ["id_student", "code_module", "code_presentation"]
    ].assign(pres=lambda d: d["code_module"] + "_" + d["code_presentation"])
    student_to_pres = reg.groupby("id_student")["pres"].apply(list).to_dict()

    candidates: list[np.ndarray] = []
    empty = np.empty(0, dtype=np.int64)
    for student in students:
        pres_list = student_to_pres.get(int(student), [])
        cols_arrays = [by_presentation_arr[p] for p in pres_list if p in by_presentation_arr]
        if not cols_arrays:
            candidates.append(empty)
        elif len(cols_arrays) == 1:
            candidates.append(cols_arrays[0])
        else:
            candidates.append(np.unique(np.concatenate(cols_arrays)))
    return candidates
