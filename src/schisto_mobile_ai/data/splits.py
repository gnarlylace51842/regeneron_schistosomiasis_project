"""Split helpers that let the caller choose label and grouping columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


def assign_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    label_column: str | None = None,
    group_column: str | None = None,
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Assign fold IDs without hardcoding any dataset-specific columns."""
    if len(frame) < n_splits:
        raise ValueError("Number of rows must be at least n_splits.")

    output = frame.copy()
    output[fold_column] = -1

    index_array = np.arange(len(output))
    targets = output[label_column] if label_column else None
    groups = output[group_column] if group_column else None

    if group_column and label_column:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    elif group_column:
        splitter = GroupKFold(n_splits=n_splits)
    elif label_column:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_id, (_, valid_index) in enumerate(splitter.split(index_array, targets, groups)):
        row_index = output.index[valid_index]
        output.loc[row_index, fold_column] = fold_id

    return output

