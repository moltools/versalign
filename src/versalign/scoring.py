"""Scoring module for sequence alignment."""

from collections.abc import Callable, Hashable, Sequence
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from versalign.aligner import substitution_matrices

T = TypeVar("T")


def default_compare(a: Any, b: Any) -> float:
    """
    Default comparison function for scoring.

    :param a: first element to compare
    :param b: second element to compare
    :return: 1.0 if elements are equal, else 0.0
    """
    return 1.0 if a == b else 0.0


def ensure_hashable_unique(labels: Sequence[Hashable]) -> None:
    """
    Ensure that all labels are hashable and unique.

    :param labels: sequence of labels to check
    :raises TypeError: if any label is not hashable
    :raises ValueError: if any label is not unique
    """
    # Hashability check; will raise TypeError if not hashable
    for lbl in labels:
        hash(lbl)

    # Uniqueness check; will raise ValueError if not unique
    if len(set(labels)) != len(labels):
        dupes = [x for x in labels if labels.count(x) > 1]
        raise ValueError(f"Labels must be unique; duplicates found: {sorted(set(dupes))}")


def labels_from_objs(objs: Sequence[T], label_fn: Callable[[T], Hashable] | None) -> tuple[Hashable, ...]:
    """
    Generate labels from a sequence of objects.

    :param objs: sequence of objects to generate labels from
    :param label_fn: function to generate a label from an object; if None, use
        the object itself or its string representation
    :return: tuple of labels
    :raises TypeError: if any label is not hashable
    :raises ValueError: if any label is not unique
    """
    if label_fn is None:
        # Use objects themselves as labels if possible; fall back to str(...)
        raw = []
        for o in objs:
            try:
                hash(o)
                raw.append(o)
            except TypeError:
                raw.append(str(o))
        labels: tuple[Hashable, ...] = tuple(raw)
    else:
        labels = tuple(label_fn(o) for o in objs)

    # Validate labels
    ensure_hashable_unique(labels)

    return labels


def create_substitution_matrix(df: pd.DataFrame) -> tuple[substitution_matrices.Array, tuple[str, ...]]:
    """
    Parse a substitution matrix from a DataFrame.

    :param df: parsed data from the input file containing the substitution matrix
    :return: tuple of substitution matrix and alphabet
    :raises ValueError: if the substitution matrix is not square
    """
    # Check if DataFrame is square
    if df.shape[0] != df.shape[1]:
        raise ValueError("Substitution matrix must be square")

    alphabet = tuple(df.columns)
    data = df.to_numpy(np.float64)

    # Create substitution matrix
    sm = substitution_matrices.Array(alphabet, 2, data, np.float64)
    sm.names = alphabet

    return sm, alphabet


def create_substituion_matrix_dynamically(
    objs: Sequence[T],
    compare: Callable[[T, T], float] | None = None,
    *,
    label_fn: Callable[[T], Hashable] | None = None,
    dtype: type = float,
    symmetric: bool = True,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """
    Create a substitution matrix dynamically.

    :param objs: sequence of objects to create the substitution matrix from
    :param compare: function to compare two objects and return a score; if None,
        use the default comparison function
    :param label_fn: function to generate a label from an object; if None, use
        the object itself or its string representation
    :param dtype: data type for the substitution matrix
    :param symmetric: whether the substitution matrix is symmetric
    :return: substitution matrix as a DataFrame
    :raises ValueError: if the input sequence is empty
    """
    if not objs:
        raise ValueError("Cannot create substitution matrix from empty sequence")

    cmp_fn = compare or default_compare
    labels = labels_from_objs(objs, label_fn)
    n = len(objs)

    data = np.empty((n, n), dtype=dtype)

    # Fill in data
    if symmetric:
        for i in range(n):
            # Diagonal first; often faster to set explicitly
            data[i, i] = cmp_fn(objs[i], objs[i])
            for j in range(i + 1, n):
                s = cmp_fn(objs[i], objs[j])
                data[i, j] = s
                data[j, i] = s
    else:
        for i in range(n):
            for j in range(n):
                data[i, j] = cmp_fn(objs[i], objs[j])

    # Create substitution matrix
    df = pd.DataFrame(data, index=labels, columns=labels)

    return create_substitution_matrix(df)
