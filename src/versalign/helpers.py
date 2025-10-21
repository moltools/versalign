"""Helpers for sequence alignment."""

from collections.abc import Callable, Hashable

import numpy as np
from numpy.typing import NDArray

from versalign.scoring import T


def seq_to_arr(
    seq: list[str],
    alphabet: list[str],
    label_fn: Callable[[T], Hashable] | None = None,
) -> NDArray[np.int32]:
    """
    Convert a sequence of strings to an array of integers based on the alphabet.

    :param seq: sequence of strings
    :param alphabet: list of strings representing the alphabet
    :param label_fn: optional function to map sequence items to labels
    :return: array of integers
    :raises ValueError: if a item in the sequence is not in the alphabet
    """
    arr = []
    for item in seq:
        try:
            if not label_fn:
                arr.append(alphabet.index(item))
            else:
                labeled_item = label_fn(item)
                arr.append(alphabet.index(labeled_item))
        except ValueError as e:
            raise ValueError(f"Item '{item}' not found in alphabet") from e
    return np.array(arr, dtype=np.int32)


def arr_to_seq(arr: NDArray[np.int32], alphabet: list[str]) -> list[str]:
    """
    Convert an array of integers to a sequence of strings based on the alphabet.

    :param arr: array of integers
    :param alphabet: list of strings representing the alphabet
    :return: sequence of strings
    :raises ValueError: if an integer in the array is not in the alphabet
    """
    return [alphabet[i] for i in arr.tolist() if i < len(alphabet)]
