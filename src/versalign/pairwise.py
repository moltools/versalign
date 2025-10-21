"""Pairwise sequence alignment module."""

from collections.abc import Callable, Hashable

import numpy as np
from numpy.typing import NDArray

from versalign.aligner import Aligner, PairwiseAligner
from versalign.config import DEFAULT_GAP_REPR
from versalign.helpers import arr_to_seq, seq_to_arr
from versalign.scoring import T


def pairwise_alignment_score(
    aligner: PairwiseAligner,
    target: NDArray[np.int32],
    query: NDArray[np.int32],
) -> float:
    """
    Align two sequences and return the alignment score.

    :param aligner: PairwiseAligner object
    :param target: target sequence
    :param query: query sequence
    :return: alignment score
    """
    return aligner.score(target, query)


def pairwise_alignment(
    aligner: PairwiseAligner,
    target: NDArray[np.int32],
    query: NDArray[np.int32],
    gap_repr: np.int32 | None = None,
) -> tuple[float, NDArray[np.int32], NDArray[np.int32]]:
    """
    Align two sequences and return the alignment score.

    :param aligner: PairwiseAligner object
    :param target: target sequence
    :param query: query sequence
    :param gap_repr: integer representation of gap, defaults to 0
    :return: alignment score or alignment as a tuple of two arrays
    :raises ValueError: if cigar string has unexpected format
    """
    if gap_repr is None:
        gap_repr = np.int32(0)

    alignments = aligner.align(seqA=target, seqB=query)

    # Pick first alignment
    alignment = alignments[0]
    score = alignments[0].score

    t_a: list[np.int32] = []
    q_a: list[np.int32] = []
    for i in range(alignment.coordinates.shape[1] - 1):
        a = alignment.coordinates[0][i : i + 2]
        b = alignment.coordinates[1][i : i + 2]
        len_a = a[1] - a[0]
        len_b = b[1] - b[0]
        if len_a == len_b:
            t_a.extend(target[a[0] : a[1]])
            q_a.extend(query[b[0] : b[1]])
        elif len_a == 0:
            t_a.extend([gap_repr] * len_b)
            q_a.extend(query[b[0] : b[1]])
        elif len_b == 0:
            t_a.extend(target[a[0] : a[1]])
            q_a.extend([gap_repr] * len_a)
        else:
            raise ValueError("unexpected alignment")

    return (
        score,
        np.array(t_a, dtype=np.int32),
        np.array(q_a, dtype=np.int32),
    )


def calc_pairwise_alignment(
    aligner: Aligner,
    target: list[str],
    query: list[str],
    gap_repr: str = DEFAULT_GAP_REPR,
) -> tuple[float, list[str], list[str]]:
    """
    Align two sequences and return the alignment score or alignment.

    :param aligner: Aligner object
    :param target: target sequence
    :param query: query sequence
    :param gap_repr: gap representation, defaults to DEFAULT_GAP_REPR. Make sure
        gap repr is in alphabet of substitution matrix
    :return: alignment score or alignment as a tuple of two sequences
    """
    aligner_obj: PairwiseAligner = aligner.aligner
    label_fn: Callable[[T], Hashable] | None = aligner.label_fn

    # Get int repr of gap; check if is in substitution matrix names
    if gap_repr not in aligner_obj.substitution_matrix.names:
        raise ValueError("'gap_repr' must be in substitution matrix alphabet")
    int_gap_repr = aligner_obj.substitution_matrix.names.index(gap_repr)

    # Convert sequences into int arrays based on substitution matrix alphabet
    int_target = seq_to_arr(target, aligner_obj.substitution_matrix.names, label_fn)
    int_query = seq_to_arr(query, aligner_obj.substitution_matrix.names, label_fn)

    score, t_a, q_a = pairwise_alignment(aligner_obj, int_target, int_query, gap_repr=np.int32(int_gap_repr))

    # Convert aligned sequences back to list of strings
    str_t_a = arr_to_seq(t_a, aligner_obj.substitution_matrix.names)
    str_q_a = arr_to_seq(q_a, aligner_obj.substitution_matrix.names)

    return score, str_t_a, str_q_a
