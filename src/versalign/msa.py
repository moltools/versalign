"""Multiple Sequence Alignment (MSA) module."""

import logging
from collections.abc import Callable, Hashable

import numpy as np

from versalign.aligner import Aligner, PairwiseAligner
from versalign.config import DEFAULT_GAP_REPR, LOGGER_NAME
from versalign.helpers import arr_to_seq, seq_to_arr
from versalign.pairwise import pairwise_alignment, pairwise_alignment_score
from versalign.scoring import T


def calc_msa(
    aligner: Aligner,
    seqs: list[list[str]],
    gap_repr: str = DEFAULT_GAP_REPR,
    center_star: int | None = None,
) -> tuple[list[list[str]], list[int]]:
    """
    Perform multiple sequence alignment using center star method.

    :param aligner: Aligner object
    :param seqs: list of sequences to be aligned
    :param gap_repr: gap representation, defaults to DEFAULT_GAP_REPR. Make sure
        gap repr is in alphabet of substitution matrix
    :param center_star: index of the center star sequence, defaults to None
    :return: multiple sequence alignment and order of input sequences in alignment
    :raises ValueError: if sequence list is empty
    """
    logger = logging.getLogger(LOGGER_NAME)

    if not seqs:
        raise ValueError("the sequence list cannot be empty")

    aligner_obj: PairwiseAligner = aligner.aligner
    label_fn: Callable[[T], Hashable] | None = aligner.label_fn

    # Set aligner mode to global
    current_mode = aligner_obj.mode
    if current_mode != "global":
        logger.info(f"Overriding aligner mode from {current_mode} to 'global' for MSA")
    aligner_obj.mode = "global"

    # Get int repr of gap
    int_gap_repr = aligner_obj.substitution_matrix.names.index(gap_repr)

    # Convert sequences into int arrays based on substitution matrix alphabet
    int_seqs = []
    for seq in seqs:
        int_seqs.append(seq_to_arr(seq, aligner_obj.substitution_matrix.names, label_fn))

    # Create pairwise similarity matrix
    sims = np.zeros((len(int_seqs), len(int_seqs)), dtype=float)
    for i, int_seq1 in enumerate(int_seqs):
        for j, int_seq2 in enumerate(int_seqs):
            # We don't register a score for i == j, because self-alignment shouldn't
            # be considered for determining center star
            if i >= j:
                continue  # only need to calcualte lower triangle

            score = pairwise_alignment_score(aligner_obj, int_seq1, int_seq2)
            sims[i, j] = score
            sims[j, i] = score

    # Find center star sequence, i.e., the sequence with the highest absolute similarity
    if center_star is None:
        center_i = int(np.argmax(np.sum(sims, axis=1)))
    else:
        center_i = center_star

    # Mask out self‑score so we can sort without special slicing
    sims[center_i, center_i] = -np.inf

    # Sort all by descending similarity, drop the center itself
    all_idx = sims[center_i].argsort()[::-1]
    others = [i for i in all_idx if i != center_i]

    # Align sequences to center star sequence
    msa = np.array([int_seqs[center_i]], dtype=np.int32)
    for item in others:
        # Align sequence to center star sequence
        target = msa[0]
        query = int_seqs[item]
        insert_repr = np.int32(-1)
        _, t_a, q_a = pairwise_alignment(aligner_obj, target, query, gap_repr=insert_repr)

        # Insert gaps in remainder sequences, if any
        remainder = msa[1:]
        if remainder.shape[0] > 0:
            insert_indices = [i for i, x in enumerate(t_a) if x == insert_repr]
            for i in insert_indices:
                remainder = np.insert(remainder, i, insert_repr, axis=1)
            msa = np.vstack([t_a, remainder, q_a])
        else:
            msa = np.vstack([t_a, q_a])

        # Rename insert_repr to gap_repr
        msa[msa == insert_repr] = int_gap_repr

    # Check if there are any gap-only columns, delete those if any
    gap_columns = np.where(np.all(msa == int_gap_repr, axis=0))[0]
    if gap_columns.size > 0:
        msa = np.delete(msa, gap_columns, axis=1)

    # Convert msa to list of list of strings
    str_msa = [arr_to_seq(row, aligner_obj.substitution_matrix.names) for row in msa]

    final_order = [center_i] + others

    return str_msa, final_order
