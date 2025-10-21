"""Aligner module for pairwise sequence alignment."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass

from Bio.Align import PairwiseAligner, substitution_matrices

from versalign.scoring import T

# Define __all__ for explicit export of PairwiseAligner class
__all__ = ["PairwiseAligner", "substitution_matrices"]


@dataclass
class Aligner:
    """Dataclass to hold a PairwiseAligner and its label function."""

    aligner: PairwiseAligner
    label_fn: Callable[[T], Hashable] | None = None


def setup_aligner(
    substitution_matrix: substitution_matrices.Array,
    mode: str = "global",
    target_internal_open_gap_score: float = -1.0,
    target_internal_extend_gap_score: float = -1.0,
    target_left_open_gap_score: float = -1.0,
    target_left_extend_gap_score: float = -1.0,
    target_right_open_gap_score: float = -1.0,
    target_right_extend_gap_score: float = -1.0,
    query_internal_open_gap_score: float = -1.0,
    query_internal_extend_gap_score: float = -1.0,
    query_left_open_gap_score: float = -1.0,
    query_left_extend_gap_score: float = -1.0,
    query_right_open_gap_score: float = -1.0,
    query_right_extend_gap_score: float = -1.0,
    label_fn: Callable[[T], Hashable] | None = None,
) -> Aligner:
    """
    Setup a PairwiseAligner with the given parameters.

    :param substitution_matrix: substitution matrix to be used for alignment
    :param mode: alignment mode ("local" or "global"), defaults to "global"
    :param target_internal_open_gap_score: internal open gap score for target sequence (default: -1.0)
    :param target_internal_extend_gap_score: internal extend gap score for target sequence (default: -1.0)
    :param target_left_open_gap_score: left open gap score for target sequence (default: -1.0)
    :param target_left_extend_gap_score: left extend gap score for target sequence (default: -1.0)
    :param target_right_open_gap_score: right open gap score for target sequence (default: -1.0)
    :param target_right_extend_gap_score: right extend gap score for target sequence (default: -1.0)
    :param query_internal_open_gap_score: internal open gap score for query sequence (default: -1.0)
    :param query_internal_extend_gap_score: internal extend gap score for query sequence (default: -1.0)
    :param query_left_open_gap_score: left open gap score for query sequence (default: -1.0)
    :param query_left_extend_gap_score: left extend gap score for query sequence (default: -1.0)
    :param query_right_open_gap_score: right open gap score for query sequence (default: -1.0)
    :param query_right_extend_gap_score: right extend gap score for query sequence (default: -1.0)
    :param label_fn: optional function to label item in sequences (default: None)
    :return: PairwiseAligner object
    :raises ValueError: if mode is not "global" or "local"
    :raises ValueError: if substitution matrix is not provided
    """
    if mode not in ["global", "local"]:
        raise ValueError("mode must be one of 'global' or 'local'")

    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.target_internal_open_gap_score = target_internal_open_gap_score
    aligner.target_internal_extend_gap_score = target_internal_extend_gap_score
    aligner.target_left_open_gap_score = target_left_open_gap_score
    aligner.target_left_extend_gap_score = target_left_extend_gap_score
    aligner.target_right_open_gap_score = target_right_open_gap_score
    aligner.target_right_extend_gap_score = target_right_extend_gap_score
    aligner.query_internal_open_gap_score = query_internal_open_gap_score
    aligner.query_internal_extend_gap_score = query_internal_extend_gap_score
    aligner.query_left_open_gap_score = query_left_open_gap_score
    aligner.query_left_extend_gap_score = query_left_extend_gap_score
    aligner.query_right_open_gap_score = query_right_open_gap_score
    aligner.query_right_extend_gap_score = query_right_extend_gap_score
    aligner.wildcard = None
    aligner.substitution_matrix = substitution_matrix

    return Aligner(aligner=aligner, label_fn=label_fn)
