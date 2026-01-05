"""Module for docking-related functionalities in versalign."""

from dataclasses import dataclass
from typing import Sequence, Literal

import numpy as np
from numpy.typing import NDArray

from versalign.aligner import Aligner, PairwiseAligner
from versalign.config import DEFAULT_GAP_REPR
from versalign.helpers import seq_to_arr, arr_to_seq
from versalign.blocking import get_symbol_score_lookup
from versalign.pairwise import pairwise_alignment, pairwise_alignment_score


DockStrategy = Literal["nonoverlap"]


@dataclass(frozen=True)
class DockPlacement:
    """
    Represents a single placement of a block within the target sequence.

    :var block_idx: index of the block in the candidates list
    :var reversed: whether the block is reversed in this placement
    :var start: start index of the placement in the target sequence
    :var end: end index of the placement in the target sequence
    :var score: alignment score of this placement
    :var center_aln: aligned target sequence segment
    :var block_aln: aligned block sequence segment
    """

    block_idx: int
    reversed: bool
    start: int
    end: int
    score: float
    center_aln: list[str]
    block_aln: list[str]


@dataclass(frozen=True)
class DockingResult:
    """
    Represents the result of docking multiple blocks against a target sequence.

    :var placements: chosen placements, reordered by center coordinate
    :var unused_blocks: per-block indices that were not used
    :var docked_row: center-anchored row for visualization
    :var total_score: total score of all placements
    """

    placements: list[DockPlacement]
    unused_blocks: list[int]
    docked_row: list[str]
    total_score: float


def _alignment_ctx(aligner: Aligner, gap_repr: str) -> tuple[PairwiseAligner, list[str], int, callable]:
    """
    Prepare alignment context for docking.

    :var aligner: Aligner object    
    :var gap_repr: representation of gaps in sequences
    :return: tuple containing the underlying PairwiseAligner, alphabet list, gap index,
    """
    aligner_obj: PairwiseAligner = aligner.aligner
    alphabet = list(aligner_obj.substitution_matrix.names)
    gap_idx = alphabet.index(gap_repr)
    label_fn = aligner.label_fn
    return aligner_obj, alphabet, gap_idx, label_fn


def _extract_center_interval(center_aln: list[str], block_aln: list[str], gap_repr: str) -> tuple[int, int] | None:
    """
    Extract the start and end indices of the block alignment within the center alignment.

    :var center_aln: aligned center sequence segment
    :var block_aln: aligned block sequence segment
    :var gap_repr: representation of gaps in sequences
    :return: tuple of (start, end) indices in the center sequence, or None if no coverage
    """
    # Map candidate columns to target positions
    center_pos = -1
    covered: list[int] = []
    for c_tok, b_tok in zip(center_aln, block_aln):
        if c_tok != gap_repr:
            center_pos += 1
            if b_tok != gap_repr:
                covered.append(center_pos)
        else:
            # Insertion relative to center -> no center coordinate
            pass

    if not covered:
        return None
    
    return min(covered), max(covered)


def _make_center_anchored_row(
    center: list[str],
    placements: list[DockPlacement],
    gap_repr: str,
) -> list[str]:
    """
    Create a center-anchored row representing the docked blocks.

    :var center: the target sequence
    :var placements: list of DockPlacement objects
    :var gap_repr: representation of gaps in sequences
    :return: list of symbols representing the docked row
    """
    # Simple mapping view: one symbol per center position; no insertion
    row = [gap_repr] * len(center)

    # If multiple placements cover the same center position, keep the best-scoring placement's symbol
    for p in placements:
        center_pos = -1
        for c_tok, b_tok in zip(p.center_aln, p.block_aln):
            if c_tok != gap_repr:
                center_pos += 1
                if b_tok != gap_repr:
                    row[center_pos] = b_tok
    
    return row


def score_docked_region(
    center_aln: list[str],
    block_aln: list[str],
    start: int,
    end: int,
    gap_repr: str,
    idx: dict[str, int],
    mat: np.ndarray,
    include_insertions: bool = False,
) -> float:
    """
    Score only the part of an alignment that maps to target positions [start, end] (inclusive).

    :param center_aln: aligned center sequence segment
    :param block_aln: aligned block sequence segment
    :param start: start index of the region in the target sequence
    :param end: end index of the region in the target sequence
    :param gap_repr: representation of gaps in sequences
    :param idx: mapping from token to index in substitution matrix
    :param mat: substitution matrix as a 2D numpy array
    :param include_insertions: whether to include insertions relative to target in scoring
    :return: score of the specified region
    """
    if len(center_aln) != len(block_aln):
        raise ValueError("alignment length mismatch")

    total = 0.0
    target_pos = -1
    in_region = False

    for c_tok, b_tok in zip(center_aln, block_aln):
        if c_tok != gap_repr:
            target_pos += 1
            in_region = (start <= target_pos <= end)
            if in_region:
                total += mat[idx[c_tok], idx[b_tok]]
        else:
            # Insertion relative to target: no target coordinate
            if include_insertions and in_region:
                total += mat[idx[c_tok], idx[b_tok]]

        if target_pos > end:
            break

    return total


def dock_against_target(
    aligner: Aligner,
    target: Sequence[str],
    candidates: Sequence[Sequence[str]],
    gap_repr: str = DEFAULT_GAP_REPR,
    allow_block_reverse: bool = False,
    strategy: DockStrategy = "nonoverlap",
) -> DockingResult:
    """
    Dock multiple blocks against a target sequence.

    :var aligner: Aligner object
    :var target: target sequence as a list of symbols
    :var candidates: list of block sequences to dock
    :var gap_repr: representation of gaps in sequences
    :var allow_block_reverse: whether to consider reversed blocks
    :var strategy: docking strategy to use (currently only "nonoverlap" supported)
    :return: DockingResult object containing placements and summary information
    """
    if not target:
        raise ValueError("target sequence must not be empty")
    if not candidates:
        return DockingResult(
            placements=[],
            unused_blocks=[],
            docked_row=[gap_repr] * len(target),
            total_score=0.0,
        )
    
    aligner_obj, alphabet, gap_idx, label_fn = _alignment_ctx(aligner, gap_repr)
    idx, mat = get_symbol_score_lookup(aligner)

    target_int = seq_to_arr(list(target), alphabet, label_fn).astype(np.int32)

    # Generate placements for each block (fwd and optionally rev)
    placements: list[DockPlacement] = []
    for bi, blk in enumerate(candidates):
        blk_fwd = seq_to_arr(list(blk), alphabet, label_fn).astype(np.int32)
        orientations = [(False, blk_fwd)]
        if allow_block_reverse:
            blk_rev = seq_to_arr(list(reversed(blk)), alphabet, label_fn).astype(np.int32)
            orientations.append((True, blk_rev))

        best_for_block: DockPlacement | None = None

        for is_rev, blk_int in orientations:
            _, tgt_aln_int, blk_aln_int = pairwise_alignment(aligner_obj, target_int, blk_int, gap_repr=gap_idx)
            tgt_aln = arr_to_seq(tgt_aln_int, alphabet)
            blk_aln = arr_to_seq(blk_aln_int, alphabet)

            interval = _extract_center_interval(tgt_aln, blk_aln, gap_repr)
            if interval is None:
                continue
            start, end = interval

            region_score = score_docked_region(
                center_aln=tgt_aln,
                block_aln=blk_aln,
                start=start,
                end=end,
                gap_repr=gap_repr,
                idx=idx,
                mat=mat,
            )

            cand = DockPlacement(
                block_idx=bi,
                reversed=is_rev,
                start=start,
                end=end,
                score=float(region_score),
                center_aln=tgt_aln,
                block_aln=blk_aln,
            )
            if best_for_block is None or cand.score > best_for_block.score:
                best_for_block = cand

        if best_for_block is not None and best_for_block.score > 0.0:
            placements.append(best_for_block)

    if not placements:
        return DockingResult(
            placements=[],
            unused_blocks=list(range(len(candidates))),
            docked_row=[gap_repr] * len(target),
            total_score=0.0,
        )
    
    # Weighted interval scheduling (non-overlapping)
    placements.sort(key=lambda p: (p.end, p.start))
    ends = [p.end for p in placements]

    def prev_nonoverlap(i: int) -> int:
        """
        Find the index of the last placement that does not overlap with placement i.
        
        :var i: index of the current placement
        :return: index of the last non-overlapping placement, or -1 if none exists
        """
        lo, hi = 0, i - 1
        ans = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if placements[mid].end < placements[i].start:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans

    # Dynamic programming to find optimal set of non-overlapping placements
    n = len(placements)
    pidx = [prev_nonoverlap(i) for i in range(n)]
    dp = [0.0] * n
    take = [False] * n

    for i in range(n):
        w = placements[i].score
        incl = w + (dp[pidx[i]] if pidx[i] != -1 else 0.0)
        excl = dp[i - 1] if i > 0 else 0.0
        if incl > excl:
            dp[i] = incl
            take[i] = True
        else:
            dp[i] = excl
    
    chosen: list[DockPlacement] = []
    i = n - 1
    while i >= 0:
        if take[i]:
            chosen.append(placements[i])
            i = pidx[i]
        else:
            i -= 1
    chosen.reverse()

    used_blocks = {p.block_idx for p in chosen}
    unused_blocks = [i for i in range(len(candidates)) if i not in used_blocks]

    docker_row = _make_center_anchored_row(list(target), chosen, gap_repr)
    total_score = float(sum(p.score for p in chosen))

    # Reorder by target coordinate for readability
    chosen_sorted = sorted(chosen, key=lambda p: (p.start, p.end))

    return DockingResult(
        placements=chosen_sorted,
        unused_blocks=unused_blocks,
        docked_row=docker_row,
        total_score=total_score,
    )
