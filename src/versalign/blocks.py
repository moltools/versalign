"""Module for block alignment."""

import logging
from dataclasses import dataclass
from typing import Sequence, Literal

import numpy as np
from numpy.typing import NDArray

from versalign.aligner import Aligner, PairwiseAligner
from versalign.pairwise import pairwise_alignment, pairwise_alignment_score
from versalign.config import DEFAULT_GAP_REPR
from versalign.helpers import seq_to_arr, arr_to_seq


log = logging.getLogger(__name__)


PairingStrategy = Literal["greedy"]


@dataclass(frozen=True)
class BlockAlignment:
    """
    Represents a block alignment between two sequences.

    :var a_aln: aligned blocks of sequence A
    :var b_aln: aligned blocks of sequence B
    :var pairs: list of matched index pairs (i, j)
    :var unmatched_a: list of unmatched indices in sequence A
    :var unmatched_b: list of unmatched indices in sequence B
    """

    a_aln: list[list[str]]
    b_aln: list[list[str]]
    pairs: list[tuple[int, int]]  # matched (i, j)
    unmatched_a: list[int]
    unmatched_b: list[int]


@dataclass(frozen=True)
class CenterStarScore:
    """
    Represents scoring information for center-star MSA.

    :var center_row: index of the center row
    :var total: total score of the MSA
    :var per_row: list of scores per row
    :var per_block: list of scores per block-column
    :var per_row_per_block: 2D list of scores per row and block-column
    :var coverage_per_row: list of coverage fractions per row
    :var topk_fraction_score_per_row: list of top-k fraction scores per row
    :var best_window_score_per_row: list of best window scores per row
    """

    center_row: int
    total: float
    per_row: list[float]
    per_block: list[float]
    per_row_per_block: list[list[float]]
    coverage_per_row: list[float]
    topk_fraction_score_per_row: list[float]
    best_window_score_per_row: list[float]


@dataclass(frozen=True)
class BlockMSA:
    """
    Represents a multiple sequence alignment of blocks.

    :var msa: aligned blocks per sequence (rows x block-columns x units)
    :var order: list of tuples (sequence label, original index)
    :var score: CenterStarScore object with scoring details
    """

    msa: list[list[list[str]]]
    order: list[tuple[str, int]]
    score: CenterStarScore


def greedy_max_weight_matching(S: np.ndarray) -> list[tuple[int, int]]:
    """
    Perform greedy maximum weight matching on the score matrix S.

    :param S: 2D numpy array of shape (na, nb) with scores
    :return: list of matched index pairs (i, j)
    """
    # S: shape (na, nb)
    na, nb = S.shape
    used_a = set()
    used_b = set()
    pairs: list[tuple[int, int]] = []

    # Flatten, sort by score desc
    flat = [(S[i, j], i, j) for i in range(na) for j in range(nb)]
    flat.sort(key=lambda x: x[0], reverse=True)

    for score, i, j in flat:
        # Check if already used
        if i in used_a or j in used_b:
            continue

        # Skip non-positive matches
        if score <= 0:
            break

        pairs.append((i, j))
        used_a.add(i)
        used_b.add(j)

    return pairs


def align_blocks(
    aligner: Aligner,
    a_blocks: Sequence[Sequence[str]],
    b_blocks: Sequence[Sequence[str]],
    gap_repr: str = DEFAULT_GAP_REPR,
    pairing: PairingStrategy = "greedy",
    preserve_a_order: bool = False,
    allow_block_reverse: bool = False,
) -> BlockAlignment:
    """
    Align two sequences of blocks using pairwise alignment and a matching strategy.

    :param aligner: Aligner object for pairwise alignment
    :param a_blocks: sequence of blocks from sequence A
    :param b_blocks: sequence of blocks from sequence B
    :param gap_repr: string representation for gaps
    :param pairing: strategy for pairing blocks ("greedy" supported)
    :param preserve_a_order: whether to preserve the order of sequence A blocks
    :param allow_block_reverse: whether to allow block reversal
    :return: BlockAlignment object with aligned sequences and pairing info
    """
    aligner_obj: PairwiseAligner = aligner.aligner
    alphabet = aligner_obj.substitution_matrix.names
    gap_idx = alphabet.index(gap_repr)
    label_fn = aligner.label_fn

    na, nb = len(a_blocks), len(b_blocks)
    if na == 0 or nb == 0:
        raise ValueError("both a_blocks and b_blocks must be non-empty")
    
    # Encode blocks (chars -> int indices) once
    a_int: list[NDArray[np.int32]] = [seq_to_arr(list(block), alphabet, label_fn).astype(np.int32) for block in a_blocks]
    b_int_fwd: list[NDArray[np.int32]] = [seq_to_arr(list(block), alphabet, label_fn).astype(np.int32) for block in b_blocks]
    b_int_rev: list[NDArray[np.int32]] = [seq_to_arr(list(reversed(block)), alphabet, label_fn).astype(np.int32) for block in b_blocks]

    # Setup score matrix; remember whether reversed was better for each (i, j)
    S = np.zeros((na, nb), dtype=float)
    use_rev = np.zeros((na, nb), dtype=bool)

    for i in range(na):
        for j in range(nb):
            s_fwd = pairwise_alignment_score(aligner_obj, a_int[i], b_int_fwd[j])
            if allow_block_reverse:
                assert b_int_rev is not None
                s_rev = pairwise_alignment_score(aligner_obj, a_int[i], b_int_rev[j])
                if s_rev > s_fwd:
                    S[i, j] = s_rev
                    use_rev[i, j] = True
                else:
                    S[i, j] = s_fwd
            else:
                S[i, j] = s_fwd

    # Choose pairs
    if pairing == "greedy":
        pairs = greedy_max_weight_matching(S)
    else:
        # TODO: incorporate hungarian method?
        raise ValueError(f"unknown pairing strategy: {pairing}")
    
    used_a = {i for i, _ in pairs}
    used_b = {j for _, j in pairs}
    unmatched_a = [i for i in range(na) if i not in used_a]
    unmatched_b = [j for j in range(nb) if j not in used_b]

    # Stitch into one alignment
    a_out: list[list[str]] = []
    b_out: list[list[str]] = []

    def _b_int_for(i: int, j: int) -> NDArray[np.int32]:
        """
        Get the appropriate block integer array for block j, considering reversal.
        
        :param i: index of block in sequence A
        :param j: index of block in sequence B
        :return: integer array of block j (reversed if needed)
        """
        if allow_block_reverse and use_rev[i, j]:
            assert b_int_rev is not None
            return b_int_rev[j]
        else:
            return b_int_fwd[j]

    if preserve_a_order:
        # Keep A-block order stable (required for MSA construction)
        pair_map: dict[int, int] = {i: j for i, j in pairs}
        pairs_sorted = sorted(pairs, key=lambda ij: ij[0])

        # Emite A blocks in original order, matched or not
        for i in range(na):
            j = pair_map.get(i)
            if j is None:
                blk = list(a_blocks[i])
                a_out.append(blk)
                b_out.append([gap_repr] * len(blk))
            else:
                b_use = _b_int_for(i, j)
                _, a_aln_int, b_aln_int = pairwise_alignment(aligner_obj, a_int[i], b_use, gap_repr=gap_idx)
                a_out.append(arr_to_seq(a_aln_int, alphabet))
                b_out.append(arr_to_seq(b_aln_int, alphabet))

        # After anchoring on all A blocks, append B-only blocks (no A position)
        for j in unmatched_b:
            blk = list(b_blocks[j])
            a_out.append([gap_repr] * len(blk))
            b_out.append(blk)

    else:
        # Order of stitching by descending match score
        pairs_sorted = sorted(pairs, key=lambda ij: S[ij[0], ij[1]], reverse=True)

        for i, j in pairs_sorted:
            b_use = _b_int_for(i, j)
            _, a_aln_int, b_aln_int = pairwise_alignment(aligner_obj, a_int[i], b_use, gap_repr=gap_idx)
            a_out.append(arr_to_seq(a_aln_int, alphabet))
            b_out.append(arr_to_seq(b_aln_int, alphabet))

        # Append unmatched blocks (keep internal order)
        for i in unmatched_a:
            blk = list(a_blocks[i])
            a_out.append(blk)
            b_out.append([gap_repr] * len(blk))

        for j in unmatched_b:
            blk = list(b_blocks[j])
            a_out.append([gap_repr] * len(blk))
            b_out.append(blk)

    return BlockAlignment(
        a_aln=a_out,
        b_aln=b_out,
        pairs=pairs_sorted,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
    )


def is_gap_block(block: Sequence[str], gap_repr: str) -> bool:
    """
    Check if a block consists entirely of gap representations.

    :param block: sequence of string tokens in the block
    :param gap_repr: string representation for gaps
    :return: True if the block is a gap block, False otherwise
    """
    return len(block) > 0 and all(u == gap_repr for u in block)


def pad_block(block: list[str], L: int, gap_repr: str) -> list[str]:
    """
    Pad a block to length L using gap_repr.

    :param block: sequence of string tokens in the block
    :param L: desired length of the block after padding
    :param gap_repr: string representation for gaps
    :return: padded block if original length < L, else original block
    """
    if len(block) >= L:
        return block
    
    return block + [gap_repr] * (L - len(block))


def insert_gaps(block: list[str], positions: list[int], gap_repr: str) -> list[str]:
    """
    Insert gaps into a block at specified positions.

    :param block: sequence of string tokens in the block
    :param positions: list of indices where gaps should be inserted
    :param gap_repr: string representation for gaps
    :return: new block with gaps inserted
    """
    out = block[:]
    for pos in positions:
        out.insert(pos, gap_repr)

    return out


def strip_gap_columns(row: list[list[str]], gap_repr: str) -> list[list[str]]:
    """
    Remove gap columns from row.

    :param row: sequence blocks
    :param gap_repr: string representation for gaps
    :return: stripped row of gap columns
    """
    return [blk for blk in row if not is_gap_block(blk, gap_repr)]


def merge_column_by_center(
    aligner: Aligner,
    col_blocks: list[list[str]],
    center_row: int,
    center_new: list[str],
    row_new: list[str],
    gap_repr: str,
) -> tuple[list[list[str]], list[str]]:
    """
    Merge a new block into an existing column of blocks using the center row as anchor.

    :param aligner: Aligner object for pairwise alignment
    :param col_blocks: existing column of blocks (one per existing row)
    :param center_row: index of the center row in col_blocks
    :param center_new: new center block to align against
    :param row_new: new block to be merged into the column
    :param gap_repr: string representation for gaps
    :return: tuple of (updated column of blocks, updated new row block)
    """
    aligner_obj: PairwiseAligner = aligner.aligner
    alphabet = aligner_obj.substitution_matrix.names
    gap_idx = alphabet.index(gap_repr)
    label_fn = aligner.label_fn

    center_old = col_blocks[center_row]

    old_int = seq_to_arr(list(center_old), alphabet, label_fn).astype(np.int32)
    newc_int = seq_to_arr(list(center_new), alphabet, label_fn).astype(np.int32)

    # Align old center to new center; decode to symbols
    _, aln_old_int, aln_newc_int = pairwise_alignment(aligner_obj, old_int, newc_int, gap_repr=gap_idx)
    old_aln = arr_to_seq(aln_old_int, alphabet)
    newc_aln = arr_to_seq(aln_newc_int, alphabet)

    # Propagate gaps from both old_aln and newc_aln
    # Gaps inserted into old center
    insert_pos_old = [k for k, u in enumerate(old_aln) if u == gap_repr]
    updated_col = [insert_gaps(list(b), insert_pos_old, gap_repr) for b in col_blocks]
    updated_row = insert_gaps(list(row_new), insert_pos_old, gap_repr)

    # Gaps inserted into new center must also be propagated
    insert_pos_new = [k for k, u in enumerate(newc_aln) if u == gap_repr]
    if insert_pos_new:
        updated_col = [insert_gaps(list(b), insert_pos_new, gap_repr) for b in updated_col]
        updated_row = insert_gaps(list(updated_row), insert_pos_new, gap_repr)

    # Rectangularize lengths (just to be sure!)
    L = max(len(b) for b in updated_col + [updated_row])
    updated_col = [pad_block(b, L, gap_repr) for b in updated_col]
    updated_row = pad_block(updated_row, L, gap_repr)
   
    return updated_col, updated_row


def get_symbol_score_lookup(aligner: Aligner) -> tuple[dict[str, int], np.ndarray]:
    """
    Get symbol to index mapping and score matrix from the aligner's substitution matrix.

    :param aligner: Aligner object
    :return: tuple of (symbol to index mapping, score matrix as numpy array)
    """
    sm = aligner.aligner.substitution_matrix
    names = sm.names
    idx = {u: i for i, u in enumerate(names)}
    mat = np.asarray(sm, dtype=float)

    return idx, mat


def score_aligned_pair(
    a: list[str],
    b: list[str],
    idx: dict[str, int],
    mat: np.ndarray,
) -> float:
    """
    Score a pair of aligned blocks using the substitution matrix.

    :param a: first aligned block
    :param b: second aligned block
    :param idx: mapping from symbols to indices in the score matrix
    :param mat: score matrix as numpy array
    :return: total score of the aligned pair
    """
    if len(a) != len(b):
        raise ValueError("aligned blocks must have the same length")
    
    total = 0.0

    # Pure substitution matrix scoring (including gap vs base)
    for x, y in zip(a, b):
        total += mat[idx[x], idx[y]]

    return total


def calc_coverage(
    center_blocks: list[list[str]],
    row_blocks: list[list[str]],
    gap_repr: str,
) -> float:
    """
    Calculate coverage fraction between center blocks and row blocks.

    :param center_blocks: blocks of the center row
    :param row_blocks: blocks of the target row
    :param gap_repr: string representation for gaps
    :return: coverage fraction (num covered positions / total non-gap positions in center)
    """
    num = 0
    den = 0
    for cb, rb in zip(center_blocks, row_blocks):
        if len(cb) != len(rb):
            raise ValueError("block length mismatch in calc_coverage")
        
        for c, r in zip(cb, rb):
            if c != gap_repr:
                den += 1
                if r != gap_repr:
                    num += 1

    return (num / den) if den > 0 else 1.0


def best_contiguous_window(scores: list[float]) -> float:
    """
    Find the best contiguous window sum in a list of scores using Kadane's algorithm.

    :param scores: list of float scores
    :return: best contiguous window sum
    """
    best = float("-inf")
    cur = 0.0
    for s in scores:
        cur = max(s, cur + s)
        best = max(best, cur)
    
    return 0.0 if best == float("-inf") else best


def center_vs_all_score(
    msa: list[list[list[str]]],
    aligner: Aligner,
    center_row: int = 0,
    gap_repr: str = DEFAULT_GAP_REPR,
    topk_fraction: float = 0.5,
) -> CenterStarScore:
    """
    Calculate center-vs-all scoring for a block MSA.
    
    :param msa: multiple sequence alignment of blocks
    :param aligner: Aligner object for pairwise alignment
    :param center_row: index of the center row in the MSA
    :param gap_repr: string representation for gaps
    :param topk_fraction: fraction for top-k scoring
    :return: CenterStarScore object with scoring details
    """
    rows = msa
    if not rows:
        return CenterStarScore(center_row, 0.0, [], [], [], [], [], [])
    
    n_rows = len(rows)
    n_cols = len(rows[0])
    if not (0 <= center_row < n_rows):
        raise ValueError("center_row out of range")
    
    # Validate rectangularity
    for r in rows:
        if len(r) != n_cols:
            raise ValueError("inconsistent number of block columns in MSA")
        
    idx, mat = get_symbol_score_lookup(aligner)

    center_blocks = rows[center_row]

    per_row = [0.0] * n_rows
    per_block = [0.0] * n_cols
    per_row_per_block: list[list[float]] = [[0.0] * n_cols for _ in range(n_rows)]
    coverage_per_row = [0.0] * n_rows
    topk_fraction_score_per_row = [0.0] * n_rows
    best_window_score_per_row = [0.0] * n_rows

    total = 0.0

    for r in range(n_rows):
        if r == center_row:
            coverage_per_row[r] = 1.0
            best_window_score_per_row[r] = 0.0
            topk_fraction_score_per_row[r] = 0.0
            continue

        row_blocks = rows[r]

        # Per-block-column scoring
        block_scores: list[float] = []
        row_total = 0.0
        for c in range(n_cols):
            cb = center_blocks[c]
            rb = row_blocks[c]
            s = score_aligned_pair(cb, rb, idx=idx, mat=mat)
            per_row_per_block[r][c] = s
            block_scores.append(s)
            row_total += s
            per_block[c] += s

        per_row[r] = row_total
        total += row_total
        coverage_per_row[r] = calc_coverage(center_blocks, row_blocks, gap_repr)

        # Partial-goodness: best top-k fractin of block-columns
        frac = max(0.0, min(1.0, topk_fraction))
        k = max(1, int(round(frac * n_cols)))
        topk = sorted(block_scores, reverse=True)[:k]
        topk_fraction_score_per_row[r] = float(sum(topk))

        # Partial-goodness: best contiguous window of block-columns
        best_window_score_per_row[r] = best_contiguous_window(block_scores)
    
    return CenterStarScore(
        center_row=center_row,
        total=total,
        per_row=per_row,
        per_block=per_block,
        per_row_per_block=per_row_per_block,
        coverage_per_row=coverage_per_row,
        topk_fraction_score_per_row=topk_fraction_score_per_row,
        best_window_score_per_row=best_window_score_per_row,
    )


def calc_block_msa(
    aligner: Aligner,
    rows: list[list[list[str]]],
    gap_repr: str = DEFAULT_GAP_REPR,
    labels: list[str] | None = None,
    center_idx: int = 0,
    allow_block_reverse: bool = False,
) -> BlockMSA:
    """
    Calculate a multiple sequence alignment (MSA) of blocks using center-star approach.

    :param aligner: Aligner object for pairwise alignment
    :param rows: list of sequences of blocks (each sequence is a list of blocks)
    :param gap_repr: string representation for gaps
    :param labels: optional list of labels for each sequence
    :param center_idx: index of the center sequence to align others against
    :param allow_block_reverse: whether to allow block reversal
    :return: BlockMSA object with aligned blocks and order info
    """
    if not rows:
        return BlockMSA(msa=[], order=[])
    
    n = len(rows)
    if labels is None:
        labels = [f"seq_{i + 1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("length of labels must match number of sequences")
    if not (0 <= center_idx < n):
        raise ValueError("center_idx out of range")

    # Start MSA with the chosen center as first row
    msa: list[list[str]] = [[list(b) for b in rows[center_idx]]]
    order: list[tuple[str, int]] = [(labels[center_idx], center_idx)]
    center_row_in_msa = 0

    for idx in range(n):
        if idx == center_idx:
            continue

        center_blocks = msa[center_row_in_msa]
        row_blocks = rows[idx]

        # Align blocks, but do not reorder center
        aln = align_blocks(
            aligner,
            center_blocks,
            row_blocks,
            gap_repr=gap_repr,
            preserve_a_order=True,
            allow_block_reverse=allow_block_reverse,
        )
        cen_aln = aln.a_aln
        row_aln = aln.b_aln

        # Expand existing MSA rows to match new block-column system
        expanded_msa: list[list[list[str]]] = []

        # The center alignment defines the target block-columns
        center_target = cen_aln

        for old_row in msa:
            # Take only the real blocks currently present in this row (ignore gap columns)
            old_real = strip_gap_columns(old_row, gap_repr)

            # Align this row's real blocks to the new center columns
            # We preserver the order of the center, blocks of new row are free to be reordered as needed
            row_fit = align_blocks(
                aligner,
                center_target,
                old_real,
                gap_repr=gap_repr,
                preserve_a_order=True,
                allow_block_reverse=allow_block_reverse,
            )

            # row_fit.a_aln should match center_target
            # row_fit.b_aln is the expanded/reordered old_row
            expanded_msa.append([list(b) for b in row_fit.b_aln])

        # For each column, merge within-block gaps using center row as anchor
        new_row_out: list[list[str]] = []
        n_cols = len(cen_aln)

        for c in range(n_cols):
            col_blocks = [expanded_msa[r][c] for r in range(len(expanded_msa))]
            cen_new = list(cen_aln[c])
            row_new = list(row_aln[c])

            # Handle gap-only columns and missing blocks
            if is_gap_block(cen_new, gap_repr) and is_gap_block(row_new, gap_repr):
                L = max(len(b) for b in col_blocks + [row_new])
                col_blocks = [pad_block(b, L, gap_repr) for b in col_blocks]
                row_new = pad_block(row_new, L, gap_repr)
                for r in range(len(expanded_msa)):
                    expanded_msa[r][c] = col_blocks[r]
                new_row_out.append(row_new)
                continue

            if is_gap_block(cen_new, gap_repr):
                # New block column inserted relative to old center
                L = max(len(b) for b in col_blocks + [row_new])
                col_blocks = [pad_block(b, L, gap_repr) for b in col_blocks]
                row_new = pad_block(row_new, L, gap_repr)
                for r in range(len(expanded_msa)):
                    expanded_msa[r][c] = col_blocks[r]
                new_row_out.append(row_new)
                continue

            if is_gap_block(row_new, gap_repr):
                # Row is missing this block; emit a gap block of the column length
                L = max(len(b) for b in col_blocks + [cen_new])
                col_blocks = [pad_block(b, L, gap_repr) for b in col_blocks]
                for r in range(len(expanded_msa)):
                    expanded_msa[r][c] = col_blocks[r]
                new_row_out.append([gap_repr] * L)
                continue

            # Both are real blocks: merge internal gaps across all existing rows
            updated_col, updated_row = merge_column_by_center(
                aligner,
                col_blocks,
                center_row=center_row_in_msa,
                center_new=cen_new,
                row_new=row_new,
                gap_repr=gap_repr,
            )
            for r in range(len(expanded_msa)):
                expanded_msa[r][c] = updated_col[r]
            new_row_out.append(updated_row)

        expanded_msa.append(new_row_out)

        # Enforce same number of block-columns across all rows
        n_cols_final = max(len(row) for row in expanded_msa)

        # Pad older rows with missing columns (as gap blocks)
        for r in range(len(expanded_msa)):
            while len(expanded_msa[r]) < n_cols_final:
                expanded_msa[r].append([gap_repr])

        # Ensure each column has consistent within-block length across rows
        for c in range(n_cols_final):
            L = max(len(expanded_msa[r][c]) for r in range(len(expanded_msa)))
            for r in range(len(expanded_msa)):
                expanded_msa[r][c] = pad_block(expanded_msa[r][c], L, gap_repr)

        # Remove block columns again that are all gaps across all rows
        all_gap_block_cols: list[int] = []
        for block_col in range(n_cols_final):
            if all(is_gap_block(expanded_msa[r][block_col], gap_repr) for r in range(len(expanded_msa))):
                all_gap_block_cols.append(block_col)

        padded_msa: list[list[list[str]]] = []
        for r in range(len(expanded_msa)):
            new_row = [expanded_msa[r][c] for c in range(n_cols_final) if c not in all_gap_block_cols]
            padded_msa.append(new_row)

        # Find all-gap columns across all blocks, remove them
        all_gap_col: list[tuple[int, int]] = []  # (block index, col index)
        for b_idx in range(len(padded_msa[0])):
            block_len = len(padded_msa[0][b_idx])
            for col_idx in range(block_len):
                if all(padded_msa[r][b_idx][col_idx] == gap_repr for r in range(len(padded_msa))):
                    all_gap_col.append((b_idx, col_idx))

        pruned_msa: list[list[list[str]]] = []
        for r in range(len(padded_msa)):
            new_row: list[list[str]] = []
            for b_idx in range(len(padded_msa[r])):
                block = padded_msa[r][b_idx]
                # Remove gap columns from this block
                gap_cols_in_block = [col_idx for (bb_idx, col_idx) in all_gap_col if bb_idx == b_idx]
                new_block = [block[col_idx] for col_idx in range(len(block)) if col_idx not in gap_cols_in_block]
                new_row.append(new_block)
            pruned_msa.append(new_row)

        msa = pruned_msa
        order.append((labels[idx], idx))

    # Score MSA
    score = center_vs_all_score(msa=msa, aligner=aligner, center_row=center_idx, gap_repr=gap_repr)

    # Reorder based on total score; center star is always first
    score_order = sorted(
        [(score.per_row[r], r) for r in range(len(msa))],
        key=lambda x: x[0],
        reverse=True,
    )
    # Ensure center row is first
    score_order = [(score.per_row[center_idx], center_idx)] + [sr for sr in score_order if sr[1] != center_idx]

    msa = [msa[r] for _, r in score_order]
    order = [order[r] for _, r in score_order]

    return BlockMSA(msa=msa, order=order, score=score)