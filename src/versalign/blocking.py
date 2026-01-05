"""Module for block alignment."""

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from versalign.aligner import Aligner, PairwiseAligner
from versalign.config import DEFAULT_GAP_REPR
from versalign.helpers import arr_to_seq, seq_to_arr
from versalign.pairwise import pairwise_alignment, pairwise_alignment_score


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
    pairs: list[tuple[int, int]]
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

    @classmethod
    def empty(self) -> "CenterStarScore":
        """
        Create an empty CenterStarScore.
        
        :return: CenterStarScore with zeroed fields
        """
        return CenterStarScore(
            center_row=0,
            total=0.0,
            per_row=[],
            per_block=[],
            per_row_per_block=[],
            coverage_per_row=[],
            topk_fraction_score_per_row=[],
            best_window_score_per_row=[],
        )


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


@dataclass(frozen=True)
class _AlphabetCtx:
    aligner_obj: PairwiseAligner
    alphabet: list[str]
    gap_idx: int
    gap_repr: str
    label_fn: object  # user-defined label function


def _ctx(aligner: Aligner, gap_repr: str) -> _AlphabetCtx:
    aligner_obj: PairwiseAligner = aligner.aligner
    alphabet = list(aligner_obj.substitution_matrix.names)
    try:
        gap_idx = alphabet.index(gap_repr)
    except ValueError as e:
        raise ValueError(f"gap_repr {gap_repr!r} not in substitution matrix alphabet") from e
    return _AlphabetCtx(
        aligner_obj=aligner_obj,
        alphabet=alphabet,
        gap_idx=gap_idx,
        gap_repr=gap_repr,
        label_fn=aligner.label_fn,
    )


def _encode_block(block: Sequence[str], alphabet: list[str], label_fn: object) -> NDArray[np.int32]:
    """
    Encode a block of sequence into a numpy array of integers.
    
    :param block: sequence block as a list of symbols
    :param alphabet: list of symbols in the alphabet
    :param label_fn: user-defined label function for encoding
    :return: numpy array of encoded integers
    """
    return seq_to_arr(list(block), alphabet, label_fn).astype(np.int32)


def _decode_block(arr: NDArray[np.int32], alphabet: list[str]) -> list[str]:
    """
    Decode a numpy array of integers back into a sequence block.
    
    :param arr: numpy array of encoded integers
    :param alphabet: list of symbols in the alphabet
    :return: sequence block as a list of symbols
    """
    return arr_to_seq(arr, alphabet)


def greedy_max_weight_matching(S: np.ndarray) -> list[tuple[int, int]]:
    """
    Perform greedy maximum weight matching on the score matrix S.

    :param S: 2D numpy array of shape (na, nb) with scores
    :return: list of matched index pairs (i, j)
    """
    na, nb = S.shape
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []

    flat = [(S[i, j], i, j) for i in range(na) for j in range(nb)]
    flat.sort(key=lambda x: x[0], reverse=True)

    for score, i, j in flat:
        if i in used_a or j in used_b:
            continue
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
    :param a_blocks: sequence of blocks for sequence A
    :param b_blocks: sequence of blocks for sequence B
    :param gap_repr: representation of gaps in the blocks
    :param pairing: strategy for pairing blocks ('greedy' supported)
    :param preserve_a_order: whether to preserve the order of sequence A
    :param allow_block_reverse: whether to allow reversing blocks in sequence B
    :return: BlockAlignment object with aligned blocks and pairing information
    """
    ctx = _ctx(aligner, gap_repr)

    na, nb = len(a_blocks), len(b_blocks)
    if na == 0 or nb == 0:
        raise ValueError("both a_blocks and b_blocks must be non-empty")

    # Encode blocks once
    a_int: list[NDArray[np.int32]] = [_encode_block(block, ctx.alphabet, ctx.label_fn) for block in a_blocks]
    b_int_fwd: list[NDArray[np.int32]] = [_encode_block(block, ctx.alphabet, ctx.label_fn) for block in b_blocks]
    b_int_rev: list[NDArray[np.int32]] | None = None
    if allow_block_reverse:
        b_int_rev = [_encode_block(list(reversed(block)), ctx.alphabet, ctx.label_fn) for block in b_blocks]

    # Score matrix + reversal choice
    S = np.zeros((na, nb), dtype=float)
    use_rev = np.zeros((na, nb), dtype=bool)

    for i in range(na):
        ai = a_int[i]
        for j in range(nb):
            s_fwd = pairwise_alignment_score(ctx.aligner_obj, ai, b_int_fwd[j])
            if allow_block_reverse:
                assert b_int_rev is not None
                s_rev = pairwise_alignment_score(ctx.aligner_obj, ai, b_int_rev[j])
                if s_rev > s_fwd:
                    S[i, j] = s_rev
                    use_rev[i, j] = True
                else:
                    S[i, j] = s_fwd
            else:
                S[i, j] = s_fwd

    # Pair blocks
    if pairing == "greedy":
        pairs = greedy_max_weight_matching(S)
    else:
        raise ValueError(f"unknown pairing strategy: {pairing}")

    used_a = {i for i, _ in pairs}
    used_b = {j for _, j in pairs}
    unmatched_a = [i for i in range(na) if i not in used_a]
    unmatched_b = [j for j in range(nb) if j not in used_b]

    def _b_int_for(i: int, j: int) -> NDArray[np.int32]:
        """
        Get the encoded block for b_blocks[j], possibly reversed.
        
        :param i: index in a_blocks
        :param j: index in b_blocks
        :return: encoded block as numpy array
        """
        if allow_block_reverse and use_rev[i, j]:
            assert b_int_rev is not None
            return b_int_rev[j]
        return b_int_fwd[j]

    a_out: list[list[str]] = []
    b_out: list[list[str]] = []

    if preserve_a_order:
        pair_map: dict[int, int] = {i: j for i, j in pairs}
        pairs_sorted = sorted(pairs, key=lambda ij: ij[0])

        for i in range(na):
            j = pair_map.get(i)
            if j is None:
                blk = list(a_blocks[i])
                a_out.append(blk)
                b_out.append([gap_repr] * len(blk))
            else:
                b_use = _b_int_for(i, j)
                _, a_aln_int, b_aln_int = pairwise_alignment(
                    ctx.aligner_obj, a_int[i], b_use, gap_repr=ctx.gap_idx
                )
                a_out.append(_decode_block(a_aln_int, ctx.alphabet))
                b_out.append(_decode_block(b_aln_int, ctx.alphabet))

        for j in unmatched_b:
            blk = list(b_blocks[j])
            a_out.append([gap_repr] * len(blk))
            b_out.append(blk)

    else:
        pairs_sorted = sorted(pairs, key=lambda ij: S[ij[0], ij[1]], reverse=True)

        for i, j in pairs_sorted:
            b_use = _b_int_for(i, j)
            _, a_aln_int, b_aln_int = pairwise_alignment(
                ctx.aligner_obj, a_int[i], b_use, gap_repr=ctx.gap_idx
            )
            a_out.append(_decode_block(a_aln_int, ctx.alphabet))
            b_out.append(_decode_block(b_aln_int, ctx.alphabet))

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
    
    :param block: sequence block as a list of symbols
    :param gap_repr: representation of gaps in the blocks
    :return: True if the block is a gap block, False otherwise
    """
    return len(block) > 0 and all(u == gap_repr for u in block)


def pad_block(block: list[str], L: int, gap_repr: str) -> list[str]:
    """
    Pad a block to length L with gap representations.
    
    :param block: sequence block as a list of symbols
    :param L: desired length of the block
    :param gap_repr: representation of gaps in the blocks
    :return: padded block as a list of symbols
    """
    if len(block) >= L:
        return block

    return block + [gap_repr] * (L - len(block))


def insert_gaps(block: list[str], positions: list[int], gap_repr: str) -> list[str]:
    """
    Insert gaps into a block at specified positions.
    
    :param block: sequence block as a list of symbols
    :param positions: list of positions to insert gaps
    :param gap_repr: representation of gaps in the blocks
    :return: block with gaps inserted as a list of symbols
    """
    out = block[:]
    for pos in positions:
        out.insert(pos, gap_repr)

    return out


def strip_gap_columns(row: list[list[str]], gap_repr: str) -> list[list[str]]:
    """
    Remove all-gap blocks from a row of blocks.
    
    :param row: list of blocks in the row
    :param gap_repr: representation of gaps in the blocks
    :return: row with all-gap blocks removed
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
    Merge a column of blocks based on a new center block alignment.
    
    :param aligner: Aligner object for pairwise alignment
    :param col_blocks: list of blocks in the column
    :param center_row: index of the center row in the column
    :param center_new: new center block after alignment
    :param row_new: new row block after alignment
    :param gap_repr: representation of gaps in the blocks
    :return: tuple of updated column blocks and updated row block
    """
    ctx = _ctx(aligner, gap_repr)

    center_old = col_blocks[center_row]
    old_int = _encode_block(center_old, ctx.alphabet, ctx.label_fn)
    newc_int = _encode_block(center_new, ctx.alphabet, ctx.label_fn)

    _, aln_old_int, aln_newc_int = pairwise_alignment(
        ctx.aligner_obj, old_int, newc_int, gap_repr=ctx.gap_idx
    )
    old_aln = _decode_block(aln_old_int, ctx.alphabet)
    newc_aln = _decode_block(aln_newc_int, ctx.alphabet)

    insert_pos_old = [k for k, u in enumerate(old_aln) if u == gap_repr]
    updated_col = [insert_gaps(list(b), insert_pos_old, gap_repr) for b in col_blocks]
    updated_row = insert_gaps(list(row_new), insert_pos_old, gap_repr)

    insert_pos_new = [k for k, u in enumerate(newc_aln) if u == gap_repr]
    if insert_pos_new:
        updated_col = [insert_gaps(list(b), insert_pos_new, gap_repr) for b in updated_col]
        updated_row = insert_gaps(list(updated_row), insert_pos_new, gap_repr)

    L = max(len(b) for b in updated_col + [updated_row])
    updated_col = [pad_block(b, L, gap_repr) for b in updated_col]
    updated_row = pad_block(updated_row, L, gap_repr)

    return updated_col, updated_row


def get_symbol_score_lookup(aligner: Aligner) -> tuple[dict[str, int], np.ndarray]:
    """
    Get a lookup table for symbol scores from the aligner's substitution matrix.
    
    :param aligner: Aligner object
    :return: tuple of (index mapping, score matrix)
    """
    sm = aligner.aligner.substitution_matrix
    names = sm.names
    idx = {u: i for i, u in enumerate(names)}
    mat = np.asarray(sm, dtype=float)

    return idx, mat


def score_aligned_pair(a: list[str], b: list[str], idx: dict[str, int], mat: np.ndarray) -> float:
    """
    Score a pair of aligned blocks using the substitution matrix.
    
    :param a: first aligned block as a list of symbols
    :param b: second aligned block as a list of symbols
    :param idx: index mapping for symbols
    :param mat: score matrix
    :return: alignment score
    """
    if len(a) != len(b):
        raise ValueError("aligned blocks must have the same length")

    total = 0.0

    for x, y in zip(a, b):
        total += mat[idx[x], idx[y]]

    return total


def calc_coverage(center_blocks: list[list[str]], row_blocks: list[list[str]], gap_repr: str) -> float:
    """
    Calculate the coverage fraction between center blocks and row blocks.
    
    :param center_blocks: list of blocks in the center row
    :param row_blocks: list of blocks in the target row
    :param gap_repr: representation of gaps in the blocks
    :return: coverage fraction
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
    Find the best contiguous window score using Kadane's algorithm.
    
    :param scores: list of scores
    :return: best contiguous window score
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
    
    :param msa: multiple sequence alignment as a list of rows of blocks
    :param aligner: Aligner object for pairwise alignment
    :param center_row: index of the center row
    :param gap_repr: representation of gaps in the blocks
    :param topk_fraction: fraction of top block scores to consider
    :return: CenterStarScore object with scoring details
    """
    rows = msa
    if not rows:
        return CenterStarScore(center_row, 0.0, [], [], [], [], [], [])

    n_rows = len(rows)
    n_cols = len(rows[0])
    if not (0 <= center_row < n_rows):
        raise ValueError("center_row out of range")

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

        frac = max(0.0, min(1.0, topk_fraction))
        k = max(1, int(round(frac * n_cols)))
        topk = sorted(block_scores, reverse=True)[:k]
        topk_fraction_score_per_row[r] = float(sum(topk))

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
    Calculate a block multiple sequence alignment using the center-star method.
    
    :param aligner: Aligner object for pairwise alignment
    :param rows: list of sequences of blocks to align
    :param gap_repr: representation of gaps in the blocks
    :param labels: optional list of sequence labels
    :param center_idx: index of the center sequence
    :param allow_block_reverse: whether to allow reversing blocks during alignment
    :return: BlockMSA object with aligned blocks and scoring details
    """
    if not rows:
        empty_score = CenterStarScore.empty()
        return BlockMSA(msa=[], order=[], score=empty_score)

    n = len(rows)
    if labels is None:
        labels = [f"seq_{i + 1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("length of labels must match number of sequences")
    if not (0 <= center_idx < n):
        raise ValueError("center_idx out of range")

    msa: list[list[list[str]]] = [[list(b) for b in rows[center_idx]]]
    order: list[tuple[str, int]] = [(labels[center_idx], center_idx)]
    center_row_in_msa = 0

    for idx in range(n):
        if idx == center_idx:
            continue

        center_blocks = msa[center_row_in_msa]
        row_blocks = rows[idx]

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

        expanded_msa: list[list[list[str]]] = []
        center_target = cen_aln

        for old_row in msa:
            old_real = strip_gap_columns(old_row, gap_repr)
            row_fit = align_blocks(
                aligner,
                center_target,
                old_real,
                gap_repr=gap_repr,
                preserve_a_order=True,
                allow_block_reverse=allow_block_reverse,
            )
            expanded_msa.append([list(b) for b in row_fit.b_aln])

        new_row_out: list[list[str]] = []
        n_cols = len(cen_aln)

        for c in range(n_cols):
            col_blocks = [expanded_msa[r][c] for r in range(len(expanded_msa))]
            cen_new = list(cen_aln[c])
            row_new = list(row_aln[c])

            if is_gap_block(cen_new, gap_repr) and is_gap_block(row_new, gap_repr):
                L = max(len(b) for b in col_blocks + [row_new])
                col_blocks = [pad_block(b, L, gap_repr) for b in col_blocks]
                row_new = pad_block(row_new, L, gap_repr)
                for r in range(len(expanded_msa)):
                    expanded_msa[r][c] = col_blocks[r]
                new_row_out.append(row_new)
                continue

            if is_gap_block(cen_new, gap_repr):
                L = max(len(b) for b in col_blocks + [row_new])
                col_blocks = [pad_block(b, L, gap_repr) for b in col_blocks]
                row_new = pad_block(row_new, L, gap_repr)
                for r in range(len(expanded_msa)):
                    expanded_msa[r][c] = col_blocks[r]
                new_row_out.append(row_new)
                continue

            if is_gap_block(row_new, gap_repr):
                L = max(len(b) for b in col_blocks + [cen_new])
                col_blocks = [pad_block(b, L, gap_repr) for b in col_blocks]
                for r in range(len(expanded_msa)):
                    expanded_msa[r][c] = col_blocks[r]
                new_row_out.append([gap_repr] * L)
                continue

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

        n_cols_final = max(len(row) for row in expanded_msa)

        for r in range(len(expanded_msa)):
            while len(expanded_msa[r]) < n_cols_final:
                expanded_msa[r].append([gap_repr])

        for c in range(n_cols_final):
            L = max(len(expanded_msa[r][c]) for r in range(len(expanded_msa)))
            for r in range(len(expanded_msa)):
                expanded_msa[r][c] = pad_block(expanded_msa[r][c], L, gap_repr)

        # Final cleanup: remove all-gap block columns and all-gap columns within blocks
        all_gap_block_cols: list[int] = []
        for block_col in range(n_cols_final):
            if all(is_gap_block(expanded_msa[r][block_col], gap_repr) for r in range(len(expanded_msa))):
                all_gap_block_cols.append(block_col)
        
        # Remove all-gap block columns
        padded_msa: list[list[list[str]]] = []
        for r in range(len(expanded_msa)):
            new_row = [expanded_msa[r][c] for c in range(n_cols_final) if c not in all_gap_block_cols]
            padded_msa.append(new_row)

        all_gap_col: list[tuple[int, int]] = []
        for b_idx in range(len(padded_msa[0])):
            block_len = len(padded_msa[0][b_idx])
            for col_idx in range(block_len):
                if all(padded_msa[r][b_idx][col_idx] == gap_repr for r in range(len(padded_msa))):
                    all_gap_col.append((b_idx, col_idx))

        # Remove all-gap columns within blocks
        pruned_msa: list[list[list[str]]] = []
        for r in range(len(padded_msa)):
            new_row: list[list[str]] = []
            for b_idx in range(len(padded_msa[r])):
                block = padded_msa[r][b_idx]
                gap_cols_in_block = [col_idx for (bb_idx, col_idx) in all_gap_col if bb_idx == b_idx]
                new_block = [block[col_idx] for col_idx in range(len(block)) if col_idx not in gap_cols_in_block]
                new_row.append(new_block)
            pruned_msa.append(new_row)

        msa = pruned_msa
        order.append((labels[idx], idx))

    # Score MSA (center row in MSA is always 0 by construction)
    score = center_vs_all_score(msa=msa, aligner=aligner, center_row=center_idx, gap_repr=gap_repr)

    # Reorder based on total score; center star is always first
    score_order = sorted(
        [(score.per_row[r], r) for r in range(len(msa))],
        key=lambda x: x[0],
        reverse=True,
    )
    score_order = [(score.per_row[center_idx], center_idx)] + [sr for sr in score_order if sr[1] != center_idx]

    msa = [msa[r] for _, r in score_order]
    order = [order[r] for _, r in score_order]

    return BlockMSA(msa=msa, order=order, score=score)
