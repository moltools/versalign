#!/usr/bin/env python3

"""Pretty printing for alignment results."""

from collections.abc import Sequence


def format_alignment(
    aligned_seqs: Sequence[Sequence[object]],
    *,
    names: Sequence[str] | None = None,
    score: float | None = None,
    gap_repr: str = "-",
    block_cols: int | None = None,  # e.g. 60 to wrap long alignments
    show_consensus: bool = True,
) -> str:
    """
    Pretty-format aligned sequences with per-column width so multi-char tokens align.

    :param aligned_seqs: list of aligned sequences (each a list/tuple of tokens)
    :param names: optional sequence names
    :param score: optional alignment score to print at the top
    :param gap_repr: string used to represent a gap when computing consensus (defaults to "-")
    :param block_cols: if given, wrap output into blocks of this many columns
    :param show_consensus: if True, include a simple consensus line
    :returns: formatted alignment string
    """
    if not aligned_seqs:
        return "(no sequences)"

    nseq = len(aligned_seqs)
    lengths = [len(s) for s in aligned_seqs]
    if len(set(lengths)) != 1:
        raise ValueError(f"All aligned sequences must have same length; got {lengths}")
    L = lengths[0]

    # Coerce everything to strings once
    seqs_str: list[list[str]] = [[str(tok) for tok in seq] for seq in aligned_seqs]

    # Default names if not provided
    if names is None:
        names = [f"seq{i + 1}" for i in range(nseq)]
    if len(names) != nseq:
        raise ValueError("names length must match number of sequences")

    # Compute per-column widths based on the widest token at that column
    col_w = [0] * L
    for j in range(L):
        col_w[j] = max(len(seqs_str[i][j]) for i in range(nseq))

    # Left margin width for names
    name_w = max(len(n) for n in names) if names else 0

    def consensus_char(j: int) -> str:
        """Return '|' if all non-gap tokens equal (and at least one non-gap), else ' '."""
        tokens = [seqs_str[i][j] for i in range(nseq)]
        nongap = [t for t in tokens if t != gap_repr]
        if not nongap:
            return " "
        # If there is at least one gap, or not all equal, return ' '
        if len(nongap) < len(tokens):
            return " "
        return "|" if len(set(nongap)) == 1 else " "

    def render_block(col_start: int, col_end: int) -> list[str]:
        # Build lines for each sequence
        lines: list[str] = []
        for i in range(nseq):
            left = (names[i].rjust(name_w) + " > ") if name_w > 0 else ""
            cells = [
                seqs_str[i][j].ljust(col_w[j])  # left-pad tokens so columns align
                for j in range(col_start, col_end)
            ]
            lines.append(left + " ".join(cells))
        if show_consensus:
            left = (" " * name_w + "   ") if name_w > 0 else ""
            c = [consensus_char(j) for j in range(col_start, col_end)]
            # Consensus is one char per column; widen to column width by padding right
            c_cells = [c[j - col_start].ljust(col_w[j]) for j in range(col_start, col_end)]
            lines.append(left + " ".join(c_cells))
        return lines

    # Wrap into blocks if requested
    blocks: list[str] = []
    if block_cols and block_cols > 0:
        for start in range(0, L, block_cols):
            end = min(start + block_cols, L)
            blocks.extend(render_block(start, end))
            if end < L:
                blocks.append("")  # blank line between blocks
    else:
        blocks.extend(render_block(0, L))

    header = []
    if score is not None:
        header.append(str(score))

    return "\n".join(header + blocks)
