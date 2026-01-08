#!/usr/bin/env python3

"""Example script that shows how to align blocks of sequences using versalign."""

from versalign.scoring import create_substitution_matrix_dynamically
from versalign.aligner import setup_aligner
from versalign.blocking import calc_block_msa


def main() -> None:
    """Main function to demonstrate block alignment."""
    objs = list("ACGT-")
    sm, _ = create_substitution_matrix_dynamically(objs)
    aligner = setup_aligner(
        sm,
        "global",
        target_internal_open_gap_score=-5.0,
        target_left_open_gap_score=-5.0,
        target_right_open_gap_score=-5.0,
        query_internal_open_gap_score=-5.0,
        query_left_open_gap_score=-5.0,
        query_right_open_gap_score=-5.0,
    )

    a_blocks = [list("T"), list("ACG"), list("GGA"), list("AAAACGT")]
    b_blocks = [list("ACGG"), list("GGT")]
    c_blocks = [list("A"), list("AAAACCT"), list("T")]
    d_blocks = [list("AAAACCT")]
    e_blocks = [list("AAAACGT"), list("ACGGGA"),  list("T"), ]
    f_blocks = [list("T"), list("AGG"), list("GCA"), list("TGCAAAA")]
    g_blocks = [list("T")]

    msa = calc_block_msa(
        aligner,
        [a_blocks, b_blocks, c_blocks, d_blocks, e_blocks, f_blocks, g_blocks],
        gap_repr='-',
        allow_block_reverse=True
    )
    
    print("\nBlock MSA example:")
    for row, label in zip(msa.msa, msa.order):
        print(label[0], "|".join(["".join(block) for block in row]), sep="\t")


if __name__ == "__main__":
    main()
