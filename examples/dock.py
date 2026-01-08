#!/usr/bin/env python3

"""Example script that shows how to dock a block-sequence to a sequence using versalign."""

from versalign.scoring import create_substitution_matrix_dynamically
from versalign.aligner import setup_aligner
from versalign.docking import dock_against_target


def main() -> None:
    """Main function to demonstrate block alignment."""

    target = ["DEC", "TRP", "ASN", "ASP", "THR", "GLY", "ORN", "ASP", "ALA", "ASP", "GLY", "SER", "3ME", "KYN"]
    query = [
        ["KYN", "3ME", "SER"],
        ["TRP", "ASN", "ASP", "THR"],
        ["GLY", "ORN", "ASP", "ALA", "ASP", "GLY"],
        ["D01", "D01", "D01", "D01", "ACE"],
    ]

    # create single set of unique items in target and query
    objs = set(["---"])
    objs.update(target)
    for block in query:
        objs.update(block)
    sm, _ = create_substitution_matrix_dynamically(list(objs))
    aligner = setup_aligner(
        sm,
        "global",
        target_internal_open_gap_score=-5.0,
        target_left_open_gap_score=-5.0,
        target_right_open_gap_score=-5.0,
        query_internal_open_gap_score=-5.0,
        query_left_open_gap_score=-1.0,
        query_right_open_gap_score=-1.0,
    )

    res = dock_against_target(
        aligner=aligner,
        target=target,
        candidates=query,
        gap_repr="---",
        allow_block_reverse=True,
        strategy="nonoverlap",
    )
    print(res.placements[0].center_aln)
    for p in res.placements:
        print(p.block_aln)


if __name__ == "__main__":
    main()
