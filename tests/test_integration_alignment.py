# -*- coding: utf-8 -*-

"""Integration tests for alignment functionalities."""

import re
import math

import numpy as np
import pytest

from versalign.aligner import setup_aligner
from versalign.pairwise import calc_pairwise_alignment
from versalign.printing import format_alignment
from versalign.scoring import create_substituion_matrix_dynamically


purines = {"A", "G"}
pyrimidines = {"C", "T"}


def dna_ti_tv_compare(a: str, b: str) -> int:
    """DNA: match=+2, transition=-1, transversion=-2, gap=-3 (gaps will be introduced by the aligner)."""
    if a == b:
        return 2
    if (a in purines and b in purines) or (a in pyrimidines and b in pyrimidines):
        return -1  # transition
    return -2      # transversion


hydrophobic = set("AILMVFWY")
polar       = set("STNQ")
positive    = set("KRH")
negative    = set("DE")
special     = set("CGP")


def aa_class_compare(a: str, b: str) -> int:
    """Proteins by coarse chemical classes."""
    if a == b:
        return 3
    def cls(x: str) -> str:
        if x in hydrophobic: return "hydro"
        if x in polar:       return "polar"
        if x in positive:    return "pos"
        if x in negative:    return "neg"
        if x in special:     return "special"
        return "other"
    return 1 if cls(a) == cls(b) else -2


def test_pairwise_dna_ti_tv_single_symbols():
    """
    Integration: dynamic matrix (DNA ti/tv), setup_aligner, calc_pairwise_alignment.
    We use 1-character sequences so the aligned score equals the substitution score.
    """
    objs = list("ACGT-")
    sm, _ = create_substituion_matrix_dynamically(objs, compare=dna_ti_tv_compare)
    aligner = setup_aligner(sm, "global")

    # A vs A -> +2 (match)
    s, aln1, aln2 = calc_pairwise_alignment(aligner, list("A"), list("A"), gap_repr='-')
    assert s == 2
    assert aln1 == ["A"] and aln2 == ["A"]

    # A vs G -> -1 (transition)
    s, aln1, aln2 = calc_pairwise_alignment(aligner, list("A"), list("G"), gap_repr='-')
    assert s == -1
    assert len(aln1) == len(aln2) == 1

    # C vs A -> -2 (transversion)
    s, aln1, aln2 = calc_pairwise_alignment(aligner, list("C"), list("A"), gap_repr='-')
    assert s == -2
    assert len(aln1) == len(aln2) == 2


def test_pairwise_protein_class_single_symbols():
    """
    Integration: protein class scoring.
    A vs V are both hydrophobic -> +1 (same class, not identical)
    A vs D is hydro vs negative -> -2 (different class)
    """
    objs = list("ACDEFGHIKLMNPQRSTVWY-")
    sm, _ = create_substituion_matrix_dynamically(objs, compare=aa_class_compare)
    aligner = setup_aligner(sm, "global")

    # A vs V (hydro vs hydro) -> +1
    s, aln1, aln2 = calc_pairwise_alignment(aligner, list("A"), list("V"), gap_repr='-')
    assert s == 1
    assert aln1 == ["A"] and aln2 == ["V"]

    # A vs D (hydro vs negative) -> -2
    s, aln1, aln2 = calc_pairwise_alignment(aligner, list("A"), list("D"), gap_repr='-')
    assert s == -2
    assert aln1 == ["A", "-"] and aln2 == ["-", "D"]


def test_format_alignment_multichar_and_consensus_alignment():
    """
    Integration: verify the pretty-printer handles multi-char tokens, uneven name widths,
    and a strict consensus line that only marks exact, non-gap matches.
    We bypass the aligner here (feed aligned tokens directly) to make assertions crisp.
    """
    aln1 = ["ALA", "---", "GLY", "THR", "VAL"]
    aln2 = ["ALA", "SER", "GLY", "THR", "VAL"]
    names = ["short", "a-very-long-name"]

    formatted = format_alignment(
        [aln1, aln2],
        names=names,
        score=42.0,
        gap_repr="---",
        show_consensus=True,
    )

    # Contains the score header
    assert formatted.splitlines()[0].strip() == "42.0"

    # Contains both name headers (right-justified left gutter) and multi-char tokens
    assert "short > ALA --- GLY THR VAL" in formatted
    assert "a-very-long-name > ALA SER GLY THR VAL" in formatted

    # Consensus should be present only at columns where tokens are identical and non-gap.
    # Here: columns 1, 3, 4, 5 are identical; column 2 differs (--- vs SER).
    lines = formatted.splitlines()
    consensus_line = next(line for line in lines if line.strip().startswith("|") or line.rstrip().endswith("|"))
    # Count '|' characters
    assert consensus_line.count("|") == 4
