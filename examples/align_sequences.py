#!/usr/bin/env python3

"""Example script that shows how to align two or more sequences using versalign."""

from versalign.aligner import setup_aligner
from versalign.msa import calc_msa
from versalign.pairwise import calc_pairwise_alignment
from versalign.printing import format_alignment
from versalign.scoring import create_substituion_matrix_dynamically


def main() -> None:
    """Main function to demonstrate sequence alignment."""
    
    # --- CASE 2 ---
    # Simple equality scoring ACGT
    objs = list("ACGT-")
    sm, _ = create_substituion_matrix_dynamically(objs)
    aligner = setup_aligner(sm, "global")

    # Define example sequences
    seq1, seq2, seq3 = list("ACGT"), list("ACGGGGGT"), list("ATCCT")

    # Pairwise alignment
    s, aln1, aln2 = calc_pairwise_alignment(aligner, seq1, seq2, gap_repr='-')
    print("\nPairwise alignment (equality scoring ACGT):")
    print(format_alignment([aln1, aln2], names=["seq1", "seq2"], score=s))

    # Multiple sequence alignment
    seqs = [seq1, seq2, seq3]
    lbls = ["seq1", "seq2", "seq3"]
    msa, order = calc_msa(aligner, seqs, gap_repr='-')
    lbls_ordered = [lbls[i] for i in order]
    print("\nMultiple sequence alignment (equality scoring ACGT):")
    print(format_alignment(msa, names=lbls_ordered))

    # --- CASE 2 ---
    # DNA: match=+2, transition=-1, transversion=-2, gap=-3
    purines = {"A", "G"}
    pyrimidines = {"C", "T"}

    def dna_ti_tv_compare(a: str, b: str) -> int:
        if a == b: return 2
        if "-" in (a, b): return -3  # if you include gap in objs
        if (a in purines and b in purines) or (a in pyrimidines and b in pyrimidines):
            return -1  # transition
        return -2      # transversion

    objs = list("ACGT-")
    sm, _ = create_substituion_matrix_dynamically(objs, compare=dna_ti_tv_compare)
    aligner = setup_aligner(sm, "global")

    # Define example sequences
    seq1, seq2 = list("ACGTAG"), list("ATCCTAG")

    # Pairwise alignment
    s, aln1, aln2 = calc_pairwise_alignment(aligner, seq1, seq2, gap_repr='-')
    print("\nPairwise alignment (DNA scoring):")
    print(format_alignment([aln1, aln2], names=["seq1", "seq2"], score=s))

    # --- CASE 3 ---
    # Proteins by coarse chemical classes
    hydrophobic = set("AILMVFWY")
    polar       = set("STNQ")
    positive    = set("KRH")
    negative    = set("DE")
    special     = set("CGP")

    def aa_class_compare(a: str, b: str) -> int:
        if a == b: return 3
        if "-" in (a, b): return -5
        def cls(x: str) -> str:
            if x in hydrophobic: return "hydro"
            if x in polar:       return "polar"
            if x in positive:    return "pos"
            if x in negative:    return "neg"
            if x in special:     return "special"
            return "other"
        return 1 if cls(a) == cls(b) else -2

    objs = list("ACDEFGHIKLMNPQRSTVWY-")
    sm, _ = create_substituion_matrix_dynamically(objs, compare=aa_class_compare)
    aligner = setup_aligner(sm, "global")

    # Define example sequences
    seq1 = list("ACDEFGHIKLMNPQRSTVWY")
    seq2 = list("ACDFGHIKLMNQRSTVWA")

    # Pairwise alignment
    s, aln1, aln2 = calc_pairwise_alignment(aligner, seq1, seq2, gap_repr='-')
    print("\nPairwise alignment (AA class scoring):")
    print(format_alignment([aln1, aln2], names=["seq1", "seq2"], score=s))

    # --- CASE 4 ---
    # Compare arbitrary objects (e.g., residues) via a property (mass)

    class Residue:
        def __init__(self, name: str, mass: float):
            self.name = name
            self.mass = mass

    def residue_compare(a: Residue | str, b: Residue | str) -> float:
        if "-" in (a, b): return -5.0
        return -abs(a.mass - b.mass)
    
    def label_fn (r: Residue | str) -> str:
        return r if r == "-" else r.name

    residues = [Residue("X", 100.0), Residue("Y", 101.3), Residue("Z", 97.8), "-"]
    sm, _ = create_substituion_matrix_dynamically(residues, compare=residue_compare, label_fn=label_fn)
    aligner = setup_aligner(sm, "global", label_fn=label_fn)

    # Define example sequences
    seq1 = [residues[0], residues[1], residues[2], residues[0]]  # X Y Z X
    seq2 = [residues[1], residues[2], residues[2], residues[0]]  # Y Z Z X
    seq3 = [residues[0], residues[0]]                            # X X
    seq4 = [residues[2], residues[0], residues[0]]               # Z X X

    # Pairwise alignment
    s, aln1, aln2 = calc_pairwise_alignment(aligner, seq1, seq2, gap_repr='-')
    print("\nPairwise alignment (residue mass scoring):")
    print(format_alignment([aln1, aln2], names=["seq1", "seq2"], score=s))

    # Multiple sequence alignment
    seqs = [seq1, seq2, seq3, seq4]
    lbls = ["seq1", "seq2", "seq3", "seq4"]
    msa, order = calc_msa(aligner, seqs, gap_repr='-')
    lbls_ordered = [lbls[i] for i in order]
    print("\nMultiple sequence alignment (residue mass scoring):")
    print(format_alignment(msa, names=lbls_ordered))

    # Multiple sequence alignment with seq1 fixed as center star; makes sure that rest is aligned to seq1
    center_star_idx = 0
    msa, order = calc_msa(aligner, seqs, gap_repr='-', center_star=center_star_idx)
    lbls_ordered = [lbls[i] for i in order]
    print("\nMultiple sequence alignment with fixed center star (residue mass scoring):")
    print(format_alignment(msa, names=lbls_ordered))


if __name__ == "__main__":
    main()
