#!/usr/bin/env python

from Bio import Align
from Bio.Align import substitution_matrices


def align_seqs(seq1, seq2, matrix="BLOSUM62"):
    """
    Align two sequences using the specified substitution matrix.
    :param seq1: First sequence AA string
    :param seq2: Second sequence AA string
    :param matrix: Substitution matrix name (default: BLOSUM62)
    :return: Alignment
    """
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load(matrix)
    return aligner.align(seq1, seq2)
