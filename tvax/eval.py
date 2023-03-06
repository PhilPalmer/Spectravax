import networkx as nx
import numpy as np
import pandas as pd

from tvax.config import EpitopeGraphConfig
from tvax.seq import load_fasta, kmerise_simple
from tvax.score import load_haplotypes, load_overlap, optivax_robust

"""
Evaluate vaccine designs.
"""


# def compute_av_score(
#     epitope_graph: nx.Graph,
#     vaccine_design: list,
#     score: str = "score",
# ) -> float:
#     """
#     Computes the average score of a vaccine design
#     :param score: String for the score to compute
#     :param vaccine_designs: List of vaccine designs
#     :param epitope_graph: Networkx graph of epitopes
#     :returns: Float for the average score
#     """
#     return np.mean([epitope_graph.nodes[e][score] for e in vaccine_design])


def compute_population_coverage(
    peptides: list, n_target: int, config: EpitopeGraphConfig
) -> float:
    """
    Computes the population coverage of a vaccine design
    """
    hap_freq, average_frequency = load_haplotypes(config)
    overlap_haplotypes = load_overlap(peptides, hap_freq, config)
    return optivax_robust(overlap_haplotypes, average_frequency, n_target, peptides)


def compute_pathogen_coverage(
    vaccine_design: list, config: EpitopeGraphConfig
) -> float:
    """
    Computes the pathogen coverage of a vaccine design
    """
    seqs_dict = load_fasta(config.fasta_path)
    n_cov = 0
    n_total = 0
    for seq_id, seq in seqs_dict.items():
        kmers = kmerise_simple(seq, config.k)
        n_cov += sum([1 for kmer in kmers if kmer in vaccine_design])
        n_total += len(kmers)
    return n_cov / n_total
