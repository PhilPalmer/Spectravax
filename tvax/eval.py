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
    peptides: list,
    n_target: int,
    config: EpitopeGraphConfig,
    mhc_type: str,
    average_frequency: pd.DataFrame = None,
    overlap_haplotypes: pd.DataFrame = None,
) -> float:
    """
    Computes the population coverage of a vaccine design i.e. the fraction of the population that is predicted to have ≥ n peptide-HLA hits produced by the vaccine
    """
    if average_frequency is None or overlap_haplotypes is None:
        hap_freq_path = (
            config.hap_freq_mhc1_path
            if mhc_type == "mhc1"
            else config.hap_freq_mhc2_path
        )
        hap_freq, average_frequency = load_haplotypes(hap_freq_path)
        overlap_haplotypes = load_overlap(peptides, hap_freq, config, mhc_type)
    return optivax_robust(overlap_haplotypes, average_frequency, n_target, peptides)


def compute_pathogen_coverage(
    vaccine_design: list, config: EpitopeGraphConfig
) -> float:
    """
    Computes the pathogen coverage of a vaccine design i.e. the fraction of kmers in the pathogen that are covered by the vaccine design
    """
    seqs_dict = load_fasta(config.fasta_path)
    n_cov = 0
    n_total = 0
    for seq_id, seq in seqs_dict.items():
        kmers = kmerise_simple(seq, config.k)
        n_cov += sum([1 for kmer in kmers if kmer in vaccine_design])
        n_total += len(kmers)
    return n_cov / n_total


def compute_eigen_dist(
    comp_df: pd.DataFrame,
    vaccine_id: str = "vaccine_design",
    pca_cols: list = ["PCA1", "PCA2", "PCA3"],
) -> float:
    """
    Compute the average distance between the vaccine design and the other sequences in the PCA space
    Assuming a normal distribution, this is the number of standard deviations away from the mean of the other sequences
    """
    vac_pca_scores = comp_df[comp_df["Sequence_id"] == vaccine_id][pca_cols].to_numpy()
    seq_pca_scores = comp_df[comp_df["Sequence_id"] != vaccine_id][pca_cols].to_numpy()
    eigen_dists = np.linalg.norm(vac_pca_scores - seq_pca_scores, axis=1)
    return np.mean(eigen_dists)
