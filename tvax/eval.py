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
    vaccine_seq: list,
    pathogen_seq: str,
    k: list,
) -> float:
    """
    Compute the fraction of k-mers in the pathogen sequence that are covered by the vaccine design
    """
    kmers = kmerise_simple(pathogen_seq, k)
    # Count the number of k-mers in the pathogen that are covered by the vaccine design
    # check if a given k-mer is a substring of any k-mers in the vaccine design
    n_cov = sum(
        [
            1
            for kmer in kmers
            if any([kmer in vaccine_kmer for vaccine_kmer in vaccine_seq])
        ]
    )
    return n_cov / len(kmers)


def compute_pathogen_coverages(
    vaccine_design: list,
    config: EpitopeGraphConfig,
    comp_df: pd.DataFrame = None,
    add_clades: bool = False,
    seqs_dict: dict = None,
) -> pd.DataFrame:
    """
    Computes the pathogen coverage of a vaccine designfor all of the target pathogen sequences
    """
    if seqs_dict is None:
        seqs_dict = load_fasta(config.fasta_path)
    # Compute the pathogen coverage for each pathogen
    path_cov = {
        seq_id: compute_pathogen_coverage(vaccine_design, seq, config.k)
        for seq_id, seq in seqs_dict.items()
    }
    # Create a dataframe of pathogen coverage with the sequence id and the pathogen coverage as columns with a numeric index
    path_cov_df = pd.DataFrame.from_dict(
        path_cov, orient="index", columns=["pathogen_coverage"]
    )
    path_cov_df["seq_name"] = path_cov_df.index
    path_cov_df = path_cov_df.reset_index(drop=True)
    # Convert pathogen coverage to percentage
    path_cov_df["pathogen_coverage"] = path_cov_df["pathogen_coverage"] * 100
    if add_clades:
        # Add clade information from comp df using the Sequence_id column
        path_cov_df["clade"] = path_cov_df["seq_name"].apply(
            lambda x: comp_df[comp_df["Sequence_id"] == x]["cluster"].values[0]
        )
        # Add colour from the comp df
        path_cov_df["colour"] = path_cov_df["seq_name"].apply(
            lambda x: comp_df[comp_df["Sequence_id"] == x]["colour"].values[0]
        )
        # Sort by clade
        path_cov_df = path_cov_df.sort_values(by="clade")
        path_cov_df = path_cov_df.reset_index(drop=True)
    return path_cov_df


def compute_av_pathogen_coverage(
    vaccine_design: list = None,
    config: EpitopeGraphConfig = None,
    path_cov_df: pd.DataFrame = None,
) -> float:
    """
    Computes the average pathogen coverage of a vaccine design i.e. the average fraction of kmers in the target pathogens that are covered by the vaccine design
    """
    if path_cov_df is None:
        path_cov_df = compute_pathogen_coverages(vaccine_design, config)
    return path_cov_df["pathogen_coverage"].mean()


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
