import mhcflurry
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tvax.pca_protein_rank import pca_protein_rank

"""
Calculate the total weighted score for each potential T-cell epitope (PTE).
"""


def add_scores(
    kmers_dict: dict, clades_dict: dict, N: int, config: EpitopeGraphConfig
) -> dict:
    """
    Add the scores and weights to each k-mer.
    """
    # TODO: Do this in a more efficient way
    if config.weights.frequency:
        kmers_dict = add_frequency_score(kmers_dict, N)
    if (
        config.weights.processing
        or config.weights.strong_mhc_binding
        or config.weights.weak_mhc_binding
    ):
        kmers_dict = add_mhcflurry_scores(kmers_dict, config)
    kmers_dict = add_clade_weights(kmers_dict, clades_dict, N)
    kmers_dict = add_score(kmers_dict, config)
    return kmers_dict


def add_score(kmers_dict: dict, config: EpitopeGraphConfig) -> dict:
    """
    Add the total weighted score to each k-mer.
    """
    # Calculate the total score for each k-mer by summing the weighted scores
    # TODO: Check if dividing by the sum of the weights is necessary
    for kmer in kmers_dict:
        kmers_dict[kmer]["score"] = sum(
            [kmers_dict[kmer][name] * val for name, val in config.weights]
        )
        # Multiple the total score by the clade weight
        kmers_dict[kmer]["score"] *= kmers_dict[kmer]["clade_weight"]
    # Min-max scale the scores
    scores = [kmers_dict[kmer]["score"] for kmer in kmers_dict]
    minmax_scaler = MinMaxScaler()
    scores = minmax_scaler.fit_transform(np.array(scores).reshape(-1, 1)).reshape(-1)
    for i, kmer in enumerate(kmers_dict):
        kmers_dict[kmer]["score"] = scores[i]
    return kmers_dict


def add_frequency_score(kmers_dict: dict, N: int) -> dict:
    """
    Add frequency score to k-mers.
    """
    for kmer in kmers_dict:
        kmers_dict[kmer]["frequency"] = kmers_dict[kmer]["count"] / N
    return kmers_dict


def add_mhcflurry_scores(kmers_dict: dict, config: EpitopeGraphConfig) -> dict:
    """
    Add MHCflurry scores to k-mers.
    """
    # Define vars
    agg_dict = {
        "processing": "mean",
        "strong_mhc_binding": "sum",
        "weak_mhc_binding": "sum",
    }
    cached_scores = False
    cols = ["processing", "strong_mhc_binding", "weak_mhc_binding"]
    peptides = list(kmers_dict.keys())

    # Check if the scores are already cached
    if config.immune_scores_path and os.path.exists(config.immune_scores_path):

        df = pd.read_csv(config.immune_scores_path)
        # TODO: Check if the alleles and thresholds are the same
        if set(df["peptide"]) == set(peptides):
            cached_scores = True

    # Only compute MHCflurry predictions if not already cached
    if not cached_scores:
        dfs = []
        median_scaler = MedianScaler()
        minmax_scaler = MinMaxScaler()
        predictor = mhcflurry.Class1PresentationPredictor.load()
        # TODO: Check if model needs to be downloaded !mhcflurry-downloads fetch models_class1_presentation

        # Compute the scores for all peptides and alleles
        for allele in config.alleles:
            df = predictor.predict(
                peptides, [allele], include_affinity_percentile=True, verbose=0
            )
            # Determine if the peptide binds to the MHC (based on the percentile threshold)
            df["weak_mhc_binding"] = df["affinity_percentile"].apply(
                lambda x: 1 if x < config.weak_percentile_threshold else 0
            )
            df["strong_mhc_binding"] = df["affinity_percentile"].apply(
                lambda x: 1 if x < config.strong_percentile_threshold else 0
            )
            df = df.rename(
                columns={"best_allele": "allele", "processing_score": "processing"}
            )
            dfs.append(df)

        # Concatenate the dataframes
        df = pd.concat(dfs, ignore_index=True)
        # Aggregate the scores for each peptide across the alleles
        df = df.groupby("peptide").agg(agg_dict).reset_index()
        # Convert MHC binding to the fraction of bound alleles
        df["weak_mhc_binding"] = df["weak_mhc_binding"] / len(config.alleles)
        df["strong_mhc_binding"] = df["strong_mhc_binding"] / len(config.alleles)
        # Scale the scores
        df[["weak_mhc_binding", "strong_mhc_binding"]] = median_scaler.fit_transform(
            df[["weak_mhc_binding", "strong_mhc_binding"]]
        )
        df[["processing"]] = minmax_scaler.fit_transform(df[["processing"]])

        df.to_csv(config.immune_scores_path, index=False)

    # Update the k-mers dictionary with the scores
    mhcflurry_dict = df[cols + ["peptide"]].set_index("peptide").to_dict()
    for kmer in kmers_dict:
        for col in cols:
            kmers_dict[kmer][col] = mhcflurry_dict[col][kmer]

    return kmers_dict


def assign_clades(seqs_dict: dict, config: EpitopeGraphConfig) -> dict:
    """
    Assign clades to sequences.
    """
    if config.equalise_clades:
        # Perform multiple sequence alignment
        msa_path = msa(config.fasta_path, config.msa_path)
        # Assign clades to sequences
        clades_dict, comp_df = pca_protein_rank(
            msa_path, n_clusters=config.n_clusters, plot=False
        )
    else:
        clades_dict = {seq_id: 1 for seq_id in seqs_dict}
    return clades_dict


def add_clade_weights(kmers_dict: dict, clades_dict: dict, N: int) -> dict:
    """
    Add clades and the average clade weight (total number of sequences / number of sequences in each clade) to k-mers.
    """
    # Compute the clade weights for each clade
    clade_weight_dict = {}
    for seq_id, clade in clades_dict.items():
        if clade not in clade_weight_dict:
            clade_weight_dict[clade] = 1
        else:
            clade_weight_dict[clade] += 1
    clade_weight_dict = {
        clade: N / clade_weight_dict[clade] for clade in clade_weight_dict
    }
    # Add the clade weights to the k-mers
    for kmer, d in kmers_dict.items():
        kmers_dict[kmer]["clade_weight"] = np.mean(
            [clade_weight_dict[clade] for clade in d["clades"]]
        )
    return kmers_dict


class MedianScaler:
    """
    Scale the data to the range [min_value, max_value] using the median and standard deviation.
    """

    def __init__(self, min_value=0, max_value=1):
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X):
        self.median_ = np.median(X)
        self.std_ = np.std(X)

    def transform(self, X):
        z_scores = (X - self.median_) / self.std_
        scaled = (z_scores - np.min(z_scores)) / (np.max(z_scores) - np.min(z_scores))
        scaled = scaled * (self.max_value - self.min_value) + self.min_value
        return scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
