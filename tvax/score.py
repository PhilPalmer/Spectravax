import mhcflurry
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tvax.config import EpitopeGraphConfig
from tvax.pca_protein_rank import pca_protein_rank
from tvax.seq import assign_clades

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
    if config.weights.population_coverage:
        kmers_dict = add_population_coverage(kmers_dict, config)
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


def add_population_coverage(kmers_dict: dict, config: EpitopeGraphConfig) -> dict:
    """
    Add the population coverage of each peptide
    """
    # Load the data
    peptides = list(kmers_dict.keys())
    hap_freq, average_frequency = load_haplotypes(config)
    overlap_haplotypes = load_overlap(peptides, hap_freq, config)

    # Calculate and save the population coverage for each peptide
    for e in kmers_dict:
        kmers_dict[e]["population_coverage"] = optivax_robust(
            overlap_haplotypes, average_frequency, config.n_target, [e]
        )

    return kmers_dict


def load_haplotypes(config: EpitopeGraphConfig) -> tuple:
    """
    Load the haplotype frequency data
    """
    hap_freq = pd.read_pickle(config.hap_freq_mhc1_path)
    # The haplotype frequency file considers White, Asians, and Black:  We just average them all out here.
    # TODO: Add support for weighted averaging based on the target population
    average_frequency = pd.DataFrame(index=["Average"], columns=hap_freq.columns)
    average_frequency.loc["Average", :] = hap_freq.sum(axis=0) / len(hap_freq)
    return hap_freq, average_frequency


def load_overlap(peptides, hap_freq, config: EpitopeGraphConfig):
    """
    Load the overlap between peptides and haplotypes
    """
    if config.immune_scores_path and os.path.exists(config.immune_scores_path):

        overlap_haplotypes = pd.read_pickle(config.immune_scores_path)

    else:

        # Load the data
        mhc_data = pd.DataFrame(peptides)
        hla_alleles = pd.read_csv(config.mhc1_alleles_path, names=["allele"])

        # Predict the binding affinity
        mhc_data["key"] = 0
        hla_alleles["key"] = 0
        pmhc_aff = mhc_data.merge(hla_alleles, how="outer")
        pmhc_aff = pmhc_aff.drop(columns=["key"])
        pmhc_aff = pmhc_aff.rename(columns={0: "peptide"})
        predictions = predict_affinity(
            peptides=pmhc_aff["peptide"].to_list(), alleles=pmhc_aff["allele"].to_list()
        )
        pmhc_aff["affinity"] = predictions
        pmhc_aff["transformed_affinity"] = pmhc_aff["affinity"].apply(
            transform_affinity
        )
        pmhc_aff["loci"] = [x[:5] for x in pmhc_aff["allele"].values]

        # Pivot the data
        a74 = (
            pmhc_aff.loc[pmhc_aff["allele"].str.contains("HLA-A74")]
            .groupby("peptide")
            .agg("mean")
            .reset_index()
        )  # ['mean', 'count'])
        a74["loci"] = "HLA-A"
        a74["allele"] = "HLA-A74"
        c17 = (
            pmhc_aff.loc[pmhc_aff["allele"].str.contains("HLA-C17")]
            .groupby("peptide")
            .agg("mean")
            .reset_index()
        )
        c17["loci"] = "HLA-C"
        c17["allele"] = "HLA-C17"
        c18 = (
            pmhc_aff.loc[pmhc_aff["allele"].str.contains("HLA-C18")]
            .groupby("peptide")
            .agg("mean")
            .reset_index()
        )
        c18["loci"] = "HLA-C"
        c18["allele"] = "HLA-C18"
        pmhc_aff_pivot = pd.concat([pmhc_aff, a74, c17, c18], sort=False).pivot_table(
            index="peptide",
            columns=["loci", "allele"],
            values="transformed_affinity",
        )
        pmhc_aff_pivot.to_pickle(config.raw_affinity_path)

        # Binarize the data to create hits
        pmhc_aff_pivot = pmhc_aff_pivot.applymap(
            lambda x: 1 if x > config.affinity_cutoff else 0
        )
        # Drop the loci from the columns and get the data down to a Single Index of alleles for the columns
        pmhc_aff_pivot = pmhc_aff_pivot.droplevel("loci", axis=1)
        # Run the add_hits_across_haplotypes function on the desired data.
        overlap_haplotypes = add_hits_across_haplotypes(pmhc_aff_pivot, hap_freq)

        # Save to file
        overlap_haplotypes.to_pickle(config.immune_scores_path)

    return overlap_haplotypes


def predict_affinity(peptides: list, alleles: list) -> np.ndarray:
    """
    Predict the binding affinity of each peptide-HLA pair
    """
    predictor = mhcflurry.Class1AffinityPredictor().load()
    predictions = predictor.predict(peptides=peptides, alleles=alleles)
    return predictions


def transform_affinity(x: float, a_min: float = None, a_max: float = 50000) -> float:
    """
    Compute logistic-transformed binding affinity
    """
    x = np.clip(x, a_min=a_min, a_max=a_max)
    return 1 - np.log(x) / np.log(a_max)


def add_hits_across_haplotypes(
    df_binarized: pd.DataFrame, df_hap_freq: pd.DataFrame
) -> pd.DataFrame:
    """
    Add the hits across haplotypes
    Convert a DataFrame of peptides vs alleles to a DataFrame of peptides vs haplotypes
    with the number of alleles that are present in each haplotype
    """
    unique_columns = df_binarized.columns.unique()
    df_overlap = pd.DataFrame(index=df_binarized.index, columns=df_hap_freq.columns)
    for pept in df_overlap.index:
        for col in df_overlap.columns:
            temp_sum = 0
            for hap_type in col:
                if hap_type in unique_columns:
                    temp_sum += int(df_binarized.at[pept, hap_type])
            df_overlap.at[pept, col] = temp_sum
    return df_overlap


def optivax_robust(over, hap, thresh, set_of_peptides):
    """
    Evaluates the objective value for optivax robust
    EvalVax-Robust over haplotypes!!!
    """

    def _my_filter(my_input, thresh):
        """
        A simple function that acts as the indicator variable in the pseudocode
        """
        for index in range(len(my_input)):
            if my_input[index] >= thresh:
                my_input[index] = 1
            else:
                my_input[index] = 0
        return my_input

    num_of_haplotypes = len(over.columns)
    total_overlays = np.zeros(num_of_haplotypes, dtype=int)

    for pept in set_of_peptides:

        total_overlays = total_overlays + np.array(over.loc[pept, :])

    filtered_overlays = _my_filter(total_overlays, thresh)

    return np.sum(filtered_overlays * np.array(hap))


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
