import os
import pandas as pd

from tvax.config import EpitopeGraphConfig, Weights
from tvax.design import design_vaccines
from tvax.eval import (
    compute_eigen_dist,
    compute_population_coverage,
    compute_pathogen_coverages,
    compute_av_pathogen_coverage,
)
from tvax.graph import build_epitope_graph
from tvax.plot import plot_population_coverage
from tvax.seq import load_fasta, path_to_seq, seq_to_kmers
from typing import Tuple


"""
Run different analyses including a parameter sweep of potential vaccine designs.
"""

#################
# Parameter sweep
#################


def run_parameter_sweep(
    n_clusters: list,
    pop_cov_weights: list,
    results_path: str,
    config: EpitopeGraphConfig,
) -> pd.DataFrame:
    """
    Run a parameter sweep over the number of clusters and the population coverage weights.
    """
    if not os.path.exists(results_path):
        param_sweep_dict = {
            "n_cluster": [],
            "pop_cov_weight": [],
            "pop_cov": [],
            "path_cov": [],
        }

        for n in n_clusters:
            for weight in pop_cov_weights:
                config.weights = Weights(frequency=1, population_coverage_mhc1=weight)
                config.n_clusters = n
                epitope_graph = build_epitope_graph(config)
                vaccine_designs = design_vaccines(epitope_graph, config)
                pop_cov = compute_population_coverage(vaccine_designs[0], 5, config)
                path_cov = compute_av_pathogen_coverage(vaccine_designs[0], config)
                param_sweep_dict["n_cluster"].append(n)
                param_sweep_dict["pop_cov_weight"].append(weight)
                param_sweep_dict["pop_cov"].append(pop_cov)
                param_sweep_dict["path_cov"].append(path_cov)

        # Process the data
        param_sweep_df = pd.DataFrame(param_sweep_dict)
        param_sweep_df["pop_cov"] = param_sweep_df["pop_cov"] * 100
        param_sweep_df["path_cov"] = param_sweep_df["path_cov"] * 100
        param_sweep_df["av_cov"] = (
            param_sweep_df["pop_cov"] + param_sweep_df["path_cov"]
        ) / 2
        param_sweep_df.to_csv(results_path, index=False)

    else:
        # TODO: Check that the results are the same as the ones in the file
        param_sweep_df = pd.read_csv(results_path)

    return param_sweep_df


###################################
# Compare different antigen designs
###################################


def antigens_dict():
    """
    Return a dictionary of antigen names and paths to fasta files.
    """
    return {
        "Betacoronavirus_N": {
            "fasta_path": "data/input/sar_mer_nuc_protein.fasta",
            "results_dir": "data/results",
        },
        "Betacoronavirus_nsp12": {
            "fasta_path": "data/input/nsp12_protein.fa",
            "results_dir": "data/results_nsp12",
        },
        "Sarbecovirus_RBD": {
            "fasta_path": "data/input/sarbeco_protein_RBD.fa",
            "results_dir": "data/results_sarbeco_rbd",
        },
        "Merbecovirus_RBD": {
            "fasta_path": "data/input/merbeco_protein_RBD.fa",
            "results_dir": "data/results_merbeco_rbd",
        },
        # "Embecovirus_S": {
        #     "fasta_path": "data/input/embeco_protein_spike.fa",
        #     "results_dir": "data/results_embeco_spike",
        # },
        # "Embecovirus_N": {
        #     "fasta_path": "data/input/embeco_protein_np.fa",
        #     "results_dir": "data/results_embeco_np",
        # },
        "Orthomyxoviridae_H3": {
            "fasta_path": "data/input/H3_human.fa",
            "results_dir": "data/results_h3",
        },
        "Orthomyxoviridae_N1": {
            "fasta_path": "data/input/N1_protein.fst",
            "results_dir": "data/results_n1",
        },
    }


def compute_coverages(
    kmers: list,
    config: EpitopeGraphConfig,
    antigen: str,
    n_targets: list = list(range(0, 11)),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the pathogen and population coverages for a given set of k-mers from an antigen.
    """
    path_cov_df = compute_pathogen_coverages(kmers, config)
    # Record population coverage for different values of n for Class I and II
    figs, pop_cov_df = plot_population_coverage(
        kmers,
        n_targets=n_targets,
        config=config,
    )
    # Select rows where ancestry is Average
    pop_cov_df = pop_cov_df[pop_cov_df["ancestry"] == "Average"]
    # Add the antigen name to the dataframes
    path_cov_df["antigen"] = antigen
    pop_cov_df["antigen"] = antigen
    return path_cov_df, pop_cov_df


def compare_antigens(
    params: dict,
    antigens_dict: dict = antigens_dict(),
    n_targets: list = list(range(0, 11)),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare different antigen vaccine designs
    """
    # Create empty lists to store the results
    path_cov_dfs = []
    pop_cov_dfs = []

    # Design vaccines for each antigen
    for antigen, antigen_dict in antigens_dict.items():
        antigen = antigen.replace("_", " ").replace("RBD", "S RBD")
        params["fasta_path"] = antigen_dict["fasta_path"]
        params["results_dir"] = antigen_dict["results_dir"]
        config = EpitopeGraphConfig(**params)
        epitope_graph = build_epitope_graph(config)
        vaccine_designs = design_vaccines(epitope_graph, config)
        vaccine_kmers = seq_to_kmers(
            path_to_seq(vaccine_designs[0]), config.k, epitope_graph
        )
        # Compute the coverages
        path_cov_df, pop_cov_df = compute_coverages(
            vaccine_kmers,
            config,
            antigen,
            n_targets,
        )
        # Save the dataframes
        path_cov_dfs.append(path_cov_df)
        pop_cov_dfs.append(pop_cov_df)

    # Concatenate the dataframes
    path_cov_df = pd.concat(path_cov_dfs)
    pop_cov_df = pd.concat(pop_cov_dfs)

    return path_cov_df, pop_cov_df


#######################
# Compare to wild-types
#######################


def betacov_strains_dict() -> dict:
    """
    Return a dictionary of betacoronavirus strains and their names.
    """
    return {
        "QHR63308_1_nucleocapsid_protein_Bat_coronavirus_RaTG13_Rhinolophus_affinis": "RaTG13",
        "YP_009825061_1_nucleocapsid_protein_SARS_coronavirus_Tor2_Homo_sapiens": "SARS",
        "YP_009047211.1": "MERS",
    }


def h3_strains_dict() -> dict:
    """
    Returns a dictionary of influenza virus H3 strains and their names.
    """
    return {
        "AIE52620_Human_2011": "A/Victoria/361/2011",
        "AJK01027_Human_2009": "A/Victoria/210/2009",
        "AJK02592_Human_2009": "A/Perth/16/2009",
        "ABW80978_Human_2005": "A/Wisconsin/67/2005",
    }


def compare_to_wts(
    config: EpitopeGraphConfig,
    strains_dict: dict = betacov_strains_dict,
    n_targets: list = list(range(0, 11)),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare vaccine
    """
    # Design vaccine
    epitope_graph = build_epitope_graph(config)
    vaccine_designs = design_vaccines(epitope_graph, config)

    # Load all seqs
    seqs_dict = load_fasta(config.fasta_path)

    # Filter to keep seqs of interest
    seqs_dict = {
        seq_id: seqs_dict[seq_id]
        for seq_id in seqs_dict
        if seq_id in strains_dict.keys()
    }

    # Add the vaccine design to the dicts
    seqs_dict["citvax_design"] = path_to_seq(vaccine_designs[0])
    strains_dict["citvax_design"] = "CITVax Design"

    # Create empty lists to store the results
    path_cov_dfs = []
    pop_cov_dfs = []

    for seq_id, seq in seqs_dict.items():
        seq_name = strains_dict[seq_id]
        # Convert seqs to k-mers
        kmers = seq_to_kmers(seq, config.k, epitope_graph)
        # Compute metrics for each seq
        path_cov_df, pop_cov_df = compute_coverages(kmers, config, seq_name, n_targets)
        # Append to lists
        path_cov_dfs.append(path_cov_df)
        pop_cov_dfs.append(pop_cov_df)

    # Concatenate the dataframes
    path_cov_df = pd.concat(path_cov_dfs)
    pop_cov_df = pd.concat(pop_cov_dfs)

    return path_cov_df, pop_cov_df
