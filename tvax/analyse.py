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
from tvax.plot import plot_vaccine_design_pca, plot_population_coverage
from tvax.seq import path_to_kmers
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
        vaccine_kmers = path_to_kmers(vaccine_designs[0], config.k, epitope_graph)
        # Record PCA results
        fig, comp_df = plot_vaccine_design_pca(
            vaccine_designs, config, interactive=False
        )
        eigen_dist = compute_eigen_dist(comp_df)
        # Record pathogen coverage
        path_cov_df = compute_pathogen_coverages(vaccine_kmers, config)
        # path_cov = compute_av_pathogen_coverage(path_cov_df=path_cov_df)
        # Record population coverage for different values of n for Class I and II
        figs, pop_cov_df = plot_population_coverage(
            vaccine_kmers,
            n_targets,
            config=config,
        )

        # Select rows where ancestry is Average
        pop_cov_df = pop_cov_df[pop_cov_df["ancestry"] == "Average"]
        # Rename the mhc_type column to coverage_type
        # pop_cov_df = pop_cov_df.rename(columns={"mhc_type": "Coverage Type", "pop_cov": "Coverage (%)"})
        # Add the pathogen coverage data to the dataframe
        # n_targets = [f"n ≥ {n}" for n in n_targets]
        # for n in n_targets:
        #     pop_cov_df = pop_cov_df.append(
        #     {
        #         "ancestry": "Average",
        #         "Coverage Type": "Pathogen",
        #         "Coverage (%)": path_cov,
        #         "n_target": n,
        #     },
        #     ignore_index=True,
        # )
        # Add the antigen name to the dataframes
        path_cov_df["antigen"] = antigen
        pop_cov_df["antigen"] = antigen
        # Save the dataframes
        path_cov_dfs.append(path_cov_df)
        pop_cov_dfs.append(pop_cov_df)

    # Concatenate the dataframes
    path_cov_df = pd.concat(path_cov_dfs)
    pop_cov_df = pd.concat(pop_cov_dfs)

    return path_cov_df, pop_cov_df
