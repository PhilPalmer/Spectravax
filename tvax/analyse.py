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
from tvax.plot import plot_population_coverage, plot_pop_cov_lineplot
from tvax.seq import load_fasta, path_to_seq, seq_to_kmers
from typing import Tuple


"""
Run different analyses including a parameter sweep of potential vaccine designs.
"""

#################
# Parameter sweep
#################


def weights() -> list:
    """
    Returns a list of weights to be used in the parameter sweep.
    """
    return [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2]


def run_parameter_sweep(
    params: dict,
    w_cons: int = 1,
    w_mhc1_lst: list = weights(),
    w_mhc2_lst: list = weights(),
    n_targets: list = list(range(0, 11)),
    results_path: str = "data/param_sweep_betacov_n.csv",
) -> pd.DataFrame:
    """
    Run a parameter sweep over the weights for the population coverage.
    """
    # Create empty dataframe
    df = pd.DataFrame(
        {
            "vaccine_seq": [],
            "w_cons": [],
            "w_mhc1": [],
            "w_mhc2": [],
            "path_cov": [],
        }
    )
    for n_target in n_targets:
        df[f"pop_cov_mhc1_n_{n_target}"] = []
        df[f"pop_cov_mhc2_n_{n_target}"] = []

    for w_mhc1 in w_mhc1_lst:
        for w_mhc2 in w_mhc2_lst:
            params["weights"] = {
                "frequency": w_cons,
                "population_coverage_mhc1": w_mhc1,
                "population_coverage_mhc2": w_mhc2,
            }
            config = EpitopeGraphConfig(**params)
            epitope_graph = build_epitope_graph(config)
            vaccine_designs = design_vaccines(epitope_graph, config)
            vaccine_seq = [path_to_seq(path) for path in vaccine_designs][0]

            vaccine_designs = design_vaccines(epitope_graph, config)
            vaccine_kmers = seq_to_kmers(
                path_to_seq(vaccine_designs[0]), config.k, epitope_graph
            )
            # Compute the coverages
            path_cov_df = compute_pathogen_coverages(vaccine_kmers, config)
            path_cov = compute_av_pathogen_coverage(vaccine_kmers, config)
            # Record population coverage for different values of n for Class I and II
            figs, pop_cov_df = plot_population_coverage(
                vaccine_kmers,
                n_targets=n_targets,
                config=config,
            )
            # Select rows where ancestry is Average
            pop_cov_df = pop_cov_df[pop_cov_df["ancestry"] == "Average"]

            # Append to dataframe
            df = df.append(
                {
                    "vaccine_seq": vaccine_seq,
                    "w_cons": w_cons,
                    "w_mhc1": w_mhc1,
                    "w_mhc2": w_mhc2,
                    "path_cov": path_cov,
                    **{
                        f"pop_cov_mhc1_n_{n_target}": pop_cov_df[
                            (pop_cov_df["mhc_type"] == "mhc1")
                            & (pop_cov_df["n_target"] == f"n ≥ {n_target}")
                        ]["pop_cov"].values[0]
                        for n_target in n_targets
                    },
                    **{
                        f"pop_cov_mhc2_n_{n_target}": pop_cov_df[
                            (pop_cov_df["mhc_type"] == "mhc2")
                            & (pop_cov_df["n_target"] == f"n ≥ {n_target}")
                        ]["pop_cov"].values[0]
                        for n_target in n_targets
                    },
                },
                ignore_index=True,
            )
            df.to_csv(results_path, index=False)
    return df


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


#################################################################
# Compare CITVax-Fast vs CITVax-Robust methods for vaccine design
#################################################################


def params_to_update() -> list:
    """
    Returns a list of dictionaries of parameters to update.
    """
    return [
        {
            "n_target": 1,
            "robust": False,
        },
        {
            "n_target": 1,
            "robust": True,
        },
        {
            "n_target": 2,
            "robust": True,
        },
        {
            "n_target": 3,
            "robust": True,
        },
        {
            "n_target": 4,
            "robust": True,
        },
        {
            "n_target": 5,
            "robust": True,
        },
    ]


def compare_citvax_methods(
    params: dict,
    params_to_update: list = params_to_update(),
    results_path: str = "data/algo_results.csv",
) -> None:
    """
    Compare CITVax-Fast vs CITVax-Robust methods for vaccine design.
    """
    # Create empty dataframe
    df = pd.DataFrame(
        {
            "vaccine_seq": [],
            "n_target": [],
            "robust": [],
        }
    )

    for updated_params in params_to_update:
        params = {**params, **updated_params}

        config = EpitopeGraphConfig(**params)
        epitope_graph = build_epitope_graph(config)
        vaccine_designs = design_vaccines(epitope_graph, config)
        vaccine_seq = [path_to_seq(path) for path in vaccine_designs][0]
        n_target = config.n_target
        robust = config.robust
        # Append to dataframe
        df = df.append(
            {
                "vaccine_seq": vaccine_seq,
                "n_target": n_target,
                "robust": robust,
            },
            ignore_index=True,
        )
        # Save dataframe
        df.to_csv(results_path, index=False)

    # Update dataframe
    df["n_target"] = df["n_target"].astype(int)
    df["robust"] = df["robust"].astype(bool)
    df["method"] = (
        df["robust"].apply(lambda x: "Robust" if x else "Fast")
        + " n ≥"
        + df["n_target"].astype(str)
    )

    # Compute population coverage for each vaccine design at different n_target values
    pop_cov_dfs = []

    for index, row in df.iterrows():
        vaccine_kmers = seq_to_kmers(row.vaccine_seq, config.k, epitope_graph)
        figs, pop_cov_df = plot_population_coverage(
            vaccine_kmers,
            config=config,
        )
        pop_cov_df = pop_cov_df[pop_cov_df["ancestry"] == "Average"]
        pop_cov_df["method"] = row.method
        pop_cov_dfs.append(pop_cov_df)

    # Concatenate the dataframes
    pop_cov_df = pd.concat(pop_cov_dfs)

    # Plot population coverage
    plot_pop_cov_lineplot(pop_cov_df, mhc_type="mhc1", hue="method")
    plot_pop_cov_lineplot(pop_cov_df, mhc_type="mhc2", hue="method")
