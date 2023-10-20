from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import networkx as nx
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns
import subprocess

from Bio import SeqIO
from dna_features_viewer import GraphicFeature, GraphicRecord
from sklearn.preprocessing import MinMaxScaler
from tvax.config import AnalysesConfig, EpitopeGraphConfig, Weights
from tvax.design import design_vaccines
from tvax.eval import (
    compute_eigen_dist,
    compute_population_coverage,
    compute_pathogen_coverages,
    compute_av_pathogen_coverage,
)
from tvax.graph import build_epitope_graph, f
from tvax.plot import (
    plot_antigens_comparison,
    plot_binding_criteria_eval,
    plot_calibration_curves,
    plot_path_exp_dps,
    plot_kmer_filtering,
    plot_kmer_graphs,
    plot_population_coverage,
    plot_pop_cov_lineplot,
    plot_scores_distribution,
    plot_vaccine_design_pca,
)
from tvax.score import (
    add_frequency_score,
    add_population_coverage,
    remove_host_kmers,
    remove_kmers,
    transform_affinity,
    load_haplotypes,
    load_overlap,
    optivax_robust,
)
from tvax.seq import (
    assign_clades,
    kmerise,
    kmerise_simple,
    load_fasta,
    path_to_seq,
    seq_to_kmers,
)
from typing import Tuple


"""
Run different analyses including a parameter sweep of potential vaccine designs.
"""

###########################################
# Define main fucntion for running analyses
###########################################


# TODO: Move these into config
def antigens_dict():
    """
    Return a dictionary of antigen names and paths to fasta files.
    """
    return {
        "Influenza A M1": {
            "fasta_path": "data/results_flua_m1/Seq_Preprocessing/flua_m1.fasta",
            "results_dir": "data/results_flua_m1/",
        },
        "Betacoronavirus N": {
            "fasta_path": "data/input/sar_mer_nuc_protein_clustered_95.fasta",
            "results_dir": "data/results",
            "n_clusters": 7,
        },
        "Coronavirinae nsp12": {
            "fasta_path": "data/input/nsp12_protein.fa",
            "results_dir": "data/results_nsp12",
        },
        "Sarbecovirus S RBD": {
            "fasta_path": "data/input/sarbeco_protein_RBD.fa",
            "results_dir": "data/results_sarbeco_rbd",
        },
        "Merbecovirus S RBD": {
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
        "Orthomyxoviridae H3": {
            "fasta_path": "data/input/H3_human.fa",
            "results_dir": "data/results_h3",
        },
        "Orthomyxoviridae N1": {
            "fasta_path": "data/input/N1_protein.fst",
            "results_dir": "data/results_n1",
        },
    }


def run_analyses(
    config: AnalysesConfig,
    params: dict,
    antigens_dict: dict = antigens_dict(),
) -> None:
    """
    Run the analyses specified in the config.
    """
    # Load the k-mer graphs for each antigen to be used in some of the analyses
    if (
        config.run_kmer_graphs
        or config.run_scores_distribution
        or config.run_compare_antigens
        or config.run_population_coverage
    ):
        if config.antigen_graphs_pkl.exists():
            print("Loading antigen graphs from pickle file")
            with open(config.antigen_graphs_pkl, "rb") as file:
                antigen_graphs = pickle.load(file)
        else:
            print("Constructing antigen graphs")
            antigen_graphs = construct_antigen_graphs(
                params, config.antigen_graphs_pkl, antigens_dict
            )
    if config.run_kmer_filtering:
        n_filtered_kmers_df = compute_n_filtered_kmers(params)
        plot_kmer_filtering(n_filtered_kmers_df, config.kmer_filtering_fig)
    if config.run_scores_distribution:
        # TODO: update this func to use the antigen_graphs dict
        if config.scores_distribution_json.exists():
            with open(config.scores_distribution_json, "r") as file:
                scores_dict = json.load(file)
        # TODO: Fix this step - I was getting "KeyError: 'MSDNGPQSN'" during the graph construction
        else:
            scores_dict = compute_antigen_scores(
                params, config.scores_distribution_json
            )
        plot_scores_distribution(scores_dict, config.scores_distribution_fig)
    if config.run_binding_criteria:
        binding_criteria_df = load_and_preprocess_data(
            config.binding_criteria_csv, config.binding_crtieria_dir
        )
        plot_binding_criteria_eval(binding_criteria_df, config.binding_criteria_fig)
    if config.run_netmhc_calibration:
        run_netmhc_calibration(
            config.netmhc_calibration_csv,
            config.netmhc_calibration_raw_mhc1_csv,
            config.netmhc_calibration_raw_mhc2_csv,
            config.netmhc_calibration_fig,
        )
    if config.run_kmer_graphs:
        print("Running k-mer graphs...")
        antigen_dict = antigens_dict[config.kmer_graph_antigen]
        params["fasta_path"] = antigen_dict["fasta_path"]
        params["results_dir"] = antigen_dict["results_dir"]
        kmergraph_config = EpitopeGraphConfig(**params)
        G = antigen_graphs[config.kmer_graph_antigen]
        Q = design_vaccines(G, kmergraph_config)
        plot_kmer_graphs(G, Q, config.kmer_graphs_fig)
    if config.run_compare_antigens:
        print("Comparing antigens...")
        params["equalise_clades"] = True
        path_cov_df, pop_cov_df = compare_antigens(
            params,
            antigens_dict,
            antigen_graphs,
        )
        compute_antigen_summary_metrics(
            path_cov_df, pop_cov_df, config.compare_antigens_csv
        )
        print("Plotting antigen comparison...")
        plot_antigens_comparison(
            path_cov_df,
            pop_cov_df,
            config.compare_antigens_fig,
        )

    ###########################
    # Antigen specific analyses
    ###########################

    if config.run_population_coverage or config.run_pathogen_coverage:
        print(f"Design vaccine for {config.antigen}")
        # TODO: Save all vaccine designs rather than computing one here
        antigen_dict = antigens_dict[config.antigen]
        for param, value in antigen_dict.items():
            params[param] = value
        kmergraph_config = EpitopeGraphConfig(**params)
        # G = antigen_graphs[config.antigen]
        G = build_epitope_graph(kmergraph_config)
        Q = design_vaccines(G, kmergraph_config)
    if config.run_population_coverage:
        print("Running population coverage...")
        generate_and_plot_population_coverage(
            csv_path=config.population_coverage_csv,
            svg_path=config.population_coverage_fig,
            vaccine_designs=Q,
            config=kmergraph_config,
            epitope_graph=G,
            n_targets=list(range(0, 81)),
        )
    if config.run_pathogen_coverage:
        print("Running pathogen coverage...")
        fig, comp_df = plot_vaccine_design_pca(
            Q, config=kmergraph_config, interactive=False, plot_type="2D"
        )
        generate_and_plot_pathogen_coverage(
            csv_path=config.pathogen_coverage_csv,
            svg_path=config.pathogen_coverage_fig,
            config=kmergraph_config,
            comp_df=comp_df,
            vaccine_designs=Q,
            epitope_graph=G,
        )


#################
# Parameter sweep
#################


def weights() -> list:
    """
    Returns a list of weights to be used in the parameter sweep.
    """
    return [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2]


def run_parameter_sweep_parallel(
    params: dict,
    w_cons: int = 1,
    w_mhc1_lst: list = weights(),
    w_mhc2_lst: list = weights(),
    n_targets: list = list(range(0, 11)),
    results_path: str = "data/param_sweep_h3.csv",
    num_threads: int = 4,
) -> pd.DataFrame:
    """
    Run a parameter sweep over the weights for the population coverage.
    """

    def process_weight(w_mhc1, w_mhc2):
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

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each combination of w_mhc1 and w_mhc2
        futures = [
            executor.submit(process_weight, w_mhc1, w_mhc2)
            for w_mhc1 in w_mhc1_lst
            for w_mhc2 in w_mhc2_lst
        ]

        # Process the results as they become available
        for future in as_completed(futures):
            result = future.result()
            df = df.append(result, ignore_index=True)
            df.to_csv(results_path, index=False)

    return df


###################################
# Compare different antigen designs
###################################


def compute_coverages(
    kmers: list,
    config: EpitopeGraphConfig,
    antigen: str,
    n_targets: list = list(range(0, 11)),
    G: nx.Graph = None,
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
        G=G,
    )
    # Select rows where ancestry is Average
    pop_cov_df = pop_cov_df[pop_cov_df["ancestry"] == "Average"]
    # Add the antigen name to the dataframes
    path_cov_df["antigen"] = antigen
    pop_cov_df["antigen"] = antigen
    return path_cov_df, pop_cov_df


def compare_antigens(
    params: dict,
    antigens_dict: dict,
    antigen_graphs: dict,
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
        params["fasta_path"] = antigen_dict["fasta_path"]
        params["results_dir"] = antigen_dict["results_dir"]
        config = EpitopeGraphConfig(**params)
        epitope_graph = antigen_graphs[antigen]
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
            G=epitope_graph,
        )
        # Save the dataframes
        path_cov_dfs.append(path_cov_df)
        pop_cov_dfs.append(pop_cov_df)

    # Concatenate the dataframes
    path_cov_df = pd.concat(path_cov_dfs)
    pop_cov_df = pd.concat(pop_cov_dfs)

    return path_cov_df, pop_cov_df


def construct_antigen_graphs(
    params: dict,
    out_path: str,
    antigens_dict: dict = antigens_dict(),
) -> dict:
    # Create an empty dictionary to store the graphs
    antigen_graphs = {}
    # Construct k-mer graph for each antigen
    for antigen, antigen_dict in antigens_dict.items():
        params["fasta_path"] = antigen_dict["fasta_path"]
        params["results_dir"] = antigen_dict["results_dir"]
        config = EpitopeGraphConfig(**params)
        epitope_graph = build_epitope_graph(config)
        antigen_graphs[antigen] = epitope_graph
    # Write to pickle file
    with open(out_path, "wb") as f:
        pickle.dump(antigen_graphs, f)
    return antigen_graphs


def compute_antigen_scores(
    params: dict,
    out_path: Path,
    antigens_dict: dict = antigens_dict(),
) -> dict:
    """
    Compute the scores for each antigen
    """
    # Create an empty dictionary to store the scores
    scores_dict = {}

    # Compute antigen graphs
    antigen_graphs = construct_antigen_graphs(params, antigens_dict)
    config = EpitopeGraphConfig(**params)

    # Get the scores for each antigen
    for antigen, G in antigen_graphs.items():
        scores_dict[antigen] = {
            "f_cons": [],
            "f_mhc1": [],
            "f_mhc2": [],
            "f": [],
            "f_clade_adjusted": [],
            "f_scaled": [],
        }

        for _, data in G.nodes(data=True):
            f_cons = data["frequency"]
            f_mhc1 = data["population_coverage_mhc1"]
            f_mhc2 = data["population_coverage_mhc2"]
            f = sum(
                [data[name] * val for name, val in config.weights if val > 0]
            ) / sum(config.weights.dict().values())
            w_clade = data["clade_weight"]
            f_clade_adjusted = f * w_clade
            # Save the scores
            scores_dict[antigen]["f_cons"].append(f_cons)
            scores_dict[antigen]["f_mhc1"].append(f_mhc1)
            scores_dict[antigen]["f_mhc2"].append(f_mhc2)
            scores_dict[antigen]["f"].append(f)
            scores_dict[antigen]["f_clade_adjusted"].append(f_clade_adjusted)
        # Min-max scale the scores
        minmax_scaler = MinMaxScaler()
        scores_dict[antigen]["f_scaled"] = (
            minmax_scaler.fit_transform(
                np.array(scores_dict[antigen]["f_clade_adjusted"]).reshape(-1, 1)
            )
            .reshape(-1)
            .tolist()
        )
    # Save the scores to JSON
    with open(out_path, "w") as f:
        json.dump(scores_dict, f)
    return scores_dict


def compute_n_filtered_kmers(
    params: dict,
    antigens_dict: dict = antigens_dict(),
) -> pd.DataFrame:
    """
    Compute the number of filtered k-mers for each antigen
    """
    # Create empty lists to store the results
    n_filtered_kmers = {
        "antigen": [],
        "n_raw_kmers": [],
        "n_rare_kmers": [],
        "n_host_kmers": [],
        "n_passed_kmers": [],
    }
    # Compute the metrics for each antigen
    for antigen, data in antigens_dict.items():
        n_filtered_kmers["antigen"].append(antigen)
        # Number of raw k-mers
        params["fasta_path"] = data["fasta_path"]
        params["equalise_clades"] = False
        config = EpitopeGraphConfig(**params)
        seqs_dict = load_fasta(config.fasta_path)
        clades_dict = assign_clades(seqs_dict, config)
        kmers_dict = kmerise(seqs_dict, clades_dict, config.k)
        kmers_dict = add_frequency_score(kmers_dict, len(seqs_dict))
        n_raw_kmers = len(kmers_dict)
        n_filtered_kmers["n_raw_kmers"].append(n_raw_kmers)

        # Number of host k-mers
        kmers_after_host_removal = remove_host_kmers(
            kmers_dict, config.human_proteome_path, config.k
        )
        n_host_kmers = n_raw_kmers - len(kmers_after_host_removal)
        n_filtered_kmers["n_host_kmers"].append(n_host_kmers)

        # Number of rare (nonconserved) k-mers
        kmers_after_rare_removal = remove_kmers(
            kmers_after_host_removal, config.conservation_threshold
        )
        n_passed_kmers = len(kmers_after_rare_removal)
        n_rare_kmers = n_raw_kmers - n_host_kmers - n_passed_kmers
        n_filtered_kmers["n_rare_kmers"].append(n_rare_kmers)

        # Number of passed k-mers
        n_filtered_kmers["n_passed_kmers"].append(n_passed_kmers)

    # Create a dataframe
    n_filtered_kmers_df = pd.DataFrame(n_filtered_kmers)

    return n_filtered_kmers_df


#########################################################
# Evaluate the (population) coverage for a single antigen
#########################################################


def generate_and_plot_population_coverage(
    csv_path: str,
    svg_path: str,
    vaccine_designs: list,
    config: object,
    epitope_graph: object,
    n_targets: list,
) -> None:
    vaccine_kmers = seq_to_kmers(
        path_to_seq(vaccine_designs[0]), config.k, epitope_graph
    )
    cov_df = load_or_compute_population_coverage(
        csv_path, vaccine_kmers, config, epitope_graph
    )

    host_cov_df = cov_df[cov_df["cov_type"] == "host"]
    cov_df = cov_df[cov_df["cov_type"] == "host+pathogen"]

    trans_host_cov_df = transform_population_data(host_cov_df)
    trans_cov_df = transform_population_data(cov_df)

    plot_population_coverage(trans_host_cov_df, trans_cov_df, cov_df, svg_path)


def load_or_compute_population_coverage(
    csv_path: str, vaccine_kmers: list, config: object, epitope_graph: object
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"Computing population coverage for {config.prefix}")
        host_cov_df = generate_population_coverage_data(vaccine_kmers, config=config)
        cov_df = generate_population_coverage_data(
            vaccine_kmers, config=config, G=epitope_graph
        )
        host_cov_df["cov_type"] = "host"
        cov_df["cov_type"] = "host+pathogen"
        cov_df = pd.concat([host_cov_df, cov_df])
        host_cov_df.to_csv(csv_path, index=False)
    else:
        print(f"Loading population coverage for {config.prefix} from {csv_path}")
        cov_df = pd.read_csv(csv_path)
    return cov_df


def generate_population_coverage_data(
    vaccine_design: list = None,
    n_targets: list = list(range(0, 81)),
    mhc_types: list = ["mhc1", "mhc2"],
    ancestries: list = ["Asian", "Black", "White", "Average"],
    config: EpitopeGraphConfig = None,
    G: nx.Graph = None,
) -> pd.DataFrame:
    """
    Compute population coverage for a given vaccine design.
    """
    peptides = vaccine_design
    pop_cov_dict = {"ancestry": [], "mhc_type": [], "n_target": [], "pop_cov": []}
    kmers_dict = None if G is None else dict(G.nodes(data=True))

    # Preprocessing
    for mhc_type in mhc_types:
        hap_freq_path = (
            config.hap_freq_mhc1_path
            if mhc_type == "mhc1"
            else config.hap_freq_mhc2_path
        )
        hap_freq, average_frequency = load_haplotypes(hap_freq_path)
        if "Asians" in hap_freq.index:
            hap_freq.index = hap_freq.index.str.replace("Asians", "Asian")

        for anc in ancestries:
            if anc == "Average":
                anc_freq = average_frequency
            else:
                anc_freq = hap_freq.loc[anc].copy()
            overlap_haplotypes = load_overlap(peptides, anc_freq, config, mhc_type)
            for n_target in n_targets:
                pop_cov = optivax_robust(
                    overlap_haplotypes, anc_freq, n_target, peptides, kmers_dict
                )
                pop_cov_dict["ancestry"].append(anc)
                pop_cov_dict["mhc_type"].append(mhc_type)
                pop_cov_dict["n_target"].append(n_target)
                pop_cov_dict["pop_cov"].append(pop_cov)

    pop_cov_df = pd.DataFrame(pop_cov_dict)
    pop_cov_df["pop_cov"] = pop_cov_df["pop_cov"] * 100

    # Sort by ancestry but put average last
    ancestries = sorted(ancestries)
    ancestries.remove("Average")
    ancestries.append("Average")
    pop_cov_df["ancestry"] = pd.Categorical(
        pop_cov_df["ancestry"], categories=ancestries, ordered=True
    )

    return pop_cov_df


def transform_population_data(
    df: pd.DataFrame, group_col="ancestry", group_cols=["ancestry", "mhc_type"]
) -> pd.DataFrame:
    """
    Transforms the population coverage data to represent the fraction
    of the population with exactly n peptide-HLA hits.
    """
    df = df.copy()
    df["n_target"] = df["n_target"].astype(int)
    transformed_data = []
    grouped = df.groupby(group_cols)
    for group_keys, group in grouped:
        group = group.sort_values(by="n_target").reset_index(drop=True)
        exact_pop_cov = [
            group["pop_cov"][i] - group["pop_cov"][i + 1]
            if i + 1 < len(group)
            else group["pop_cov"][i]
            for i in range(len(group))
        ]
        data_dict = {
            group_col: group_keys[0],  # By default, the first key is group_col
            "mhc_type": group_keys[1],  # By default, the second key is mhc_type
            "n_target": None,
            "pop_cov": None,
        }

        # If there are more keys, it means vaccine_coverage is present.
        if len(group_keys) > 2:
            data_dict["vaccine_coverage"] = group_keys[2]

        for n_target, pop_cov in zip(group["n_target"], exact_pop_cov):
            data_dict["n_target"] = n_target
            data_dict["pop_cov"] = max(0, pop_cov)
            transformed_data.append(data_dict.copy())

    return pd.DataFrame(transformed_data)


#####################################################
# Evaluate the pathogen coverage for a single antigen
#####################################################


def compute_coverage(
    config: EpitopeGraphConfig,
    comp_df: pd.DataFrame,
    vaccine_designs: list,
    epitope_graph: nx.Graph,
    mhc_types: list = ["mhc1", "mhc2"],
    vaccine_coverage: list = [False, True],
) -> pd.DataFrame:
    """
    Compute coverage for all sequences in the input components dataframe.
    """

    # Initialise dict
    pop_cov_dict = {
        "mhc_type": [],
        "seq_id": [],
        "pop_cov": [],
        "cov_type": [],
        "n_target": [],
        "vaccine_coverage": [],
    }

    # Get all sequences and k-mers
    seqs_dict = load_fasta(config.fasta_path)
    seqs_dict["vaccine_design"] = path_to_seq(vaccine_designs[0])
    kmers_dict = dict(epitope_graph.nodes(data=True))

    # Get the vaccine sequence and k-mers
    vaccine_seq = seqs_dict["vaccine_design"]
    vaccine_kmers = seq_to_kmers(vaccine_seq, config.k, epitope_graph)

    # For each MHC type
    for mhc_type in mhc_types:
        hap_freq_path = (
            config.hap_freq_mhc1_path
            if mhc_type == "mhc1"
            else config.hap_freq_mhc2_path
        )
        hap_freq, average_frequency = load_haplotypes(hap_freq_path)
        overlap_haplotypes = load_overlap(kmers_dict, hap_freq, config, mhc_type)

        # For each sequence in comp_df
        for i, row in comp_df.iterrows():
            seq_id = row["Sequence_id"]
            seq = seqs_dict[seq_id]

            for coverage in sorted(vaccine_coverage):
                kmers = seq_to_kmers(seq, config.k, epitope_graph)
                if coverage:
                    kmers = [kmer for kmer in kmers if kmer in vaccine_kmers]

                # Compute the coverage
                pop_cov = 1
                n_target = 0
                while pop_cov > 0:
                    pop_cov = optivax_robust(
                        overlap_haplotypes,
                        average_frequency,
                        n_target,
                        kmers,
                        kmers_dict,
                    )
                    n_target += 1
                    pop_cov_dict["mhc_type"].append(mhc_type)
                    pop_cov_dict["seq_id"].append(seq_id)
                    pop_cov_dict["pop_cov"].append(pop_cov)
                    pop_cov_dict["cov_type"].append("cov")
                    pop_cov_dict["n_target"].append(n_target)
                    pop_cov_dict["vaccine_coverage"].append(coverage)
                vax_cov = "VAX" if coverage else "WTs"
                print(f"{seq_id} {mhc_type} {vax_cov} {pop_cov}")

    pop_cov_df = pd.DataFrame(pop_cov_dict)
    return pop_cov_df


def compute_exp_dps(
    pop_cov_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    mhc_types: list = ["mhc1", "mhc2"],
) -> pd.DataFrame:
    """
    Add the expected # of displayed peptides to the components dataframe.
    """
    pop_cov_df = transform_population_data(
        pop_cov_df,
        group_col="seq_id",
        group_cols=["seq_id", "mhc_type", "vaccine_coverage"],
    )
    # For each sequence in the dataframe compute the expected number of displayed peptides
    for mhc_type in mhc_types:
        for seq_id in pop_cov_df["seq_id"].unique():
            for coverage in pop_cov_df["vaccine_coverage"].unique():
                cov_label = "vax" if coverage else "wts"
                seq_df = pop_cov_df[
                    (pop_cov_df["seq_id"] == seq_id)
                    & (pop_cov_df["mhc_type"] == mhc_type)
                    & (pop_cov_df["vaccine_coverage"] == coverage)
                ]
                count = seq_df["n_target"].values
                freq = seq_df["pop_cov"].values
                exp = (count * freq).sum()
                comp_df.loc[
                    comp_df["Sequence_id"] == seq_id,
                    f"E(#DPs)_{cov_label}_cov_{mhc_type}",
                ] = exp
    return comp_df


def generate_and_plot_pathogen_coverage(
    csv_path: str,
    svg_path: str,
    config: EpitopeGraphConfig,
    comp_df: pd.DataFrame,
    vaccine_designs: list,
    epitope_graph: nx.Graph,
):
    """
    Generate and plot the expected number of displayed peptides for each target input sequence.
    """
    # Check if the coverage dataframe already exists
    if os.path.exists(csv_path):
        comp_df = pd.read_csv(csv_path)
    else:
        pop_cov_df = compute_coverage(
            config,
            comp_df,
            vaccine_designs,
            epitope_graph,
            mhc_types=["mhc1", "mhc2"],
            vaccine_coverage=[True, False],
        )
        comp_df = compute_exp_dps(pop_cov_df, comp_df, mhc_types=["mhc1", "mhc2"])
        comp_df.to_csv(csv_path, index=False)
    plot_path_exp_dps(comp_df, svg_path)


###############################################################
# Evaluate the binding criteria using SARS-CoV-2 stability data
###############################################################


def load_and_preprocess_data(
    filepath: str,
    workdir: str,
    mhc2_allele: str = "DRB1_0401",
    num_threads: int = 4,
):
    """Load and preprocess the experimental peptide-MHC binding data."""
    xls = pd.ExcelFile(filepath)
    dfs = []
    sheet_names = xls.sheet_names

    # Process each sheet except the "BINDERS" sheet
    for sheet in sheet_names:
        if sheet != "BINDERS":
            df = xls.parse(sheet)
            df["allele"] = sheet
            dfs.append(df)

    # Concatenate the dataframes
    concat_df = pd.concat(dfs, ignore_index=True)
    concat_df["binder"] = 0
    binders_df = xls.parse("BINDERS")

    # Update the binder column based on the "BINDERS" sheet
    for _, row in binders_df.iterrows():
        peptides = row["peptide_seq"]
        alleles = row["allele"].split(", ")
        for allele in alleles:
            concat_df.loc[
                (concat_df["peptide_seq"] == peptides)
                & (concat_df["allele"] == allele),
                "binder",
            ] = 1

    # Match the mhc2_allele format with netMHCIIpan
    concat_df.loc[
        concat_df["allele"] == mhc2_allele.replace("_", "-"), "allele"
    ] = mhc2_allele
    # Rename the columns
    concat_df = concat_df.rename(columns={"peptide_seq": "peptide"})

    # MHCflurry predictions
    mhc1_df = concat_df.copy()
    mhc1_df = mhc1_df[mhc1_df["allele"] != mhc2_allele]
    peptides = list(mhc1_df["peptide"].unique())
    alleles = {
        a: [a] for a in mhc1_df["allele"].unique()
    }  # f"HLA-{a[0]}*{a[1:3]}:{a[3:]}"
    mhcflurry_preds = predict_mhcflurry(peptides, alleles)

    # NetMHC predictions
    peptides_df = pd.DataFrame(concat_df["peptide"].unique(), columns=["peptide"])
    peptides_path = f"{workdir}/peptides.txt"
    mhc_alleles = {
        "mhc1": [f"HLA-{a}" for a in mhc1_df["allele"].unique()],
        "mhc2": [mhc2_allele],
    }

    # Setup
    dfs = []
    peptides_df["peptide"].to_csv(peptides_path, index=False, header=False)
    os.makedirs(workdir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each combination of mhc_class and allele
        futures = [
            executor.submit(
                predict_affinity_netmhcpan, peptides_path, allele, mhc_type, workdir
            )
            for mhc_type, alleles in mhc_alleles.items()
            for allele in alleles
        ]

        # Process the results as they become available
        for future in as_completed(futures):
            result = future.result()
            dfs.append(result)

    netmhc_preds = pd.concat(dfs)
    netmhc_preds["allele"] = netmhc_preds["allele"].str.replace("HLA-", "")
    netmhc_preds["nM"] = 50000 ** (1 - netmhc_preds["BA-score"])
    netmhc_preds["EL-Rank"] = 1 - netmhc_preds["EL_Rank"] / 100

    # Combine the dataframes
    concat_df = combine_dataframes(
        concat_df, mhcflurry_preds, netmhc_preds, mhc2_allele
    )

    return concat_df


def predict_mhcflurry(peptides, alleles):
    """Make binding predictions with MHCflurry."""
    import mhcflurry

    predictor = mhcflurry.Class1PresentationPredictor().load()
    mhcflurry_preds = predictor.predict(peptides=peptides, alleles=alleles)
    mhcflurry_preds = mhcflurry_preds.rename(columns={"best_allele": "allele"})
    mhcflurry_preds["affinity-score"] = 1 - np.log(
        mhcflurry_preds["affinity"]
    ) / np.log(50000)

    return mhcflurry_preds


def combine_dataframes(
    concat_df, mhcflurry_preds, netmhc_preds, mhc2_allele: str = "DRB1_0401"
):
    """Combine dataframes and ensemble the MHCFlurry and NetMHCpan scores."""
    # Add the presentation score from the mhcflurry_preds to the concat_df
    concat_df = concat_df.merge(
        mhcflurry_preds[
            [
                "peptide",
                "allele",
                "affinity",
                "affinity-score",
                "processing_score",
                "presentation_score",
            ]
        ],
        on=["peptide", "allele"],
        how="left",
    )

    # Add the EL-score and BA-score from the netmhc_preds to the concat_df
    concat_df = concat_df.merge(
        netmhc_preds[["peptide", "allele", "nM", "BA-score", "EL-score", "EL-Rank"]],
        on=["peptide", "allele"],
        how="left",
    )

    # Rename the columns
    cols = {
        "affinity": "MHCFlurry2.0.6_BA",
        "affinity-score": "MHCFlurry2.0.6_BA_score",
        "processing_score": "MHCFlurry2.0.6_processing_score",
        "presentation_score": "MHCFlurry2.0.6_presentation_score",
        "nM": "NetMHCpan4.1_BA",
        "BA-score": "NetMHCpan4.1_BA_score",
        "EL-score": "NetMHCpan4.1_EL_score",
        "EL-Rank": "NetMHCpan4.1_EL_Rank",
    }
    concat_df = concat_df.rename(columns=cols)

    # Ensemble the MHCFlurry and NetMHCpan scores
    concat_df["Ensemble_BA"] = concat_df[["MHCFlurry2.0.6_BA", "NetMHCpan4.1_BA"]].mean(
        axis=1
    )
    concat_df.loc[concat_df["allele"] == mhc2_allele, "Ensemble_BA"] = np.nan
    concat_df["Ensemble_BA_score"] = 1 - np.log(concat_df["Ensemble_BA"]) / np.log(
        50000
    )
    concat_df["Ensemble_EL_score"] = concat_df[
        ["MHCFlurry2.0.6_presentation_score", "NetMHCpan4.1_EL_score"]
    ].mean(axis=1)
    concat_df.loc[concat_df["allele"] == mhc2_allele, "Ensemble_BA"] = np.nan

    return concat_df


#########################################################
# MHC Calibration Curves for NetMHCpan validation dataset
#########################################################


def predict_netmhcpan(netmhc_df, peptides_dir, alleles, mhc_type, num_threads):
    """Run NetMHCpan/NetMHCIIpan and return binding predictions."""

    dfs = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for allele in alleles:
            allele_peptides_df = netmhc_df[netmhc_df["allele"] == allele][
                ["peptide"]
            ].drop_duplicates()
            allele_peptides_path = os.path.join(peptides_dir, f"peptides_{allele}.txt")
            allele_peptides_df["peptide"].to_csv(
                allele_peptides_path, index=False, header=False
            )

            futures.append(
                executor.submit(
                    predict_affinity_netmhcpan,
                    allele_peptides_path,
                    allele,
                    mhc_type,
                    peptides_dir,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            dfs.append(result)

    netmhc_preds = pd.concat(dfs)

    return netmhc_preds


def predict_affinity_netmhcpan(peptides_path, allele, mhc_type, results_dir):
    """Run NetMHCpan/NetMHCIIpan for a single allele."""

    # Run NetMHCpan/NetMHCIIpan
    netmhcpan_cmd = "netMHCpan" if mhc_type == "mhc1" else "netMHCIIpan"

    preds_dir = f"{results_dir}/net{mhc_type}_preds"
    os.makedirs(preds_dir, exist_ok=True)

    peptides_flag = "-p" if mhc_type == "mhc1" else "-inptype 1 -f"
    outpath = f"{preds_dir}/{allele}.xls"

    cmd = f"{netmhcpan_cmd} {peptides_flag} {peptides_path} -a {allele} -BA -xls -xlsfile {outpath}"
    subprocess.run(cmd, shell=True)

    # Process predictions
    df = pd.read_csv(outpath, delimiter="\t", skiprows=[0])
    df["mhc_class"] = mhc_type
    df["allele"] = allele
    df = df.rename(
        columns={
            "Peptide": "peptide",
            "Core": "core",
            "Score_BA": "BA-score",
            "Rank_BA": "BA_Rank",
            "Score": "EL-score",
            "Rank": "EL_Rank",
        }
    )

    return df


def run_netmhc_calibration(
    netmhc_eval_path: str, netmhc1_eval_path: str, netmhc2_eval_path: str, out_path: str
):
    """
    Run NetMHCpan/NetMHCIIpan on the validation dataset, save the predictions and plot the results.
    """

    if os.path.exists(netmhc_eval_path):
        print("Loading preprocessed NetMHCpan evaluation data...")
        netmhc_preds = pd.read_csv(netmhc_eval_path)
    else:
        print("Preprocessed data not found, generating from raw datasets...")

        # glob_pattern = "/home/phil/Downloads/NetMHCIIpan_train/test_EL*.txt"
        # out_path = "data/input/analyses/NetMHCIIpan_EL_eval.txt"
        # alleles_file = "../optivax/scoring/MHC2_allele_mary_cleaned.txt"
        # allele_names_path = "~/Downloads/NetMHCIIpan_train/allelelist.txt"
        # allele_names_df = pd.read_csv(allele_names_path, sep=' ', header=None, names=['allele_name', 'allele_short_name'])
        # allele_name_to_short_name = allele_names_df.set_index('allele_name')['allele_short_name'].to_dict()
        # alleles_df = pd.read_csv(alleles_file, sep='\t', header=None, names=['allele'])
        # dfs = []

        # # Read dfs
        # for file in glob.glob(glob_pattern):
        #     print(file)
        #     df = pd.read_csv(file, sep="\t", names=["peptide","binder","allele_name","core"])
        #     dfs.append(df)

        # # Process dfs
        # df = pd.concat(dfs)
        # df['allele_name'] = df['allele_name'].map(allele_name_to_short_name).combine_first(df['allele_name'])
        # df = df.rename(columns={"allele_name": "allele"})
        # df = df[df["allele"].isin(alleles_df["allele"].values)]

        # df[["peptide","binder","allele"]].to_csv(out_path, sep=" ", index=False, header=False)

        # Load and preprocess
        netmhc1_df = pd.read_csv(
            netmhc1_eval_path, sep=" ", names=["peptide", "binder", "allele"]
        )
        netmhc2_df = pd.read_csv(
            netmhc2_eval_path, sep=" ", names=["peptide", "binder", "allele"]
        )

        netmhc1_df["mhc_type"] = "mhc1"
        netmhc2_df["mhc_type"] = "mhc2"

        netmhc_df = pd.concat([netmhc1_df, netmhc2_df])

        # Filter peptides
        netmhc_df = netmhc_df[netmhc_df["peptide"].str.len().isin([9, 15])]

        # Define parameters
        peptides_dir = "data/outputs/data/netmhc_bind_probs"
        num_threads = 4

        mhc_alleles = {
            "mhc1": netmhc_df[netmhc_df["mhc_type"] == "mhc1"]["allele"]
            .unique()
            .tolist(),
            "mhc2": netmhc_df[netmhc_df["mhc_type"] == "mhc2"]["allele"]
            .unique()
            .tolist(),
        }

        # Make predictions
        netmhc_preds = predict_netmhcpan(
            netmhc_df, peptides_dir, mhc_alleles, num_threads
        )

        # Process predictions
        netmhc_preds["nM"] = 50000 ** (1 - netmhc_preds["BA-score"])
        netmhc_preds["EL_Rank"] = 1 - netmhc_preds["EL_Rank"] / 100

        # Add back missing columns
        netmhc_preds = netmhc_preds.merge(
            netmhc_df, on=["peptide", "allele"], how="left"
        )

        # Save preprocessed data
        netmhc_preds.to_csv(netmhc_eval_path, index=False)

    # Plot calibration curves
    palette = sns.color_palette("colorblind", n_colors=netmhc_preds["allele"].nunique())
    plot_calibration_curves(netmhc_preds, out_path, palette)


def cons_vs_path_cov(
    params: dict, antigens_dict: dict = antigens_dict()
) -> pd.DataFrame:
    """
    Compare the conservation score and population coverage for all antigens
    """
    antigen_graphs = construct_antigen_graphs(params, antigens_dict)
    config = EpitopeGraphConfig(**params)

    # For each antigen, for each k-mer compute the conservation and pathogen coverage
    antigens_cons_path_cov = {
        "antigen": [],
        "kmer": [],
        "conservation": [],
        "pathogen_coverage": [],
    }

    # Iterate over the antigen_graphs and antigens_dict at the same time
    for (antigen, G), (_, data) in zip(antigen_graphs.items(), antigens_dict.items()):
        print(antigen)
        params["fasta_path"] = data["fasta_path"]
        params["equalise_clades"] = False
        config = EpitopeGraphConfig(**params)
        seqs_dict = load_fasta(config.fasta_path)
        for kmer, data in G.nodes(data=True):
            path_cov_df = compute_pathogen_coverages(
                [kmer],
                config,
                seqs_dict=seqs_dict,
            )
            path_cov = compute_av_pathogen_coverage([kmer], config, path_cov_df)
            antigens_cons_path_cov["antigen"].append(antigen)
            antigens_cons_path_cov["kmer"].append(kmer)
            antigens_cons_path_cov["conservation"].append(data["frequency"])
            antigens_cons_path_cov["pathogen_coverage"].append(path_cov)

    antigens_cons_path_cov_df = pd.DataFrame.from_dict(antigens_cons_path_cov)
    return antigens_cons_path_cov_df


def compute_antigen_summary_metrics(
    path_cov_df: pd.DataFrame = None,
    pop_cov_df: pd.DataFrame = None,
    out_path=None,
    antigens_dict: dict = antigens_dict(),
) -> pd.DataFrame:
    """
    Compute summary metrics for each antigen
    """

    def _add_pop_cov_col(antigen, mhc, n_target):
        return [
            pop_cov_df[
                (pop_cov_df["antigen"] == antigen)
                & (pop_cov_df["mhc_type"] == mhc)
                & (pop_cov_df["n_target"] == n_target)
            ]["pop_cov"].values[0]
            for antigen in summary_df["antigen"]
        ]

    # Create empty lists to store the results
    n_seqs = []
    median_seq_len = []
    # Compute the metrics for each antigen
    for antigen, antigen_dict in antigens_dict.items():
        fasta_path = antigen_dict["fasta_path"]
        # Compute the number of sequences
        n_seqs.append(len(list(SeqIO.parse(fasta_path, "fasta"))))
        # Compute the median sequence length
        seq_lens = [len(seq) for seq in SeqIO.parse(fasta_path, "fasta")]
        median_seq_len.append(np.median(seq_lens))
    # Create a dataframe
    summary_df = pd.DataFrame(
        {
            "antigen": list(antigens_dict.keys()),
            "n_seqs": n_seqs,
            "median_seq_len": median_seq_len,
        }
    )
    if path_cov_df is not None and pop_cov_df is not None:
        summary_df["mean_pathogen_coverage"] = [
            path_cov_df[path_cov_df["antigen"] == antigen]["pathogen_coverage"].mean()
            for antigen in summary_df["antigen"]
        ]
        for mhc in ["mhc1", "mhc2"]:
            for n in list(range(0, 11)):
                summary_df[f"pop_cov_{mhc}_n{n}"] = _add_pop_cov_col(
                    antigen, mhc, f"n ≥ {n}"
                )

    # Write to file
    summary_df.to_csv(out_path, index=False)
    return summary_df


###############################################
# Comparison to wild-types and existing methods
###############################################


def compare_vaccine_design(
    params: dict,
    fasta_path: Path = Path("data/mice_mhcs/NP_designs_June2023.txt"),
    updated_immune_scores_mhc1_path: Path = Path(
        "data/results/MHC_Binding/sar_mer_nuc_protein_immune_scores_ensemble_updated.pkl"
    ),
    updated_immune_scores_mhc2_path: Path = Path(
        "data/results/MHC_Binding/sar_mer_nuc_protein_immune_scores_netmhcii_updated.pkl"
    ),
    n_targets: list = list(range(0, 11)),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare sequences coverage metrics e.g. CITVax vaccine design to wild-type sequences and existing methods.
    """
    config = EpitopeGraphConfig(**params)
    seqs_dict = load_fasta(fasta_path)

    # Split into k-mers
    clades_dict = {seq_id: 1 for seq_id in seqs_dict}
    kmers_dict = kmerise(seqs_dict, clades_dict, config.k)

    # Get k-mers without existing predictions
    overlap_haplotypes = pd.read_pickle(config.immune_scores_mhc1_path)
    kmers_dict = {
        kmer: v
        for kmer, v in kmers_dict.items()
        if kmer not in overlap_haplotypes.index
    }

    # Get existing immune score paths
    immune_scores_mhc1_path = config.immune_scores_mhc1_path
    immune_scores_mhc2_path = config.immune_scores_mhc2_path

    # Predict binding using MHCflurry and NetMHCpan for these k-mers
    params_to_change = {
        "prefix": "betacov_n_updated",
        "results_dir": Path("data/results_betacov_n_updated"),
    }
    # Update the params dict using params_to_change
    params.update(params_to_change)
    config = EpitopeGraphConfig(**params)

    # Predict binding using MHCflurry and NetMHCpan for these k-mers
    kmers_dict = add_population_coverage(kmers_dict, config, "mhc1")
    kmers_dict = add_population_coverage(kmers_dict, config, "mhc2")

    # Update the immune scores
    exist_immune_scores_mhc1 = pd.read_pickle(immune_scores_mhc1_path)
    exist_immune_scores_mhc2 = pd.read_pickle(immune_scores_mhc2_path)
    new_immune_scores_mhc1 = pd.read_pickle(config.immune_scores_mhc1_path)
    new_immune_scores_mhc2 = pd.read_pickle(config.immune_scores_mhc2_path)
    immune_scores_mhc1 = pd.concat([exist_immune_scores_mhc1, new_immune_scores_mhc1])
    immune_scores_mhc2 = pd.concat([exist_immune_scores_mhc2, new_immune_scores_mhc2])

    # Save the updated immune scores
    immune_scores_mhc1.to_pickle(updated_immune_scores_mhc1_path)
    immune_scores_mhc2.to_pickle(updated_immune_scores_mhc2_path)

    # Use the updated immune scores
    params_to_change = {
        "immune_scores_mhc1_path": updated_immune_scores_mhc1_path,
        "immune_scores_mhc2_path": updated_immune_scores_mhc2_path,
    }
    params.update(params_to_change)
    config = EpitopeGraphConfig(**params)

    # Create empty lists to store the results
    path_cov_dfs = []
    pop_cov_dfs = []

    for seq_id, seq in seqs_dict.items():
        seq_name = seq_id
        # Convert seqs to k-mers
        kmers = kmerise_simple(seq, config.k)
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


##########################################################
# Predict the number of hits for sequences in a FASTA file
##########################################################


def pred_human_n_hits(config: EpitopeGraphConfig) -> pd.DataFrame:
    """
    Predict the number of peptide-HLA hits for the human population.
    """
    return pred_n_hits_parallel(
        workdir="data/human_mhcs",
        fasta_path="data/mice_mhcs/NP_designs_June2023.txt",
        mhc_alleles=get_hla_alleles(config),
        mhc1_epitopes_path="data/input/epitopes_human_mhc1.csv",
        mhc2_epitopes_path="data/input/epitopes_human_mhc2.csv",
    )


def pred_mouse_n_hits() -> pd.DataFrame:
    """
    Predict the number of peptide-H-2 hits for C57BL/6 mice.
    """
    return pred_n_hits_parallel(
        workdir="data/mice_mhcs",
        fasta_path="data/mice_mhcs/NP_designs_June2023.txt",
        mhc1_epitopes_path="data/input/epitopes_mice_c57bl6_mhc1.csv",
        mhc2_epitopes_path="data/input/epitopes_mice_c57bl6_mhc2.csv",
    )


def get_hla_alleles(config: EpitopeGraphConfig) -> dict:
    """
    Returns a dictionary of HLA alleles for MHC1 and MHC2.
    """
    mhc1_alleles = pd.read_csv(
        config.mhc1_alleles_path, sep="\t", header=None, names=["allele"]
    )
    mhc2_alleles = pd.read_csv(
        config.mhc2_alleles_path, sep="\t", header=None, names=["allele"]
    )
    return {
        "mhc1": mhc1_alleles["allele"].tolist(),
        "mhc2": mhc2_alleles["allele"].tolist(),
    }


def pred_n_hits_parallel(
    workdir: str = "data/mice_mhcs",
    fasta_path: str = "data/mice_mhcs/NP_designs_June2023.txt",
    mhc_alleles: dict = {"mhc1": ["H-2-Kb", "H-2-Db"], "mhc2": ["H-2-IAb"]},
    mhc1_epitopes_path: str = "data/input/mhc1_epitopes.csv",
    mhc2_epitopes_path: str = "data/input/mhc2_epitopes.csv",
    num_threads: int = 4,
    out_path: str = None,
):
    """
    Predict the number of hits for all sequences in a FASTA file.
    """
    dfs = []

    def process_allele(mhc_class, allele):
        netmhcpan_cmd = "netMHCpan" if mhc_class == "mhc1" else "netMHCIIpan"
        allele_clean = allele.replace("*", "_").replace(":", "").replace("-", "")
        outpath = f"{workdir}/{allele_clean}_{mhc_class}_predictions.txt"
        cmd = f"{netmhcpan_cmd} -f {fasta_path} -a {allele} -BA -xls -xlsfile {outpath}"
        subprocess.run(cmd, shell=True)

        # Read in and process the predictions into a dataframe
        df = pd.read_csv(outpath, delimiter="\t", skiprows=[0])
        df["mhc_class"] = mhc_class
        df["allele"] = allele
        df = df.rename(
            columns={
                "Peptide": "peptide",
                "Core": "core",
                "Score_BA": "BA-score",
                "Rank_BA": "BA_Rank",
                "Score": "EL-score",
                "Rank": "EL_Rank",
            }
        )
        if "nM" not in df.columns:
            df["nM"] = 50000 ** (1 - df["BA-score"])
        df["transformed_affinity"] = df["nM"].apply(transform_affinity)
        return df

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each combination of mhc_class and allele
        futures = [
            executor.submit(process_allele, mhc_class, allele)
            for mhc_class, alleles in mhc_alleles.items()
            for allele in alleles
        ]

        # Process the results as they become available
        for future in as_completed(futures):
            result = future.result()
            dfs.append(result)

    # Create a dataframe of the number of epitopes, <50nm, strong and weak binders for each vaccine design
    pmhc_aff = pd.concat(dfs)
    binders_df = pd.DataFrame(columns=["ID", "mhc_class", "<50nM", "strong", "weak"])

    if mhc1_epitopes_path is not None and mhc2_epitopes_path is not None:
        mhc1_epitopes = pd.read_csv(mhc1_epitopes_path, header=1)
        mhc2_epitopes = pd.read_csv(mhc2_epitopes_path, header=1)
        mhc1_epitopes = set(mhc1_epitopes["Name"].tolist())
        mhc2_epitopes = set(mhc2_epitopes["Name"].tolist())

    for id in pmhc_aff.ID.unique():
        for mhc in mhc_alleles.keys():
            df = pmhc_aff[(pmhc_aff["mhc_class"] == mhc) & (pmhc_aff["ID"] == id)]
            vstrong_affinity_cutoff = 0.638
            strong_affinity_cutoff = 0.5 if mhc == "mhc1" else 1.0
            weak_affinity_cutoff = 2.0 if mhc == "mhc1" else 5.0
            if mhc == "mhc1":
                n_epitopes = int(
                    len([m for m in df["peptide"].unique() if m in mhc1_epitopes])
                )
            else:
                n_epitopes = int(
                    len([m for m in df["peptide"].unique() if m in mhc2_epitopes])
                )
            n_vstrong_binders = len(
                df[df["transformed_affinity"] > vstrong_affinity_cutoff]
            )
            n_strong_binders = len(df[df["BA_Rank"] < strong_affinity_cutoff])
            n_weak_binders = len(df[df["BA_Rank"] < weak_affinity_cutoff])
            # TODO: Add a column for the number of epitopes found in IEDB
            binders_df = binders_df.append(
                {
                    "ID": id,
                    "mhc_class": mhc,
                    "weak": n_weak_binders,
                    "strong": n_strong_binders,
                    "<50nM": n_vstrong_binders,
                    "epitope": n_epitopes,
                },
                ignore_index=True,
            )

    # Melt and rename values in the dataframe
    binders_df = binders_df.melt(
        id_vars=["ID", "mhc_class"], var_name="binding_threshold", value_name="hits"
    )
    binders_df["mhc_class"] = binders_df["mhc_class"].replace(
        {"mhc1": "MHC Class I", "mhc2": "MHC Class II"}
    )
    binders_df["binding_threshold"] = binders_df["binding_threshold"].replace(
        {
            "epitope": "Experimentally Observed (IEDB)",
            "<50nM": "Very Strong (<50nM)",
            "strong": "Strong",
            "weak": "Weak",
        }
    )
    binders_df["ID"] = binders_df["ID"].replace({"CITVax_dStab": "CITVax-dStab"})

    if out_path is None:
        out_path = f"{workdir}/pred_hits.csv"
    binders_df.to_csv(out_path, index=False)

    return binders_df


############################################
# Annotation and coverage scores by position
############################################


def load_annotation(
    gff_file: str = "data/input/P0DTC9.gff",
    out_path: str = "data/figures/cov_by_pos.png",
    colors: dict = {
        "Domain": "blue",
        "Region": "green",
        "Motif": "orange",
        "Compositional bias": "purple",
        "Binding site": "red",
    },
    sequence_length: int = 419,
) -> GraphicRecord:
    """
    Load annotation from a GFF file.
    """
    features = []
    with open(gff_file, "r") as file:
        for line in file:
            if line.startswith("#") or line.strip() == "":
                continue
            columns = line.strip().split("\t")
            feature_type = columns[2]
            note_list = [
                x.split("=")[1] for x in columns[8].split(";") if x.startswith("Note=")
            ]
            if note_list:
                note = note_list[0]
            else:
                note = "Binding site"  # or any default value you want to assign
            if feature_type in colors:
                features.append(
                    GraphicFeature(
                        start=int(columns[3]),
                        end=int(columns[4]),
                        strand=+1,
                        color=colors[feature_type],
                        label=note,
                    )
                )

    # Sort features by start position
    features.sort(key=lambda f: f.start)
    record = GraphicRecord(sequence_length=sequence_length, features=features)
    return record


def compute_cov_by_pos(
    vaccine_designs: list,
    epitope_graph: nx.digraph,
    config: EpitopeGraphConfig = None,
    comp_df: pd.DataFrame = None,
    sarbeco_clades=[
        1,
        3,
        6,
    ],
    merbeco_clades=[2, 4, 5],
) -> pd.DataFrame:
    """
    Compute the coverage by position for a vaccine design.
    """
    pos = 0
    kmer_scores_dict = {
        "kmer": [],
        "position": [],
        "conservation_total": [],
        "population_coverage_mhc1": [],
        "population_coverage_mhc2": [],
    }
    if config is not None:
        kmer_scores_dict["conservation_sarbeco"] = []
        kmer_scores_dict["conservation_merbeco"] = []
        seqs_dict = load_fasta(config.fasta_path)
        N = len(seqs_dict)
        clades_dict = (
            comp_df[["Sequence_id", "cluster"]]
            .set_index("Sequence_id")
            .to_dict()["cluster"]
        )
        sar_seqs_dict = {
            seq_id: seq
            for seq_id, seq in seqs_dict.items()
            if clades_dict[seq_id] in sarbeco_clades
        }
        mer_seqs_dict = {
            seq_id: seq
            for seq_id, seq in seqs_dict.items()
            if clades_dict[seq_id] in merbeco_clades
        }
        N_sar = len(sar_seqs_dict)
        N_mer = len(mer_seqs_dict)

    # For each k-mer in the vaccine design
    for i, kmer in enumerate(vaccine_designs[0]):
        # Get the position of the k-mer in the vaccine design
        k = len(kmer)
        if i == 0:
            pos = k
        else:
            overlap = min(k, prev_k) - 1
            pos += k - overlap
        prev_k = k
        # Get the total conservation
        cons_tot = f(epitope_graph, kmer, f="frequency") * 100
        if config is not None:
            # Calculate the sarbeco conservation
            n_sar = 0
            for seq_id, seq in sar_seqs_dict.items():
                if kmer in seq:
                    n_sar += 1
            cons_sar = n_sar / N_sar * 100
            # Calculate the merbeco conservation
            n_mer = 0
            for seq_id, seq in mer_seqs_dict.items():
                if kmer in seq:
                    n_mer += 1
            cons_mer = n_mer / N_mer * 100
        # Get the population coverage
        pop_cov_mhc1 = f(epitope_graph, kmer, f="population_coverage_mhc1") * 100
        pop_cov_mhc2 = f(epitope_graph, kmer, f="population_coverage_mhc2") * 100
        # Add the k-mer and its scores to the dict
        kmer_scores_dict["kmer"].append(kmer)
        kmer_scores_dict["position"].append(pos)
        kmer_scores_dict["conservation_total"].append(cons_tot)
        kmer_scores_dict["population_coverage_mhc1"].append(pop_cov_mhc1)
        kmer_scores_dict["population_coverage_mhc2"].append(pop_cov_mhc2)
        if config is not None:
            kmer_scores_dict["conservation_sarbeco"].append(cons_sar)
            kmer_scores_dict["conservation_merbeco"].append(cons_mer)

    # Convert the dict to a dataframe
    kmer_scores_df = pd.DataFrame(kmer_scores_dict)
    return kmer_scores_df


if __name__ == "__main__":
    kmer_graph_params = {
        "human_proteome_path": "data/input/human_proteome_2023.03.14.fasta.gz",
        "mhc1_alleles_path": "../optivax/scoring/MHC1_allele_mary_cleaned.txt",
        "mhc2_alleles_path": "../optivax/scoring/MHC2_allele_mary_cleaned.txt",
        "hap_freq_mhc1_path": "../optivax/haplotype_frequency_marry.pkl",
        "hap_freq_mhc2_path": "../optivax/haplotype_frequency_marry2.pkl",
        "k": [9, 15],
        "m": 1,
        "n_target": 1,
        "robust": False,
        "aligned": False,
        "decycle": True,
        "equalise_clades": True,
        "n_clusters": None,
        "weights": {
            "frequency": 1.0,
            "population_coverage_mhc1": 2.0,
            "population_coverage_mhc2": 1.0,
            "clade": 1.0,
        },
        "affinity_predictors": ["netmhcpan"],
        "immune_scores_mhc1_path": "data/results/MHC_Binding/sar_mer_nuc_protein_immune_scores_netmhc.pkl",
        "immune_scores_mhc2_path": "data/results/MHC_Binding/sar_mer_nuc_protein_immune_scores_netmhcii.pkl",
    }
    analyses_params = {
        "results_dir": "data/outputs",
        "run_kmer_filtering": False,
        "run_scores_distribution": False,
        "run_binding_criteria": False,
        "run_netmhc_calibration": False,
        "run_kmer_graphs": False,
        "run_compare_antigens": False,
        "run_population_coverage": False,
    }
    config = AnalysesConfig(**analyses_params)
    run_analyses(config, kmer_graph_params)
