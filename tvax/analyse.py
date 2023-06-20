import concurrent.futures
import networkx as nx
import os
import pandas as pd
from pathlib import Path
import subprocess

from dna_features_viewer import GraphicFeature, GraphicRecord
from tvax.config import EpitopeGraphConfig, Weights
from tvax.design import design_vaccines
from tvax.eval import (
    compute_eigen_dist,
    compute_population_coverage,
    compute_pathogen_coverages,
    compute_av_pathogen_coverage,
)
from tvax.graph import build_epitope_graph, f
from tvax.plot import plot_population_coverage, plot_pop_cov_lineplot
from tvax.score import transform_affinity, add_population_coverage
from tvax.seq import load_fasta, path_to_seq, seq_to_kmers, kmerise, kmerise_simple
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each combination of w_mhc1 and w_mhc2
        futures = [
            executor.submit(process_weight, w_mhc1, w_mhc2)
            for w_mhc1 in w_mhc1_lst
            for w_mhc2 in w_mhc2_lst
        ]

        # Process the results as they become available
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            df = df.append(result, ignore_index=True)
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
    num_threads: int = 4,
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each combination of mhc_class and allele
        futures = [
            executor.submit(process_allele, mhc_class, allele)
            for mhc_class, alleles in mhc_alleles.items()
            for allele in alleles
        ]

        # Process the results as they become available
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            dfs.append(result)

    # Create a dataframe of the number of strong and weak binders for each vaccine design
    pmhc_aff = pd.concat(dfs)
    binders_df = pd.DataFrame(columns=["ID", "mhc_class", "<50nM", "strong", "weak"])

    for id in pmhc_aff.ID.unique():
        for mhc in mhc_alleles.keys():
            df = pmhc_aff[(pmhc_aff["mhc_class"] == mhc) & (pmhc_aff["ID"] == id)]
            vstrong_affinity_cutoff = 0.638
            strong_affinity_cutoff = 0.5 if mhc == "mhc1" else 1.0
            weak_affinity_cutoff = 2.0 if mhc == "mhc1" else 5.0
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
        {"<50nM": "Very Strong (<50nM)", "strong": "Strong", "weak": "Weak"}
    )
    binders_df["ID"] = binders_df["ID"].replace({"CITVax_dStab": "CITVax-dStab"})
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
