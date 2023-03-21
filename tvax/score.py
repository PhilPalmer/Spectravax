import mhcflurry
import numpy as np
import os
import pandas as pd
import subprocess

from sklearn.preprocessing import MinMaxScaler
from tvax.config import EpitopeGraphConfig
from tvax.seq import load_fasta, kmerise_simple

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
    if config.weights.population_coverage_mhc1:
        kmers_dict = add_population_coverage(kmers_dict, config, "mhc1")
    if config.weights.population_coverage_mhc2:
        kmers_dict = add_population_coverage(kmers_dict, config, "mhc2")
    kmers_dict = add_clade_weights(kmers_dict, clades_dict, N)
    kmers_dict = add_score(kmers_dict, config)
    if config.human_proteome_path:
        kmers_dict = zero_undesired_peptides(
            kmers_dict, config.human_proteome_path, config.k
        )
    return kmers_dict


def add_score(kmers_dict: dict, config: EpitopeGraphConfig) -> dict:
    """
    Add the total weighted score to each k-mer.
    """
    # Calculate the total score for each k-mer by summing the weighted scores
    # TODO: Check if dividing by the sum of the weights is necessary
    for kmer in kmers_dict:
        kmers_dict[kmer]["score"] = sum(
            [kmers_dict[kmer][name] * val for name, val in config.weights if val > 0]
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


def add_population_coverage(
    kmers_dict: dict,
    config: EpitopeGraphConfig,
    mhc_type: str = "mhc1",
) -> dict:
    """
    Add the population coverage of each peptide
    """
    # Load the data
    peptides = list(kmers_dict.keys())
    hap_freq_path = (
        config.hap_freq_mhc1_path if mhc_type == "mhc1" else config.hap_freq_mhc2_path
    )
    hap_freq, average_frequency = load_haplotypes(hap_freq_path)
    overlap_haplotypes = load_overlap(peptides, hap_freq, config, mhc_type)

    # Calculate and save the population coverage for each peptide
    for e in kmers_dict:
        kmers_dict[e][f"population_coverage_{mhc_type}"] = optivax_robust(
            overlap_haplotypes, average_frequency, config.n_target, [e]
        )

    return kmers_dict


def load_haplotypes(hap_freq_path: str) -> tuple:
    """
    Load the haplotype frequency data
    """
    hap_freq = pd.read_pickle(hap_freq_path)
    # The haplotype frequency file considers White, Asians, and Black:  We just average them all out here.
    # TODO: Add support for weighted averaging based on the target population
    average_frequency = pd.DataFrame(index=["Average"], columns=hap_freq.columns)
    average_frequency.loc["Average", :] = hap_freq.sum(axis=0) / len(hap_freq)
    return hap_freq, average_frequency


def load_overlap(peptides, hap_freq, config: EpitopeGraphConfig, mhc_type: str):
    """
    Load the overlap between peptides and haplotypes
    """
    if mhc_type == "mhc1":
        alleles_path = config.mhc1_alleles_path
        immune_scores_path = config.immune_scores_mhc1_path
        affinity_cutoff = config.affinity_cutoff_mhc1
    else:
        alleles_path = config.mhc2_alleles_path
        immune_scores_path = config.immune_scores_mhc2_path
        affinity_cutoff = config.affinity_cutoff_mhc2

    if immune_scores_path and os.path.exists(immune_scores_path):

        overlap_haplotypes = pd.read_pickle(immune_scores_path)

    else:

        # Load the data
        peptides = pd.DataFrame({"peptide": peptides})
        hla_alleles = pd.read_csv(alleles_path, names=["allele"])

        # Predict the binding affinity
        if mhc_type == "mhc1":
            if "mhcflurry" in config.affinity_predictors:
                mhcflurry_pivot = predict_affinity_mhcflurry(
                    peptides, hla_alleles, config.raw_affinity_mhcflurry_path
                )
            if "netmhcpan" in config.affinity_predictors:
                netmhc_pivot = predict_affinity_netmhcpan(
                    peptides,
                    hla_alleles,
                    config.raw_affinity_netmhc_path,
                    config,
                    "mhc1",
                )
            if (
                "mhcflurry" in config.affinity_predictors
                and "netmhcpan" in config.affinity_predictors
            ):
                pmhc_aff_pivot = ensemble_predictions(netmhc_pivot, mhcflurry_pivot)
            elif "mhcflurry" in config.affinity_predictors:
                pmhc_aff_pivot = mhcflurry_pivot
            elif "netmhcpan" in config.affinity_predictors:
                pmhc_aff_pivot = netmhc_pivot
        else:
            pmhc_aff_pivot = predict_affinity_netmhcpan(
                peptides, hla_alleles, config.raw_affinity_netmhcii_path, config, "mhc2"
            )

        # Binarize the data to create hits
        pmhc_aff_pivot = pmhc_aff_pivot.applymap(
            lambda x: 1 if x > affinity_cutoff else 0
        )
        # Drop the loci from the columns and get the data down to a Single Index of alleles for the columns
        pmhc_aff_pivot = pmhc_aff_pivot.droplevel("loci", axis=1)
        # Run the add_hits_across_haplotypes function on the desired data.
        overlap_haplotypes = add_hits_across_haplotypes(pmhc_aff_pivot, hap_freq)

        # Save to file
        overlap_haplotypes.to_pickle(immune_scores_path)

    return overlap_haplotypes


def predict_affinity_mhcflurry(
    peptides: pd.DataFrame,
    hla_alleles: pd.DataFrame,
    raw_affinity_path: str,
):
    """
    Predict the binding affinity of each peptide-HLA pair using MHCflurry
    """
    if os.path.exists(raw_affinity_path):

        # TODO: Check if the peptides and alleles are the same as the ones in the file
        pmhc_aff_pivot = pd.read_pickle(raw_affinity_path)

    else:
        mhcflurry_presentation_cmd = (
            "mhcflurry-downloads fetch models_class1_presentation"
        )
        mhcflurry_pan_cmd = "mhcflurry-downloads fetch models_class1_pan"
        subprocess.call(mhcflurry_presentation_cmd, shell=True)
        subprocess.call(mhcflurry_pan_cmd, shell=True)
        predictor = mhcflurry.Class1AffinityPredictor().load()
        peptides["key"] = 0
        hla_alleles["key"] = 0
        pmhc_aff = peptides.merge(hla_alleles, how="outer")
        pmhc_aff = pmhc_aff.drop(columns=["key"])
        predictions = predictor.predict(
            peptides=pmhc_aff["peptide"].to_list(), alleles=pmhc_aff["allele"].to_list()
        )
        pmhc_aff["affinity"] = predictions
        pmhc_aff["transformed_affinity"] = pmhc_aff["affinity"].apply(
            transform_affinity
        )
        pmhc_aff = aggregate_binding_mhc1(pmhc_aff)

        # Pivot the data
        pmhc_aff_pivot = pmhc_aff.pivot_table(
            index="peptide",
            columns=["loci", "allele"],
            values="transformed_affinity",
        )
        pmhc_aff_pivot.to_pickle(raw_affinity_path)
    return pmhc_aff_pivot


def predict_affinity_netmhcpan(
    peptides: pd.DataFrame,
    hla_alleles: pd.DataFrame,
    raw_affinity_path: str,
    config: EpitopeGraphConfig,
    mhc_type: str = "mhc1",
) -> pd.DataFrame:
    """
    Predict the binding affinity of each peptide-HLA pair using NetMHCpan4.1
    """
    if os.path.exists(raw_affinity_path):

        # TODO: Check if the peptides and alleles are the same as the ones in the file
        pmhc_aff_pivot = pd.read_pickle(raw_affinity_path)

    else:
        # Define vars
        args_path = f"{config.results_dir}/MHC_Binding/net{mhc_type}_args.txt"
        preds_dir = f"{config.results_dir}/MHC_Binding/net{mhc_type}_preds"
        if mhc_type == "mhc1":
            cmd_template = (
                "-p {peptides_path} -a {allele} -BA -xls -xlsfile {allele_path}"
            )
            netmhcpan_cmd = config.netmhcpan_cmd
        else:
            cmd_template = "-inptype 1 -f {peptides_path} -a {allele}  -BA -xls -xlsfile {allele_path}"
            netmhcpan_cmd = config.netmhcpanii_cmd

        # Create a file with the peptides.
        peptides.to_csv(config.peptides_path, index=False, header=False)

        # Create commands for running NetMHCpan4.1.
        cmds = []
        for allele in hla_alleles["allele"].values:
            cmd = cmd_template.format(
                peptides_path=config.peptides_path,
                allele=allele.replace("*", "_").replace(":", ""),
                allele_path=f"{preds_dir}/{allele.replace('*', '_').replace(':', '')}_preds.xls",
            )
            cmds.append(cmd)

        with open(args_path, "w") as f:
            for cmd in cmds:
                f.write(cmd + "\n")

        # Create directory for results if it doesn't exist
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)

        # Run NetMHCpan4.1
        netmhcpan_cmd = f"cat {args_path} | xargs -P {config.netmhc_max_procs} -d '\n' -n 1 {netmhcpan_cmd}"
        subprocess.call(netmhcpan_cmd, shell=True)

        # Load binding predictions
        dfs = []
        for allele in hla_alleles["allele"].values:
            try:
                df = pd.read_csv(
                    f"{preds_dir}/{allele.replace('*', '_').replace(':', '')}_preds.xls",
                    delimiter="\t",
                    skiprows=[0],
                )
            except:
                continue
            df["allele"] = allele
            df = df.drop(columns=["Pos", "ID", "Ave", "NB"])
            dfs.append(df)

        pmhc_aff = pd.concat(dfs)
        pmhc_aff = pmhc_aff.rename(columns={"Peptide": "peptide"})
        pmhc_aff["transformed_affinity"] = pmhc_aff["nM"].apply(transform_affinity)

        if mhc_type == "mhc1":
            pmhc_aff = aggregate_binding_mhc1(pmhc_aff)
        else:
            pmhc_aff = aggregate_binding_mhc2(pmhc_aff)
        pmhc_aff_pivot = pmhc_aff.pivot_table(
            index="peptide",
            columns=["loci", "allele"],
            values="transformed_affinity",
        )
        pmhc_aff_pivot.to_pickle(raw_affinity_path, protocol=2)

    return pmhc_aff_pivot


def aggregate_binding_mhc1(pmhc_aff):
    """
    Aggregate the binding predictions for MHC Class I alleles: HLA-A74, HLA-C17, and HLA-C18.
    """
    pmhc_aff["loci"] = [x[:5] for x in pmhc_aff["allele"].values]
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
    pmhc_aff = pd.concat([pmhc_aff, a74, c17, c18], sort=False)
    return pmhc_aff


def aggregate_binding_mhc2(pmhc_aff):
    """
    Aggregate the binding predictions MHC class II alleles.
    """
    pmhc_aff["loci"] = [
        x[:4] if x[:3] == "DRB" else x[:6] for x in pmhc_aff["allele"].values
    ]
    pmhc_aff2 = (
        pmhc_aff.groupby(["peptide", "loci"]).count().reset_index()[["peptide", "loci"]]
    )
    pmhc_aff2["allele"] = "unknown"
    pmhc_aff2["Score"] = 0.0
    pmhc_aff2["Rank"] = 0.0
    pmhc_aff2["Score_BA"] = 0.0
    pmhc_aff2["Rank_BA"] = 0.0
    pmhc_aff2["nM"] = 0.0
    pmhc_aff2["transformed_affinity"] = 0.0
    pmhc_aff_pivot = pd.concat([pmhc_aff, pmhc_aff2], sort=False)
    return pmhc_aff_pivot


def transform_affinity(x: float, a_min: float = None, a_max: float = 50000) -> float:
    """
    Compute logistic-transformed binding affinity
    """
    x = np.clip(x, a_min=a_min, a_max=a_max)
    return 1 - np.log(x) / np.log(a_max)


def reverse_transform_affinity(
    x: float, a_min: float = None, a_max: float = 50000
) -> float:
    """
    Compute the reverse transformation of the logistic-transformed binding affinity
    """
    return a_max ** (1 - x)


def ensemble_predictions(netmhc_pivot: pd.DataFrame, mhcflurry_pivot: pd.DataFrame):
    """
    Ensemble the predictions from NetMHCpan4.1 and MHCflurry2.
    """
    assert netmhc_pivot.shape == mhcflurry_pivot.shape
    assert set(netmhc_pivot.T.index.values.tolist()) == set(
        mhcflurry_pivot.T.index.values.tolist()
    )
    assert set(netmhc_pivot.index.values.tolist()) == set(
        mhcflurry_pivot.index.values.tolist()
    )
    netmhc_pivot_nm = reverse_transform_affinity(netmhc_pivot)
    mhcflurry_pivot_nm = reverse_transform_affinity(mhcflurry_pivot)
    pmhc_aff_pivot = (netmhc_pivot_nm + mhcflurry_pivot_nm) / 2
    pmhc_aff_pivot = transform_affinity(pmhc_aff_pivot)
    return pmhc_aff_pivot


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


def zero_undesired_peptides(
    kmers_dict: dict, human_proteome_path: str, k: list
) -> dict:
    """
    Set the score to zero for k-mers that are in the human proteome
    """
    human_proteome = load_fasta(human_proteome_path)
    # kmerise the human proteome
    human_proteome_kmers = set(
        [kmer for seq in human_proteome.values() for kmer in kmerise_simple(seq, k)]
    )
    # zero the k-mers that are in the human proteome
    for kmer in kmers_dict:
        if kmer in human_proteome_kmers:
            kmers_dict[kmer]["score"] = 0
    return kmers_dict
