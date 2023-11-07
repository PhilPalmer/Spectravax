#!/usr/bin/env python

import os

from pathlib import Path
from pydantic import BaseModel, validator
from typing import List, Optional
from typing_extensions import Literal


"""
Config object for the epitope graph.
"""


# Top 27 HLAs
default_alleles = [
    "HLA-A*01:01",
    "HLA-A*02:01",
    "HLA-A*02:03",
    "HLA-A*02:06",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*23:01",
    "HLA-A*24:02",
    "HLA-A*26:01",
    "HLA-A*30:01",
    "HLA-A*30:02",
    "HLA-A*31:01",
    "HLA-A*32:01",
    "HLA-A*33:01",
    "HLA-A*68:01",
    "HLA-A*68:02",
    "HLA-B*07:02",
    "HLA-B*08:01",
    "HLA-B*15:01",
    "HLA-B*35:01",
    "HLA-B*40:01",
    "HLA-B*44:02",
    "HLA-B*44:03",
    "HLA-B*51:01",
    "HLA-B*53:01",
    "HLA-B*57:01",
    "HLA-B*58:01",
]


class Weights(BaseModel):
    """
    Config Object for Weights to Score Epitopes in the Graph.
    """

    frequency: float = 1.0
    population_coverage_mhc1: float = 1.0
    population_coverage_mhc2: float = 1.0
    clade: float = 1.0


class EpitopeGraphConfig(BaseModel):
    """
    Config Object for Epitope Graph Construction.
    """

    fasta_nt_path: Path = None
    fasta_path: Path = None
    prefix: str = None
    results_dir: Path = None
    k: list[int] = [9, 15]
    m: int = 1
    n_target: int = 1
    robust: bool = True
    seq_identity: float = 0.95
    scoring_method: Literal["weighted_average", "multiplicative"] = "multiplicative"
    binding_criteria: Literal["transformed_affinity", "EL-score"] = "EL-score"
    affinity_cutoff_mhc1: float = 0.638  # 50nM after logistic transform
    affinity_cutoff_mhc2: float = 0.638  # 0.426 = 500nM after logistic transform
    # TODO: Use elbow method to determine the optimal conservation threshold
    conservation_threshold: float = 0.01  # Exclude k-mers with < 1% conservation
    aligned: bool = False
    decycle: bool = True
    equalise_clades: bool = True
    n_clusters: Optional[int] = None
    peptide_chunk_size: int = 500
    n_threads: int = 4
    redun_db_path: Path = "redun.db"
    netmhcpan_tmpdir: Path = "/tmp/netMHCpanXXXXXX"
    edge_colour = "#BFBFBF"
    weights: Weights = Weights()
    affinity_predictors: List[Literal["mhcflurry", "netmhcpan"]] = [
        "netmhcpan",
    ]
    # TODO: Download the human proteome from UniProt and store it locally
    human_proteome_path: Path = None
    mhc1_alleles_path: Path = None
    mhc2_alleles_path: Path = None
    hap_freq_mhc1_path: Path = None
    hap_freq_mhc2_path: Path = None
    peptides_dir: Path = None
    # TODO: Rename these files and gzip the immune scores
    immune_scores_mhc1_path: Optional[Path] = None
    immune_scores_mhc2_path: Optional[Path] = None
    raw_affinity_mhcflurry_path: Optional[Path] = None
    raw_affinity_netmhc_path: Optional[Path] = None
    raw_affinity_netmhcii_path: Optional[Path] = None
    msa_path: Optional[Path] = None
    # TODO: Validate that the paths to netMHCpan and netMHCIIpan are valid if:
    # 1. The affinity_predictors contains "netmhcpan"
    # 2. The weights.population_coverage_mhc1 or weights.population_coverage_mhc2
    netmhcpan_cmd: str = "netMHCpan"
    netmhcpanii_cmd: str = "netMHCIIpan"
    # TODO: Add file path for list of alleles
    alleles: List[str] = default_alleles

    @validator("fasta_nt_path", pre=True, always=True)
    def validate_fasta_nt_path(cls, value):
        if value is not None and not Path(value).exists():
            raise ValueError(f"File {value} does not exist.")
        return value

    @validator("fasta_path", pre=True, always=True)
    def validate_fasta_path(cls, value, values):
        fasta_nt_path = values.get("fasta_nt_path")
        if value is None and fasta_nt_path is None:
            raise ValueError("Please provide a FASTA file path.")
        if value is not None and not Path(value).exists():
            raise ValueError(f"File {value} does not exist.")
        return value

    @validator("prefix", pre=True, always=True)
    def validate_prefix(cls, value, values):
        if value is None:
            if values.get("fasta_path") is not None:
                fasta_path = values.get("fasta_path")
            if values.get("fasta_nt_path") is not None:
                fasta_path = values.get("fasta_nt_path")
            fasta_base = os.path.splitext(os.path.basename(fasta_path))[0]
            return fasta_base
        else:
            return value

    @validator("results_dir", pre=True, always=True)
    def validate_results_dir(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            value = f"results_{prefix}"
        subdirs = ["MSA", "MHC_Binding", "Seq_Preprocessing"]
        for subdir in subdirs:
            if not Path(f"{value}/{subdir}").exists():
                Path(f"{value}/{subdir}").mkdir(parents=True, exist_ok=True)
        return value

    @validator("msa_path", pre=True, always=True)
    def validate_msa_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            msa_path = f"{results_dir}/MSA/{prefix}.msa"
            return Path(msa_path)

    @validator("peptides_dir", pre=True, always=True)
    def validate_peptides_dir(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            peptides_dir = Path(f"{results_dir}/MHC_Binding/peptides")
            os.makedirs(peptides_dir, exist_ok=True)
            return peptides_dir

    @validator("immune_scores_mhc1_path", pre=True, always=True)
    def validate_immune_scores_mhc1_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            affinity_predictors = values.get("affinity_predictors")
            if (
                "mhcflurry" in affinity_predictors
                and "netmhcpan" in affinity_predictors
            ):
                predictors = "ensemble"
            elif "mhcflurry" in affinity_predictors:
                predictors = "mhcflurry"
            elif "netmhcpan" in affinity_predictors:
                predictors = "netmhc"
            immune_scores_path = (
                f"{results_dir}/MHC_Binding/{prefix}_immune_scores_{predictors}.pkl"
            )
            return Path(immune_scores_path)
        return value

    @validator("immune_scores_mhc2_path", pre=True, always=True)
    def validate_immune_scores_mhc2_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            immune_scores_path = (
                f"{results_dir}/MHC_Binding/{prefix}_immune_scores_netmhcii.pkl"
            )
            return Path(immune_scores_path)
        return value

    @validator("raw_affinity_mhcflurry_path", pre=True, always=True)
    def validate_raw_affinity_mhcflurry_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            raw_affinity_mhcflurry_path = (
                f"{results_dir}/MHC_Binding/{prefix}_raw_affinity_mhcflurry.pkl"
            )
            return Path(raw_affinity_mhcflurry_path)
        return value

    @validator("raw_affinity_netmhc_path", pre=True, always=True)
    def validate_raw_affinity_netmhc_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            raw_affinity_netmhc_path = (
                f"{results_dir}/MHC_Binding/{prefix}_raw_affinity_netmhc.pkl.gz"
            )
            return Path(raw_affinity_netmhc_path)
        return value

    @validator("raw_affinity_netmhcii_path", pre=True, always=True)
    def validate_raw_affinity_netmhcii_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            raw_affinity_netmhcii_path = (
                f"{results_dir}/MHC_Binding/{prefix}_raw_affinity_netmhcii.pkl.gz"
            )
            return Path(raw_affinity_netmhcii_path)
        return value


class AnalysesConfig(BaseModel):
    """
    Config Object for performing the different analyses.
    """

    results_dir: Path = None
    # Antigen graphs
    antigen_graphs_pkl: Path = None
    # K-mer filtering
    run_kmer_filtering: bool = True
    kmer_filtering_fig: Path = None
    # Scores distribution
    run_scores_distribution: bool = True
    scores_distribution_json: Path = None
    scores_distribution_fig: Path = None
    # Binding criteria
    run_binding_criteria: bool = True
    binding_criteria_csv: Path = None
    binding_crtieria_dir: Path = None
    binding_criteria_fig: Path = None
    # NetMHCpan Calibration
    run_netmhc_calibration: bool = True
    netmhc_calibration_raw_mhc1_csv: Path = None
    netmhc_calibration_raw_mhc2_csv: Path = None
    netmhc_calibration_csv: Path = None
    netmhc_calibration_fig: Path = None
    # K-mer graphs
    run_kmer_graphs: bool = True
    kmer_graph_antigen: str = "Sarbecovirus S RBD"
    kmer_graphs_fig: Path = None
    # Compare antigens
    run_compare_antigens: bool = True
    compare_antigens_csv: Path = None
    compare_antigens_fig: Path = None

    antigen: str = "Sarbeco-Merbeco N"
    # Population coverage
    run_population_coverage: bool = True
    population_coverage_csv: Path = None
    population_coverage_fig: Path = None
    # Pathogen coverage
    run_pathogen_coverage: bool = True
    pathogen_coverage_csv: Path = None
    pathogen_coverage_fig: Path = None
    # Prediction of experimental results
    run_experimental_prediction: bool = True
    exp_fasta_path: Path = None
    exp_pred_dir: Path = None
    mhc1_epitopes_path: Path = None
    mhc2_epitopes_path: Path = None
    experimental_prediction_csv: Path = None
    experimental_prediction_fig: Path = None
    # Coverage by position
    run_coverage_by_position: bool = True
    gff_path: Path = None
    coverage_by_position_fig: Path = None
    # Peptide-MHC heatmap
    run_peptide_mhc_heatmap: bool = True
    peptide_mhc_heatmap_fig: Path = None
    # Run expected number of displayed peptides E(#DPs) by country analysis
    run_exp_dps_by_country: bool = True
    exp_dps_by_country_data_csv: Path = None
    exp_dps_by_country_csv: Path = None
    exp_dps_by_country_fig: Path = None

    @validator("results_dir", pre=True, always=True)
    def validate_results_dir(cls, value, values):
        if value is None:
            value = "results"
        subdirs = ["data", "figures"]
        for subdir in subdirs:
            if not Path(f"{value}/{subdir}").exists():
                Path(f"{value}/{subdir}").mkdir(parents=True, exist_ok=True)
        return value

    @validator("antigen_graphs_pkl", pre=True, always=True)
    def validate_antigen_graphs_pkl(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            antigen_graphs_pkl = f"{results_dir}/data/antigen_graphs.pkl"
            return Path(antigen_graphs_pkl)
        return Path(value)

    @validator("kmer_filtering_fig", pre=True, always=True)
    def validate_kmer_filtering_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            kmer_filtering_fig = f"{results_dir}/figures/kmer_filtering.svg"
            return Path(kmer_filtering_fig)
        return Path(value)

    @validator("scores_distribution_json", pre=True, always=True)
    def validate_scores_distribution_json(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            scores_distribution_json = f"{results_dir}/data/scores_distribution.json"
            return Path(scores_distribution_json)
        return Path(value)

    @validator("scores_distribution_fig", pre=True, always=True)
    def validate_scores_distribution_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            scores_distribution_fig = f"{results_dir}/figures/scores_distribution.svg"
            return Path(scores_distribution_fig)
        return Path(value)

    @validator("binding_criteria_csv", pre=True, always=True)
    def validate_binding_criteria_csv(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            binding_criteria_csv = f"{results_dir}/data/binding_criteria.csv"
            return Path(binding_criteria_csv)
        return Path(value)

    @validator("binding_crtieria_dir", pre=True, always=True)
    def validate_binding_crtieria_dir(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            value = f"{results_dir}/data/binding_criteria"
        if not Path(value).exists():
            Path(value).mkdir(parents=True, exist_ok=True)
        return value

    @validator("binding_criteria_fig", pre=True, always=True)
    def validate_binding_criteria_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            binding_criteria_fig = f"{results_dir}/figures/binding_criteria.svg"
            return Path(binding_criteria_fig)
        return Path(value)

    @validator("netmhc_calibration_raw_mhc1_csv", pre=True, always=True)
    def validate_netmhc_calibration_raw_mhc1_csv(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            netmhc_calibration_raw_mhc1_csv = (
                f"{results_dir}/data/netmhc_calibration_raw_mhc1.csv"
            )
            return Path(netmhc_calibration_raw_mhc1_csv)
        return Path(value)

    @validator("netmhc_calibration_raw_mhc2_csv", pre=True, always=True)
    def validate_netmhc_calibration_raw_mhc2_csv(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            netmhc_calibration_raw_mhc2_csv = (
                f"{results_dir}/data/netmhc_calibration_raw_mhc2.csv"
            )
            return Path(netmhc_calibration_raw_mhc2_csv)
        return Path(value)

    @validator("netmhc_calibration_csv", pre=True, always=True)
    def validate_netmhc_calibration_csv(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            netmhc_calibration_csv = f"{results_dir}/data/netmhc_calibration.csv"
            return Path(netmhc_calibration_csv)
        return Path(value)

    @validator("netmhc_calibration_fig", pre=True, always=True)
    def validate_netmhc_calibration_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            netmhc_calibration_fig = f"{results_dir}/figures/netmhc_calibration.svg"
            return Path(netmhc_calibration_fig)
        return Path(value)

    @validator("kmer_graphs_fig", pre=True, always=True)
    def validate_kmer_graphs_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            kmer_graphs_fig = f"{results_dir}/figures/kmer_graphs.svg"
            return Path(kmer_graphs_fig)
        return Path(value)

    @validator("compare_antigens_csv", pre=True, always=True)
    def validate_compare_antigens_csv(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            compare_antigens_csv = f"{results_dir}/data/compare_antigens.csv"
            return Path(compare_antigens_csv)
        return Path(value)

    @validator("compare_antigens_fig", pre=True, always=True)
    def validate_compare_antigens_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            compare_antigens_fig = f"{results_dir}/figures/compare_antigens.svg"
            return Path(compare_antigens_fig)
        return Path(value)

    @validator("population_coverage_csv", pre=True, always=True)
    def validate_population_coverage_csv(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            antigen = values.get("antigen").replace(" ", "_")
            population_coverage_csv = (
                f"{results_dir}/data/{antigen}_population_coverage.csv"
            )
            return Path(population_coverage_csv)
        return Path(value)

    @validator("population_coverage_fig", pre=True, always=True)
    def validate_population_coverage_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            antigen = values.get("antigen").replace(" ", "_")
            population_coverage_fig = (
                f"{results_dir}/figures/{antigen}_population_coverage.svg"
            )
            return Path(population_coverage_fig)
        return Path(value)

    @validator("pathogen_coverage_csv", pre=True, always=True)
    def validate_pathogen_coverage_csv(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            antigen = values.get("antigen").replace(" ", "_")
            pathogen_coverage_csv = (
                f"{results_dir}/data/{antigen}_pathogen_coverage.csv"
            )
            return Path(pathogen_coverage_csv)
        return Path(value)

    @validator("pathogen_coverage_fig", pre=True, always=True)
    def validate_pathogen_coverage_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            antigen = values.get("antigen").replace(" ", "_")
            pathogen_coverage_fig = (
                f"{results_dir}/figures/{antigen}_pathogen_coverage.svg"
            )
            return Path(pathogen_coverage_fig)
        return Path(value)

    @validator("exp_pred_dir", pre=True, always=True)
    def validate_exp_pred_dir(cls, value, values):
        return Path(value)

    @validator("exp_fasta_path", pre=True, always=True)
    def validate_exp_fasta_path(cls, value, values):
        return Path(value)

    @validator("mhc1_epitopes_path", pre=True, always=True)
    def validate_mhc1_epitopes_path(cls, value, values):
        return Path(value)

    @validator("mhc2_epitopes_path", pre=True, always=True)
    def validate_mhc2_epitopes_path(cls, value, values):
        return Path(value)

    @validator("experimental_prediction_csv", pre=True, always=True)
    def validate_experimental_prediction_csv(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            antigen = values.get("antigen").replace(" ", "_")
            experimental_prediction_csv = (
                f"{results_dir}/data/{antigen}_experimental_prediction.csv"
            )
            return Path(experimental_prediction_csv)
        return Path(value)

    @validator("experimental_prediction_fig", pre=True, always=True)
    def validate_experimental_prediction_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            antigen = values.get("antigen").replace(" ", "_")
            experimental_prediction_fig = (
                f"{results_dir}/figures/{antigen}_experimental_prediction.svg"
            )
            return Path(experimental_prediction_fig)
        return Path(value)

    @validator("coverage_by_position_fig", pre=True, always=True)
    def validate_coverage_by_position_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            antigen = values.get("antigen").replace(" ", "_")
            coverage_by_position_fig = (
                f"{results_dir}/figures/{antigen}_coverage_by_position.svg"
            )
            return Path(coverage_by_position_fig)
        return Path(value)

    @validator("gff_path", pre=True, always=True)
    def validate_gff_path(cls, value, values):
        return Path(value)

    @validator("peptide_mhc_heatmap_fig", pre=True, always=True)
    def validate_peptide_mhc_heatmap_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            antigen = values.get("antigen").replace(" ", "_")
            peptide_mhc_heatmap_fig = (
                f"{results_dir}/figures/{antigen}_peptide_mhc_heatmap.svg"
            )
            return Path(peptide_mhc_heatmap_fig)
        return Path(value)

    @validator("exp_dps_by_country_data_csv", pre=True, always=True)
    def validate_exp_dps_by_country_data_csv(cls, value, values):
        return Path(value)

    @validator("exp_dps_by_country_csv", pre=True, always=True)
    def validate_exp_dps_by_country_csv(cls, value, values):
        return Path(value)

    @validator("exp_dps_by_country_fig", pre=True, always=True)
    def validate_exp_dps_by_country_fig(cls, value, values):
        if value is None:
            results_dir = values.get("results_dir")
            antigen = values.get("antigen").replace(" ", "_")
            exp_dps_by_country_fig = (
                f"{results_dir}/figures/{antigen}_exp_dps_by_country.svg"
            )
            return Path(exp_dps_by_country_fig)
        return Path(value)
