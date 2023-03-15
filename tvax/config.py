#!/usr/bin/env python

import mhcflurry
import os

from pathlib import Path
from pydantic import BaseModel, validator
from typing import List, Optional
from typing_extensions import Literal


"""
Config object for the epitope graph.
"""


def supported_alleles() -> List[str]:
    """
    Returns a list of all supported alleles by MHCflurry.
    """
    predictor = mhcflurry.Class1AffinityPredictor.load()
    return predictor.supported_alleles


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

    frequency: float = 1
    population_coverage_mhc1: float = 10
    population_coverage_mhc2: float = 10


class EpitopeGraphConfig(BaseModel):
    """
    Config Object for Epitope Graph Construction.
    """

    fasta_path: Path = None
    prefix: str = None
    results_dir: Path = "results"
    k: list[int] = [9, 12]
    m: int = 1
    n_target: int = 1
    affinity_cutoff_mhc1: float = 0.638  # 50nM after logistic transform
    affinity_cutoff_mhc2: float = 0.426  # 500nM after logistic transform
    aligned: bool = False
    decycle: bool = True
    equalise_clades: bool = True
    n_clusters: Optional[int] = None
    n_threads: int = 1
    edge_colour = "#BFBFBF"
    weights: Weights = Weights()
    # TODO: Use the specified affinity predictors
    mhc_affinity_predictors: Literal["mhcflurry", "netmhcpan"] = [
        "mhcflurry",
        "netmhcpan",
    ]
    # TODO: Download the human proteome from UniProt and store it locally
    human_proteome_path: Path = None
    mhc1_alleles_path: Path = None
    mhc2_alleles_path: Path = None
    hap_freq_mhc1_path: Path = None
    hap_freq_mhc2_path: Path = None
    peptides_path: Path = None
    immune_scores_mhc1_path: Optional[Path] = None
    immune_scores_mhc2_path: Optional[Path] = None
    raw_affinity_mhc1_path: Optional[Path] = None
    raw_affinity_mhc2_path: Optional[Path] = None
    msa_path: Optional[Path] = None
    netmhcpanii_cmd: str = "netMHCIIpan"
    # TODO: Add file path for list of alleles
    alleles: List[Literal[tuple(supported_alleles())]] = default_alleles

    @validator("fasta_path", pre=True, always=True)
    def validate_fasta_path(cls, value):
        if value is None:
            raise ValueError("Please provide a FASTA file path.")
        if not Path(value).exists():
            raise ValueError(f"File {value} does not exist.")
        return value

    @validator("prefix", pre=True, always=True)
    def validate_prefix(cls, value, values):
        if value is None:
            fasta_path = values.get("fasta_path")
            fasta_base = os.path.splitext(os.path.basename(fasta_path))[0]
            return fasta_base
        else:
            return value

    @validator("results_dir", pre=True, always=True)
    def validate_results_dir(cls, value):
        subdirs = ["MSA", "MHC_Binding"]
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

    @validator("peptides_path", pre=True, always=True)
    def validate_peptides_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            peptides_path = f"{results_dir}/MHC_Binding/{prefix}_peptides.txt"
            return Path(peptides_path)

    @validator("immune_scores_mhc1_path", pre=True, always=True)
    def validate_immune_scores_mhc1_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            immune_scores_path = (
                f"{results_dir}/MHC_Binding/{prefix}_immune_scores_mhc1.pkl"
            )
            return Path(immune_scores_path)

    @validator("immune_scores_mhc2_path", pre=True, always=True)
    def validate_immune_scores_mhc2_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            immune_scores_path = (
                f"{results_dir}/MHC_Binding/{prefix}_immune_scores_mhc2.pkl"
            )
            return Path(immune_scores_path)

    @validator("raw_affinity_mhc1_path", pre=True, always=True)
    def validate_raw_affinity_mhc1_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            raw_affinity_mhc1_path = (
                f"{results_dir}/MHC_Binding/{prefix}_raw_affinity_mhc1.pkl"
            )
            return Path(raw_affinity_mhc1_path)

    @validator("raw_affinity_mhc2_path", pre=True, always=True)
    def validate_raw_affinity_mhc2_path(cls, value, values):
        if value is None:
            prefix = values.get("prefix")
            results_dir = values.get("results_dir")
            raw_affinity_mhc2_path = (
                f"{results_dir}/MHC_Binding/{prefix}_raw_affinity_mhc2.pkl.gz"
            )
            return Path(raw_affinity_mhc2_path)

    @validator("alleles", pre=True, always=True)
    def validate_alleles(cls, value):
        if value == default_alleles:
            return value
        else:
            valid_alleles = supported_alleles()
            unsupported_alleles = [
                allele for allele in value if allele not in valid_alleles
            ]
            if unsupported_alleles:
                raise ValueError(
                    f"The following alleles are not supported by MHCflurry: {unsupported_alleles}"
                )
            return value
