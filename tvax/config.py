#!/usr/bin/env python

import mhcflurry

from pathlib import Path
from pydantic import BaseModel, validator
from typing import List, Optional
from typing_extensions import Literal


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
    processing: float = 1
    weak_mhc_binding: float = 1
    strong_mhc_binding: float = 1


class EpitopeGraphConfig(BaseModel):
    """
    Config Object for Epitope Graph Construction.
    """

    fasta_path: Path = None
    k: int = 9
    m: int = 1
    weak_percentile_threshold: float = 2
    strong_percentile_threshold: float = 0.5
    aligned: bool = False
    decycle: bool = True
    equalise_clades: bool = True
    n_clusters: Optional[int] = None
    edge_colour = "#BFBFBF"
    weights: Weights = Weights()
    hla_path: Optional[Path] = None
    iedb_path: Optional[Path] = None
    immune_scores_path: Optional[Path] = None
    msa_path: Optional[Path] = None
    # TODO: Add file path for list of alleles
    alleles: List[Literal[tuple(supported_alleles())]] = default_alleles

    @validator("fasta_path", pre=True, always=True)
    def validate_fasta_path(cls, value):
        if value is None:
            raise ValueError("Please provide a FASTA file path.")
        if not Path(value).exists():
            raise ValueError(f"File {value} does not exist.")
        return value

    @validator("msa_path", pre=True, always=True)
    def validate_msa_path(cls, value, values):
        if value is None:
            fasta_path = values.get("fasta_path")
            msa_path = ".".join(str(fasta_path).split(".")[:-1]) + ".msa"
            return Path(msa_path)

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
