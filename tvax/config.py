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


class EpitopeGraphConfig(BaseModel):
    """
    Config Object for Epitope Graph Construction.
    """

    fasta_path: Path = None
    k: int = 9
    m: int = 1
    aligned: bool = False
    decycle: bool = True
    equalise_clades: bool = True
    n_clusters: Optional[int] = None
    weights: dict = {
        "freq": 1,
        "immunogenicity": 1,
        "weak_mhc_binding": 1,
        "strong_mhc_binding": 1,
        "known_epitope": 0,
    }
    hla_path: Optional[Path] = None
    iedb_path: Optional[Path] = None
    immune_scores_path: Optional[Path] = None
    # TODO: Add file path for list of alleles
    alleles: List[Literal[tuple(supported_alleles())]] = default_alleles

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
