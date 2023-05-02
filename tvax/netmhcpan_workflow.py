import os
import pandas as pd

from collections import defaultdict
from math import ceil
from pathlib import Path
from redun import task, Scheduler, script, File
from redun.config import Config
from tvax.config import EpitopeGraphConfig
from typing import List, Tuple

"""
Redun workflow to run NetMHCpan and NetMHCIIpan to predict peptide-HLA binding affinities.
"""

redun_namespace = "netmhcpan"


def chunk_peptides(
    peptides: pd.DataFrame,
    peptides_dir: Path,
    peptide_chunk_size: int = 500,
) -> List[File]:
    """
    Split the input peptides into chunks and save each chunk to a file.
    """
    peptide_chunks = [
        peptides[i : i + peptide_chunk_size]
        for i in range(0, len(peptides), peptide_chunk_size)
    ]
    peptide_paths = []
    for chunk_idx, peptide_chunk in enumerate(peptide_chunks):
        peptide_path_chunk = f"{peptides_dir}/peptides_{chunk_idx}.txt"
        peptide_chunk.to_csv(peptide_path_chunk, index=False, header=False)
        peptide_paths.append(File(str(peptide_path_chunk)))
    return peptide_paths


@task
def predict_affinity_netmhcpan(
    peptides_path: File,
    allele: str,
    mhc_type: str = "mhc1",
    results_dir: Path = None,
    netmhcpan_cmd: str = None,
):
    """
    Run NetMHCpan or NetMHCIIpan to predict peptide-HLA binding affinities for a single allele.
    """
    peptide_chunk_idx = peptides_path.path.split("_")[-1].split(".")[0]
    preds_dir = f"{results_dir}/net{mhc_type}_peptide_preds"
    os.makedirs(preds_dir, exist_ok=True)
    allele_path = f"{preds_dir}/{allele.replace('*', '_').replace(':', '')}_preds_p{peptide_chunk_idx}.xls"
    peptides_path = File(os.path.abspath(peptides_path.path))
    peptides_flag = f"-p" if mhc_type == "mhc1" else f"-inptype 1 -f"
    return script(
        f"""
        {netmhcpan_cmd} {peptides_flag} {peptides_path.path} -a {allele} -BA -xls -xlsfile {allele_path}
        """,
        inputs=[peptides_path.stage(peptides_path.path)],
        outputs=[allele, File(allele_path).stage(allele_path)],
    )


@task
def group_by_allele(
    peptide_allele_paths: List[Tuple[str, File]]
) -> List[Tuple[str, List[File]]]:
    """
    Group the peptide paths by allele.
    """
    allele2files = defaultdict(list)
    for allele, file in peptide_allele_paths:
        allele2files[allele].append(file)
    return [[allele, files] for allele, files in allele2files.items()]


@task
def concat_allele_files(
    allele: str,
    allele_paths: List[File],
    mhc_type: str,
    results_dir: Path,
) -> List[File]:
    """
    Concatenate the per allele files into one file per allele.
    """
    allele_fnames = [f.path for f in allele_paths]
    allele_fname = allele_fnames[0]
    allele_fnames = " ".join(allele_fnames)
    preds_dir = f"{results_dir}/net{mhc_type}_preds"
    os.makedirs(preds_dir, exist_ok=True)
    out_path = f"{preds_dir}/{allele.replace('*', '_').replace(':', '')}_preds.xls"
    return script(
        f"""
        head -n 2 -q {allele_fname} > {out_path}
        tail -n +3 -q {allele_fnames} >> {out_path}
        """,
        inputs=[f.stage(f.path) for f in allele_paths],
        outputs=[allele, File(out_path).stage(out_path)],
    )


def run_netmhcpan(
    peptides: pd.DataFrame,
    hla_alleles: pd.DataFrame,
    mhc_type: str,
    config: EpitopeGraphConfig,
):
    """
    Run NetMHCpan or NetMHCIIpan for a given set of peptides and HLA alleles, splitting the input peptides
    into chunks and running netmhcpan for each chunk before concatenating the results into the per allele files.
    """
    # Define variables
    n_peptide_chunks = ceil(len(peptides) / config.peptide_chunk_size)
    n_hlas = len(hla_alleles)
    netmhcpan_cmd = (
        config.netmhcpan_cmd if mhc_type == "mhc1" else config.netmhcpanii_cmd
    )
    results_dir = Path(f"{config.results_dir}/MHC_Binding")
    # Split peptides into chunks
    peptide_paths = chunk_peptides(
        peptides, config.peptides_dir, config.peptide_chunk_size
    )
    # For each allele, run netmhcpan for each peptide chunk
    peptide_paths = peptide_paths[:n_peptide_chunks]
    peptide_allele_paths = [
        predict_affinity_netmhcpan(
            peptide_path,
            allele,
            mhc_type,
            results_dir,
            netmhcpan_cmd,
        )
        for allele in hla_alleles["allele"].values
        for peptide_path in peptide_paths
    ]
    # Concatenate the results for each allele
    peptide_allele_paths = group_by_allele(peptide_allele_paths)
    peptide_allele_paths = peptide_allele_paths[:n_hlas]
    allele_paths = [
        concat_allele_files(allele_files[0], allele_files[1], mhc_type, results_dir)
        for allele_files in peptide_allele_paths
    ]
    return allele_paths


def scheduler(max_workers: int = 20, redun_db_path: Path = "redun.db"):
    """
    Create a scheduler for the pipeline.
    """
    return Scheduler(
        config=Config(
            {
                "backend": {
                    f"db_uri": "sqlite:///{redun_db_path}",
                },
                "executors.default": {"max_workers": max_workers},
            }
        )
    )


if __name__ == "__main__":
    # Example data
    # peptides_path = "data/results_redun_test/peptides.txt"
    # peptides = pd.read_csv(peptides_path, header=None, names=["peptide"])
    peptides = pd.DataFrame({"peptide": ["RVSPSKEVV", "VSPSKEVVR", "SPSKEVVRF"]})
    hla_alleles = pd.DataFrame({"allele": ["HLA-B44:04", "HLA-B44:05"]})
    # hla_alleles = pd.DataFrame(
    #     {"allele": ["HLA-DPA10301-DPB11301", "HLA-DPA10201-DPB155801", "DRB1_0701"]}
    # )
    mhc_type = "mhc1"
    params = {
        "fasta_path": "data/input/sar_mer_nuc_protein.fasta",
        "results_dir": "data/results_redun_test",
        "human_proteome_path": "data/input/human_proteome_2023.03.14.fasta.gz",
        "mhc1_alleles_path": "../optivax/scoring/MHC1_allele_mary_cleaned.txt",
        "mhc2_alleles_path": "../optivax/scoring/MHC2_allele_marry.txt",
        "hap_freq_mhc1_path": "../optivax/haplotype_frequency_marry.pkl",
        "hap_freq_mhc2_path": "../optivax/haplotype_frequency_marry2.pkl",
        "peptide_chunk_size": 2,
    }
    config = EpitopeGraphConfig(**params)

    # Run the workflow
    s = scheduler()
    s.load()  # Auto-creates the redun.db file as needed and starts a db connection.
    result = s.run(run_netmhcpan(peptides, hla_alleles, mhc_type, config))
    print(result)
