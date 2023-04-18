import os
import pandas as pd
from pathlib import Path
from redun import task, Scheduler, script, File
from redun.config import Config
from redun.functools import map_
from tvax.config import EpitopeGraphConfig
from typing import List

"""
Redun workflow to run NetMHCpan and NetMHCIIpan to predict peptide-HLA binding affinities.
"""

redun_namespace = "netmhcpan"


@task(version="1")
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


@task(version="1")
def predict_affinity_netmhcpan(
    peptides_path: File,
    allele: str,
    mhc_type: str = "mhc1",
    results_dir: str = None,
    netmhcpan_cmd: str = None,
):
    peptide_chunk_idx = peptides_path.path.split("_")[-1].split(".")[0]
    preds_dir = f"{results_dir}/MHC_Binding/net{mhc_type}_preds"
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
    # Split peptides into chunks
    peptide_paths = chunk_peptides(
        peptides, config.peptides_dir, config.peptide_chunk_size
    )
    netmhcpan_cmd = (
        config.netmhcpan_cmd if mhc_type == "mhc1" else config.netmhcpanii_cmd
    )
    # Run netmhcpan for each allele and peptide chunk
    peptide_allele_paths = []
    for allele in hla_alleles["allele"].values:
        # Partially apply all the arguments that don't change.
        predict = predict_affinity_netmhcpan.partial(
            allele=allele,
            mhc_type=mhc_type,
            results_dir=config.results_dir,
            netmhcpan_cmd=netmhcpan_cmd,
        )
        peptide_allele_paths.append(map_(predict, peptide_paths))
    # TODO: Get the peptide allele paths for each allele and concatenate the results
    return peptide_allele_paths


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
    }
    config = EpitopeGraphConfig(**params)

    # Run the workflow
    # scheduler = Scheduler()
    scheduler = Scheduler(
        config=Config(
            {
                "backend": {
                    "db_uri": "sqlite:///redun.db",
                }
            }
        )
    )
    scheduler.load()  # Auto-creates the redun.db file as needed and starts a db connection.
    result = scheduler.run(run_netmhcpan(peptides, hla_alleles, mhc_type, config))
    print(result)
