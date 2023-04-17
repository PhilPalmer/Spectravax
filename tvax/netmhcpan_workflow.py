import os
import pandas as pd
from pathlib import Path
from redun import task, Scheduler, script, File
from redun.config import Config
from tvax.config import EpitopeGraphConfig


"""
Redun workflow to run NetMHCpan and NetMHCIIpan to predict peptide-HLA binding affinities.
"""

redun_namespace = "netmhcpan"


@task(version="1")
def save_peptides(peptides: pd.DataFrame, peptides_path: Path) -> File:
    """
    Save the input peptides to a file.
    """
    peptides.to_csv(peptides_path, index=False, header=False)
    return File(str(peptides_path))


@task(version="1")
def predict_affinity_netmhcpan(
    allele: str,
    peptides_path: str,
    mhc_type: str = "mhc1",
    results_dir: str = None,
    netmhcpan_cmd: str = None,
    netmhcpan_tmpdir: str = None,
):
    preds_dir = f"{results_dir}/MHC_Binding/net{mhc_type}_preds"
    allele_path = f"{preds_dir}/{allele.replace('*', '_').replace(':', '')}_preds.xls"
    peptides_path = File(os.path.abspath(peptides_path.path))
    peptides_flag = (
        "-p peptides.txt" if mhc_type == "mhc1" else "-inptype 1 -f peptides.txt"
    )
    return script(
        f"""
        {netmhcpan_cmd} {peptides_flag} -a {allele} -BA -xls -xlsfile {allele_path} -tdir {netmhcpan_tmpdir}
        """,
        inputs=[peptides_path.stage("peptides.txt")],
        outputs=[File(allele_path).stage(allele_path)],
    )


def run_netmhcpan(
    peptides: pd.DataFrame,
    hla_alleles: pd.DataFrame,
    mhc_type: str,
    config: EpitopeGraphConfig,
):
    """
    Run NetMHCpan or NetMHCIIpan for a given set of peptides and HLA alleles.
    """
    peptide_path = save_peptides(peptides, config.peptides_path)
    results_dir = config.results_dir
    netmhcpan_tmpdir = config.netmhcpan_tmpdir
    netmhcpan_cmd = (
        config.netmhcpan_cmd if mhc_type == "mhc1" else config.netmhcpanii_cmd
    )
    allele_paths = [
        predict_affinity_netmhcpan(
            allele, peptide_path, mhc_type, results_dir, netmhcpan_cmd, netmhcpan_tmpdir
        )
        for allele in hla_alleles["allele"].values
    ]
    return allele_paths


if __name__ == "__main__":
    # Example data
    peptides = pd.DataFrame({"peptide": ["RVSPSKEVV", "VSPSKEVVR", "SPSKEVVRF"]})
    # hla_alleles = pd.DataFrame({"allele": ["HLA-A*02:01", "HLA-A*02:02"]})
    hla_alleles = pd.DataFrame(
        {"allele": ["HLA-DPA10301-DPB11301", "HLA-DPA10201-DPB155801", "DRB1_0701"]}
    )
    mhc_type = "mhc2"
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
