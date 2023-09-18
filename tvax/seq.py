import gzip
import numpy as np
import os
import subprocess

from Bio import pairwise2 as pw2
from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path
from tvax.config import EpitopeGraphConfig
from tvax.pca_protein_rank import pca_protein_rank


"""
Utils to process sequences.
"""


def preprocess_seqs(
    fasta_nt_path: Path,
    fasta_path: Path,
    prefix: str,
    results_dir: Path,
    seq_identity: float,
    n_threads: int,
) -> Path:
    """
    Preprocess sequences.
    """
    # Replace invalid characters in the FASTA file
    subprocess.run(f"sed -i -E 's/(\|| )/_/g' {fasta_nt_path}", shell=True)
    # n_seqs = int(subprocess.check_output(f"grep '>' {fasta_nt_path} | wc -l", shell=True))
    # print(f"Number of initial sequences: {n_seqs}")

    # Cluster sequences at NT level to reduce redundancy
    cluster_path = f"{results_dir}/Seq_Preprocessing/{prefix}_clstr.fasta"
    cluster_path = cluster_seqs(fasta_nt_path, cluster_path, seq_identity, n_threads)

    # Truncate to start and stop codons, then eliminate the stop codons
    seqs_dict = load_fasta(cluster_path, remove_chars=False)
    # print(f"Number of sequences after clustering: {len(seqs_dict)}")
    seqs_dict = {
        seq_id: truncate_to_start_stop(seq) for seq_id, seq in seqs_dict.items()
    }
    seqs_dict = {seq_id: seq for seq_id, seq in seqs_dict.items() if seq is not ""}
    # print(f"Number of sequences after truncating: {len(seqs_dict)}")

    # Filter sequences
    seq_lens = [len(seq) for seq in seqs_dict.values()]
    median_seq_len = int(np.median(seq_lens))
    seqs_dict = {
        seq_id: seq
        for seq_id, seq in seqs_dict.items()
        # Remove sequences that deviate from the median length by >10%
        if abs(len(seq) - median_seq_len) / median_seq_len <= 0.1 and
        # Remove sequences containing >3 successive ambiguous nucleotides
        "NNNN" not in seq
    }
    # print(f"Number of sequences after filtering: {len(seqs_dict)}")
    with open(cluster_path, "w") as f:
        for seq_id, seq in seqs_dict.items():
            f.write(f">{seq_id}\n{seq}\n")

    # Re-cluster sequences
    cluster2_path = f"{results_dir}/Seq_Preprocessing/{prefix}_clstr2.fasta"
    cluster2_path = cluster_seqs(cluster_path, cluster2_path, seq_identity, n_threads)

    # Translate sequences to AA and save to file
    seqs_dict = load_fasta(cluster2_path, remove_chars=False)
    # print(f"Number of sequences after second clusterng: {len(seqs_dict)}")
    seqs_dict = {seq_id: str(Seq(seq).translate()) for seq_id, seq in seqs_dict.items()}
    with open(fasta_path, "w") as f:
        for seq_id, seq in seqs_dict.items():
            f.write(f">{seq_id}\n{seq}\n")

    # Align sequences with Clustal Omega
    align_path = f"{results_dir}/Seq_Preprocessing/{prefix}_align.fasta"
    align_path = align_seqs(fasta_path, align_path)

    return Path(fasta_path)


def cluster_seqs(
    fasta_path: Path,
    cluster_path: Path,
    seq_identity: float = 0.95,
    n_threads=4,
    cd_hit_path: str = "cd-hit-est",
) -> Path:
    """
    Cluster sequences.
    """
    subprocess.run(
        f"{cd_hit_path} -i {fasta_path}  -o {cluster_path} -c {seq_identity} -T {n_threads}",
        shell=True,
    )
    return Path(cluster_path)


def align_seqs(
    input_file: Path, output_file: Path, clustal_omega_path: str = "clustalo"
) -> Path:
    """
    Align sequences using Clustal Omega.
    """
    subprocess.run(
        f"{clustal_omega_path} --infile {input_file} --outfile {output_file} --auto --force",
        shell=True,
    )

    return Path(output_file)


def truncate_to_start_stop(
    seq: str, start_codon: str = "ATG", stop_codons: list = ["TAA", "TAG", "TGA"]
):
    """
    Truncate sequence to start and stop codons and remove the stop codon.
    Only consider in-frame stop codons.
    """
    # Remove all gaps
    seq = seq.replace("-", "")

    # Find the start codon
    start_index = seq.find(start_codon)
    if start_index == -1:
        return ""

    # Initialize stop_index to None (no valid stop codon found yet)
    stop_index = None

    # Check each potential stop codon
    for stop_codon in stop_codons:
        current_stop_index = (
            start_index + 3
        )  # Start checking 3 nucleotides after the start codon

        while (
            current_stop_index < len(seq) - 2
        ):  # Ensure there's room for a 3-nucleotide codon
            current_stop_index = seq.find(stop_codon, current_stop_index)

            # If no more occurrences of this stop codon, move to the next one
            if current_stop_index == -1:
                break

            # Check if this stop codon is in-frame
            if (current_stop_index - start_index) % 3 == 0:
                # If this is the first in-frame stop codon found, or it's closer than a previously found one
                if stop_index is None or current_stop_index < stop_index:
                    stop_index = current_stop_index
                break

            # Move past this occurrence of the stop codon to continue searching
            current_stop_index += 3

    if stop_index is None:
        return ""

    truncated_seq = seq[start_index:stop_index]
    return truncated_seq


def load_fasta(fasta_path: Path, remove_chars: bool = True) -> dict:
    """
    Load a FASTA file into a dictionary.
    """
    open_func = gzip.open if fasta_path.suffix == ".gz" else open
    with open_func(fasta_path, "rt") as f:
        fasta_seqs = SeqIO.parse(f, "fasta")
        return {
            seq.id: remove_invalid_chars(str(seq.seq)) if remove_chars else str(seq.seq)
            for seq in fasta_seqs
        }


def remove_invalid_chars(
    seq: str,
    aa_dict: dict = {"B": ["D", "N"], "J": ["L", "I"], "Z": ["Q", "E"]},
    invalid_chars: list = ["-", "*"],
) -> str:
    """
    Remove invalid characters from a sequence.
    Including replacing ambiguous amino acid codes with the most appropriate canonical amino acid based on the context of the protein
    If the canonical amino acids are equally likely, the first one in the list is used.
    """
    # Remove invalid characters
    for invalid_char in invalid_chars:
        seq = seq.replace(invalid_char, "")
    # Replace ambiguous amino acids with the most common amino acid
    for ambig_aa in aa_dict.keys():
        if ambig_aa in seq:
            # Count the number of times each of the possible amino acids appears in the sequence
            aa_counts = {aa: seq.count(aa) for aa in aa_dict[ambig_aa]}
            # Replace the ambiguous amino acid with the most common amino acid
            canon_aa = max(aa_counts, key=aa_counts.get)
            seq = seq.replace(ambig_aa, canon_aa)
    return seq


def kmerise_simple(seq: str, ks: list = [9]):
    """
    Returns a list of k-mers of lengths k for a given string of amino acid sequence
    """
    return [seq[i : i + k] for k in ks for i in range(len(seq) - k + 1)]


def kmerise(seqs_dict: dict, clades_dict: dict, ks: list = [9]) -> dict:
    """
    Split sequences into k-mers and return a dictionary of k-mers with their clades and counts.
    """
    kmers_dict = {}
    for seq_id, seq in seqs_dict.items():
        clade = clades_dict[seq_id]
        for k in ks:
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                if kmer in kmers_dict:
                    kmers_dict[kmer]["count"] += 1
                    if clade not in kmers_dict[kmer]["clades"]:
                        kmers_dict[kmer]["clades"].append(clade)
                else:
                    kmers_dict[kmer] = {"count": 1, "clades": [clade]}
    return kmers_dict


def assign_clades(seqs_dict: dict, config: EpitopeGraphConfig) -> dict:
    """
    Assign clades to sequences.
    """
    if config.equalise_clades:
        # Perform multiple sequence alignment
        msa_path = msa(config.fasta_path, config.msa_path)
        # Assign clades to sequences
        clades_dict, comp_df = pca_protein_rank(
            msa_path, n_clusters=config.n_clusters, plot=False
        )
    else:
        clades_dict = {seq_id: 1 for seq_id in seqs_dict}
    return clades_dict


def msa(fasta_path: Path, msa_path: Path, overwrite: bool = False) -> Path:
    """
    Perform multiple sequence alignment.
    """
    if not os.path.exists(msa_path) or overwrite:
        subprocess.run(f"mafft --auto {fasta_path} > {msa_path}", shell=True)
    return msa_path


def compute_percent_match(first_seq, second_seq):
    """
    Compute the percent match between two sequences.
    :param first_seq: First sequence
    :param second_seq: Second sequence
    :return: Percent match
    """
    global_align = pw2.align.globalxx(first_seq, second_seq)
    seq_length = min(len(first_seq), len(second_seq))
    matches = global_align[0][2]
    percent_match = (matches / seq_length) * 100
    return pw2.format_alignment(*global_align[0]), percent_match


def path_to_seq(path: list) -> str:
    """
    Given an ordered list of overlapping k-mers of varying lengths (path) merge them to create an AA string sequence
    """
    seq = path[0]
    for i in range(1, len(path)):
        ma = path[i - 1]
        mb = path[i]
        overlap_l = min(len(ma), len(mb)) - 1
        seq += mb[overlap_l:]
    return seq


def seq_to_kmers(seq, k, G):
    """
    Convert a path to a list of all possible k-mers in the k-mer graph
    """
    kmers = kmerise_simple(seq, k)
    kmers = [kmer for kmer in kmers if kmer in G.nodes()]
    return kmers


# def get_kmers(kmers, k):
#     return [kmer for kmer in kmers if len(kmer) == k]
