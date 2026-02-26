import gzip
import numpy as np
import os
import subprocess

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
    Preprocess sequences: Cleans, clusters, filters, translates, and aligns the sequences.
    :fasta_nt_path: Path to the input nucleotide FASTA file.
    :fasta_path: Path to the output FASTA file.
    :prefix: Prefix for intermediate and output files.
    :results_dir: Directory for saving results.
    :seq_identity: Sequence identity threshold for clustering.
    :n_threads: Number of threads for clustering.
    :returns: Path to the preprocessed sequences.
    """

    # Directory for sequence preprocessing files
    seq_dir = results_dir / "Seq_Preprocessing"

    # Clean the FASTA file
    clean_fasta_file(fasta_nt_path)

    # Cluster sequences at NT level to reduce redundancy
    clstr1_path = seq_dir / f"{prefix}_nt_clstr1.fasta"
    clstr1_path = cluster_seqs(fasta_nt_path, clstr1_path, seq_identity, n_threads)

    # Load, truncate, and filter sequences
    seqs_dict_clstr1 = load_and_filter_seqs(clstr1_path)

    # Write filtered sequences to file
    clstr1_path = write_to_fasta(seqs_dict_clstr1, clstr1_path)

    # Re-cluster sequences
    clstr2_path = seq_dir / f"{prefix}_nt_clstr2.fasta"
    clstr2_path = cluster_seqs(clstr1_path, clstr2_path, seq_identity, n_threads)

    # Load sequences and translate to AA
    seqs_dict_aa_clstr2 = load_and_filter_seqs(clstr2_path)
    seqs_dict_aa_clstr2 = translate_seqs(seqs_dict_aa_clstr2)

    # Write AA sequences to file
    aa_clstr2_path = seq_dir / f"{prefix}_aa_clstr2.fasta"
    aa_clstr2_path = write_to_fasta(seqs_dict_aa_clstr2, aa_clstr2_path)

    # Perform final clustering at the protein level
    aa_clstr3_path = seq_dir / f"{prefix}_aa_clstr3.fasta"
    aa_clstr3_path = cluster_seqs(
        aa_clstr2_path, aa_clstr3_path, seq_identity, n_threads, "cd-hit"
    )

    # Align sequences with Clustal Omega
    fasta_path = align_seqs(aa_clstr3_path, fasta_path)

    # Trim alignment
    trimmed_dict = trim_alignment(fasta_path)
    fasta_path = write_to_fasta(trimmed_dict, fasta_path)

    return fasta_path


def load_seqs_from_fasta(fasta_path):
    """
    Load sequences from a FASTA file.
    """
    return list(SeqIO.parse(fasta_path, "fasta"))


def fasta_to_peptides_dict(fasta_path: Path, ks: list = [9, 15]):
    """
    Load a FASTA file and return a dictionary of peptides
    """
    seq_records = load_seqs_from_fasta(fasta_path)
    peptides_dict = {
        record.id: kmerise_simple(str(record.seq), ks) for record in seq_records
    }
    return peptides_dict


def clean_fasta_file(fasta_nt_path: Path) -> None:
    """
    Clean the FASTA file by replacing invalid characters.
    """
    cmd = f"sed -i -E 's/(\|| )/_/g' {fasta_nt_path}"
    subprocess.run(cmd, shell=True)


def load_and_filter_seqs(fasta_path: Path) -> dict:
    """
    Load sequences, truncate to ORFs, and filter based on criteria.
    """
    # Load sequences
    seqs_dict = load_fasta(fasta_path, remove_chars=False)

    # Truncate to ORFs and eliminate the stop codons
    seqs_dict = {seq_id: get_longest_orf(seq) for seq_id, seq in seqs_dict.items()}
    seqs_dict = {seq_id: seq for seq_id, seq in seqs_dict.items() if seq != ""}

    # Filter sequences based on length and ambiguous nucleotides
    seq_lens = [len(seq) for seq in seqs_dict.values()]
    median_seq_len = int(np.median(seq_lens))
    seqs_dict = {
        seq_id: seq
        for seq_id, seq in seqs_dict.items()
        if abs(len(seq) - median_seq_len) / median_seq_len <= 0.1 and "NNNN" not in seq
    }

    return seqs_dict


def write_to_fasta(seqs_dict: dict, output_path: Path) -> Path:
    """
    Write sequences to a FASTA file.
    """
    with output_path.open("w") as f:
        for seq_id, seq in seqs_dict.items():
            f.write(f">{seq_id}\n{seq}\n")
    return Path(output_path)


def translate_seqs(seqs_dict: dict, remove_stop=True) -> dict:
    """
    Translate nucleotide sequences to amino acid sequences removing stop codons.
    """
    if remove_stop:
        return {
            seq_id: str(Seq(seq).translate()).replace("*", "")
            for seq_id, seq in seqs_dict.items()
        }
    else:
        return {seq_id: str(Seq(seq).translate()) for seq_id, seq in seqs_dict.items()}


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


def get_longest_orf(seq: str, table: int = 1):
    """
    Get the longest ORF in a sequence.
    """
    seq = Seq(seq)

    def find_orfs_with_trans(nuc, trans_table):
        orfs = []
        seq_len = len(nuc)
        for frame in range(3):
            try:
                trans = nuc[frame:].translate(trans_table)
            except:
                continue
            trans_len = len(trans)
            aa_start = 0
            aa_end = 0
            while aa_start < trans_len:
                aa_end = trans.find("*", aa_start)
                if aa_end == -1:
                    aa_end = trans_len
                start = frame + aa_start * 3
                end = min(seq_len, frame + aa_end * 3 + 3)
                orf_nucleotide = nuc[start:end]
                orfs.append(orf_nucleotide)
                aa_start = aa_end + 1
        return orfs

    orf_list = find_orfs_with_trans(seq, table)
    if orf_list:
        return str(max(orf_list, key=len))

    else:
        return ""


def trim_alignment(align_path: Path) -> dict:
    """ "
    Trim the alignment to the most common start and end columns.
    """
    align_dict = load_fasta(align_path, remove_chars=False)
    # Convert dictionary values to a list of sequences
    sequences = list(align_dict.values())

    # Find the total number of sequences
    num_sequences = len(sequences)

    # Initialize start and end columns
    start_column = None
    end_column = None

    # Find the start column
    for column in range(len(sequences[0])):
        non_gap_count = sum([1 for seq in sequences if seq[column] != "-"])
        if (
            non_gap_count > num_sequences / 2
        ):  # Checking if more than half of the sequences have a non-gap character
            start_column = column
            break

    # Find the end column
    for column in range(len(sequences[0]) - 1, -1, -1):
        non_gap_count = sum([1 for seq in sequences if seq[column] != "-"])
        if non_gap_count > num_sequences / 2:
            end_column = column
            break

    align_dict = {
        seq_id: seq[start_column : end_column + 1].replace("-", "")
        for seq_id, seq in align_dict.items()
    }

    return align_dict


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
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = "global"
    alignment = aligner.align(first_seq, second_seq)[0]
    seq_length = min(len(first_seq), len(second_seq))
    matches = alignment.score
    percent_match = (matches / seq_length) * 100
    return str(alignment), percent_match


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


def find_kmer_in_msa(msa, k_mer):
    """
    Returns the first found position of a k-mer in a multiple sequence alignment.
    """
    for identifier, sequence in msa.items():
        # Create a version of the sequence without gaps for searching
        sequence_no_gaps = sequence.replace("-", "")
        position_no_gaps = sequence_no_gaps.find(k_mer)

        if position_no_gaps != -1:
            # Found the k-mer in the no-gaps version, now find the equivalent position in the original sequence
            real_position = 0
            while position_no_gaps > 0:
                if sequence[real_position] != "-":
                    position_no_gaps -= 1
                real_position += 1

            # Adjust real_position for the start of the k-mer in the original sequence
            return real_position
    # k-mer not found in any sequence
    return None


# def get_kmers(kmers, k):
#     return [kmer for kmer in kmers if len(kmer) == k]
