import gzip
import os
import subprocess

from Bio import pairwise2 as pw2
from Bio import SeqIO
from pathlib import Path
from tvax.config import EpitopeGraphConfig
from tvax.pca_protein_rank import pca_protein_rank


"""
Utils to process sequences.
"""


def load_fasta(fasta_path: Path) -> dict:
    """
    Load a FASTA file into a dictionary.
    """
    open_func = gzip.open if fasta_path.suffix == ".gz" else open
    with open_func(fasta_path, "rt") as f:
        fasta_seqs = SeqIO.parse(f, "fasta")
        return {
            seq.id: replace_ambiguous_aas(str(seq.seq)).replace("-", "")
            for seq in fasta_seqs
        }


def replace_ambiguous_aas(
    seq: str, aa_dict: dict = {"B": ["D", "N"], "J": ["L", "I"], "Z": ["Q", "E"]}
):
    """
    Replaces ambiguous amino acid codes with the most appropriate canonical amino acid based on the context of the protein
    If the canonical amino acids are equally likely, the first one in the list is used.
    """
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
    Returns an AA string for a list of epitopes (path)
    """
    seq = [path[0]] + [e[-1] for e in path[1:]]
    return "".join(seq)
