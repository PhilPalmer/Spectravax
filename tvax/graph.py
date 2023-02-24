#!/usr/bin/env python

import mhcflurry
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import subprocess

from Bio import SeqIO
from itertools import product
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tvax.config import EpitopeGraphConfig
from tvax.pca_protein_rank import pca_protein_rank
from typing import Optional


#############################
# Construct the epitope graph
#############################


def build_epitope_graph(config: Optional[EpitopeGraphConfig] = None) -> nx.Graph:
    """
    Construct an epitope graph from a configuration dictionary.
    """
    # Load the FASTA file
    seqs_dict = load_fasta(config.fasta_path)
    N = len(seqs_dict)

    # Assign the sequences to clades
    clades_dict = assign_clades(seqs_dict, config)

    # Split the sequences into k-mers
    kmers_dict = kmerise(seqs_dict, clades_dict, config.k)

    # Add scores/weights to the k-mers
    if config.weights.frequency:
        kmers_dict = add_frequency_score(kmers_dict, N)
    if (
        config.weights.processing
        or config.weights.strong_mhc_binding
        or config.weights.weak_mhc_binding
    ):
        kmers_dict = add_mhcflurry_scores(kmers_dict, config)
    if config.equalise_clades:
        kmers_dict = add_clade_weights(kmers_dict, clades_dict, N)
    kmers_dict = add_score(kmers_dict, config)

    # Construct the graph
    G = nx.DiGraph()
    # Add nodes
    for kmer, attrs in kmers_dict.items():
        G.add_node(kmer, **attrs)
    # Add edges - where the last k−1 characters of ea match the first k−1 characters of eb
    for ea, eb in product(G.nodes(), G.nodes()):
        if not G.has_edge(ea, eb) and ea[1:] == eb[:-1]:
            G.add_edge(ea, eb, colour=config.edge_colour)
    # Decycle graph
    if config.decycle:
        G = decycle_graph(G)
    # Add begin/end nodes and positions
    G = add_begin_end_nodes(G, config.edge_colour)
    G = add_pos(G, config.aligned)
    return G


def add_begin_end_nodes(G: nx.Graph, edge_colour: str) -> nx.Graph:
    """
    Add begin and end nodes to the graph.
    """
    # Get all nodes lacking predecessors and successors
    begin_nodes = [e for e in list(G.nodes) if not P(G, e)]
    end_nodes = [e for e in list(G.nodes) if not S(G, e)]
    # Add begin and end nodes
    empty_attrs = {attr: 0 for attr in G.nodes[begin_nodes[0]].keys()}
    if "clades" in empty_attrs:
        empty_attrs["clades"] = []
    G.add_node("BEGIN", **empty_attrs, pos=(0, 0))
    G.add_node("END", **empty_attrs, pos=(0, 0))
    # Add edges
    G.add_edges_from([("BEGIN", e) for e in begin_nodes], colour=edge_colour)
    G.add_edges_from([(e, "END") for e in end_nodes], colour=edge_colour)
    return G


def add_pos(G: nx.Graph, aligned: bool = False) -> nx.Graph:
    """
    Add position attribute to the graph (useful for plotting).
    """
    if not aligned:
        paths = nx.shortest_path_length(G, source="BEGIN")
        pos = {n: (d, f(G, n)) for n, d in paths.items() if n != "BEGIN" and n != "END"}
        nx.set_node_attributes(G, pos, "pos")
    # Ensure the 'END' node is always at the end of the graph
    end_pos = (
        max([pos[0] for pos in list(nx.get_node_attributes(G, "pos").values())]) + 1
    )
    nx.set_node_attributes(G, {"END": (end_pos, 0)}, "pos")
    return G


############################
# Utils to process sequences
############################


def load_fasta(fasta_path: Path) -> dict:
    """
    Load a FASTA file into a dictionary.
    """
    fasta_seqs = SeqIO.parse(fasta_path, "fasta")
    seqs_dict = {seq.id: str(seq.seq) for seq in fasta_seqs}
    return seqs_dict


def kmerise_simple(seq, k):
    """
    Returns a list of k-mers of length k for a given string of amino acid sequence
    """
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def kmerise(seqs_dict: dict, clades_dict: dict, k: int = 9) -> dict:
    """
    Split sequences into k-mers and return a dictionary of k-mers with their clades and counts.
    """
    kmers_dict = {}
    for seq_id, seq in seqs_dict.items():
        clade = clades_dict[seq_id]
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if kmer in kmers_dict:
                kmers_dict[kmer]["count"] += 1
                if clade not in kmers_dict[kmer]["clades"]:
                    kmers_dict[kmer]["clades"].append(clade)
            else:
                kmers_dict[kmer] = {"count": 1, "clades": [clade]}
    return kmers_dict


def msa(fasta_path: Path, msa_path: Path) -> Path:
    """
    Perform multiple sequence alignment.
    """
    if not os.path.exists(msa_path):
        subprocess.run(f"mafft --auto {fasta_path} > {msa_path}", shell=True)
    return msa_path


###########################
# Utils to calculate scores
###########################


def add_score(kmers_dict: dict, config: EpitopeGraphConfig) -> dict:
    """
    Add a score to each k-mer.
    """
    # Calculate the total score for each k-mer by summing the weighted scores
    # TODO: Check if dividing by the sum of the weights is necessary
    for kmer in kmers_dict:
        kmers_dict[kmer]["score"] = sum(
            [kmers_dict[kmer][name] * val for name, val in config.weights]
        )
        # If equalise clades is True, then multiple the total score by the clade weight
        if config.equalise_clades:
            kmers_dict[kmer]["score"] *= kmers_dict[kmer]["clade_weight"]
    # Min-max scale the scores
    scores = [kmers_dict[kmer]["score"] for kmer in kmers_dict]
    minmax_scaler = MinMaxScaler()
    scores = minmax_scaler.fit_transform(np.array(scores).reshape(-1, 1)).reshape(-1)
    for i, kmer in enumerate(kmers_dict):
        kmers_dict[kmer]["score"] = scores[i]
    return kmers_dict


def add_frequency_score(kmers_dict: dict, N: int) -> dict:
    """
    Add frequency score to k-mers.
    """
    for kmer in kmers_dict:
        kmers_dict[kmer]["frequency"] = kmers_dict[kmer]["count"] / N
    return kmers_dict


def add_mhcflurry_scores(kmers_dict: dict, config: EpitopeGraphConfig) -> dict:
    """
    Add MHCflurry scores to k-mers.
    """
    # Define vars
    agg_dict = {
        "processing": "mean",
        "strong_mhc_binding": "sum",
        "weak_mhc_binding": "sum",
    }
    cached_scores = False
    cols = ["processing", "strong_mhc_binding", "weak_mhc_binding"]
    peptides = list(kmers_dict.keys())

    # Check if the scores are already cached
    if config.immune_scores_path and os.path.exists(config.immune_scores_path):

        df = pd.read_csv(config.immune_scores_path)
        # TODO: Check if the alleles and thresholds are the same
        if set(df["peptide"]) == set(peptides):
            cached_scores = True

    # Only compute MHCflurry predictions if not already cached
    if not cached_scores:
        dfs = []
        median_scaler = MedianScaler()
        minmax_scaler = MinMaxScaler()
        predictor = mhcflurry.Class1PresentationPredictor.load()
        # TODO: Check if model needs to be downloaded !mhcflurry-downloads fetch models_class1_presentation

        # Compute the scores for all peptides and alleles
        for allele in config.alleles:
            df = predictor.predict(
                peptides, [allele], include_affinity_percentile=True, verbose=0
            )
            # Determine if the peptide binds to the MHC (based on the percentile threshold)
            df["weak_mhc_binding"] = df["affinity_percentile"].apply(
                lambda x: 1 if x < config.weak_percentile_threshold else 0
            )
            df["strong_mhc_binding"] = df["affinity_percentile"].apply(
                lambda x: 1 if x < config.strong_percentile_threshold else 0
            )
            df = df.rename(
                columns={"best_allele": "allele", "processing_score": "processing"}
            )
            dfs.append(df)

        # Concatenate the dataframes
        df = pd.concat(dfs, ignore_index=True)
        # Aggregate the scores for each peptide across the alleles
        df = df.groupby("peptide").agg(agg_dict).reset_index()
        # Convert MHC binding to the fraction of bound alleles
        df["weak_mhc_binding"] = df["weak_mhc_binding"] / len(config.alleles)
        df["strong_mhc_binding"] = df["strong_mhc_binding"] / len(config.alleles)
        # Scale the scores
        df[["weak_mhc_binding", "strong_mhc_binding"]] = median_scaler.fit_transform(
            df[["weak_mhc_binding", "strong_mhc_binding"]]
        )
        df[["processing"]] = minmax_scaler.fit_transform(df[["processing"]])

        df.to_csv(config.immune_scores_path, index=False)

    # Update the k-mers dictionary with the scores
    mhcflurry_dict = df[cols + ["peptide"]].set_index("peptide").to_dict()
    for kmer in kmers_dict:
        for col in cols:
            kmers_dict[kmer][col] = mhcflurry_dict[col][kmer]

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


def add_clade_weights(kmers_dict: dict, clades_dict: dict, N: int) -> dict:
    """
    Add clades and the average clade weight (total number of sequences / number of sequences in each clade) to k-mers.
    """
    # Compute the clade weights for each clade
    clade_weight_dict = {}
    for seq_id, clade in clades_dict.items():
        if clade not in clade_weight_dict:
            clade_weight_dict[clade] = 1
        else:
            clade_weight_dict[clade] += 1
    clade_weight_dict = {
        clade: N / clade_weight_dict[clade] for clade in clade_weight_dict
    }
    # Add the clade weights to the k-mers
    for kmer, d in kmers_dict.items():
        kmers_dict[kmer]["clade_weight"] = np.mean(
            [clade_weight_dict[clade] for clade in d["clades"]]
        )
    return kmers_dict


class MedianScaler:
    """
    Scale the data to the range [min_value, max_value] using the median and standard deviation.
    """

    def __init__(self, min_value=0, max_value=1):
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X):
        self.median_ = np.median(X)
        self.std_ = np.std(X)

    def transform(self, X):
        z_scores = (X - self.median_) / self.std_
        scaled = (z_scores - np.min(z_scores)) / (np.max(z_scores) - np.min(z_scores))
        scaled = scaled * (self.max_value - self.min_value) + self.min_value
        return scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


############################################
# Utils to get the min/max index from a list
############################################


def argmax(lst):
    """
    Returns the index for the maximum value in a list
    """
    return lst.index(max(lst))


def argmin(lst):
    """
    Returns the index for the minimum value in a list
    """
    return lst.index(min(lst))


#####################################
# Utils to retrieve info from a graph
#####################################


def P(G, e):
    """
    Returns the predecessors for a given graph G and node e
    :param G: Directed Graph containing epitopes
    :param e: String for a given potential T-cell epitope (PTE)
    :returns: List of predecessors
    """
    return list(G.predecessors(e))


def S(G, e):
    """
    Returns the successors for a given graph G and node e
    :param G: Directed Graph containing epitopes
    :param e: String for a given potential T-cell epitope (PTE)
    :returns: List of successors
    """
    return list(G.successors(e))


def f(G, e, f="score"):
    """
    Returns the feature for a given epitope e eg frequency or score in the population
    :param G: Directed Graph containing epitopes
    :param e: String for a given potential T-cell epitope (PTE)
    :param e: String for the node feature (default = 'Score')
    :returns: Float for the epitope feature eg score
    """
    return G.nodes[e][f]


############################################
# Decycling - remove all cycles from a graph
############################################


def decycle_graph(G):
    """
    Return a Directed Graph with no cycles
    :param G: Directed Graph containing epitopes
    :returns: Directed Graph containing epitopes and no cycles
    """
    # j is a list of all compnents; each component is a list of nodes in G
    components = list(nx.strongly_connected_components(G))
    # Discard all single node components - no cycles there!
    components = [j for j in components if len(j) != 1]
    if len(components) != 0:
        for j in components:
            # Randomly choose two nodes from the selected component
            ea, eb = random.sample(list(j), k=2)
            cycle = cycle_from_two_nodes(G, ea, eb)
            if cycle:
                ea, eb = weak_edge_in_cycle(G, cycle)
                G.remove_edge(ea, eb)
                # Repeat until graph is acyclic
                G = decycle_graph(G)
    return G


def cycle_from_two_nodes(G, ea, eb):
    """
    Returns the cycle (i.e. path that starts and ends in with the same epitope) for two nodes
    :param G: Directed Graph containing epitopes
    :param ea: String for the first given potential T-cell epitope (PTE)
    :param eb: String for the second given potential T-cell epitope (PTE)
    :returns: List of epitope strings on path that is a cycle
    """
    try:
        path_ab = nx.shortest_path(G, source=ea, target=eb)
        path_ba = nx.shortest_path(G, source=eb, target=ea)
        # Merge two paths into a cycle
        cycle = path_ab[:-1] + path_ba
    except nx.NetworkXNoPath:
        cycle = []
    return cycle


def weak_edge_in_cycle(G, cycle):
    """
    Returns the weak edge (edge with the lowest score) in a cycle
    :param G: Directed Graph containing epitopes
    :param cycle: List of epitope strings on path that is a cycle
    :returns: Tuple for the weak edge containing the two epitope strings
    """
    edges = kmerise_simple(cycle, k=2)
    values = []
    for ea, eb in edges:
        # v is heuristic “value” of edge
        v = f(G, ea) + f(G, eb)
        # Add value if cutting edge would isolate ea
        if len(S(G, ea)) == 1:
            v = v + f(G, ea)
        # Add value if cutting edge would isolate eb
        if len(P(G, eb)) == 1:
            v = v + f(G, eb)
        values.append(v)
    ea, eb = edges[argmin(values)]
    return ea, eb
