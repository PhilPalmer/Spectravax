#!/usr/bin/env python

import networkx as nx
import random

from itertools import product
from tvax.config import EpitopeGraphConfig
from tvax.seq import load_fasta, kmerise, kmerise_simple, assign_clades
from tvax.score import add_scores
from typing import Optional


"""
Construct the potential T-cell epitope (PTE) graph.
"""


def build_epitope_graph(
    config: Optional[EpitopeGraphConfig] = None,
    seqs_dict: dict = None,
    kmers_dict: dict = None,
) -> nx.Graph:
    """
    Construct an epitope graph from a configuration dictionary.
    """
    if not kmers_dict:
        # Load the FASTA file
        if not seqs_dict:
            seqs_dict = load_fasta(config.fasta_path)
        N = len(seqs_dict)
        # Assign the sequences to clades
        clades_dict = assign_clades(seqs_dict, config)
        # Split the sequences into k-mers
        kmers_dict = kmerise(seqs_dict, clades_dict, config.k)
        # Add scores/weights to the k-mers
        kmers_dict = add_scores(kmers_dict, clades_dict, N, config)

    # Construct the graph
    G = nx.DiGraph()
    # Add nodes
    for kmer, attrs in kmers_dict.items():
        G.add_node(kmer, **attrs)
    # Add edges between overlapping k-mers of any length
    # where the last k−1 characters of ea match the first k−1 characters of eb
    for ea, eb in product(G.nodes(), G.nodes()):
        shortest_k = min(len(ea), len(eb))
        if not G.has_edge(ea, eb) and ea[-shortest_k + 1 :] == eb[: shortest_k - 1]:
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
    Add begin and end nodes to the graph (for computational convenience).
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
    Add position attribute to the graph (for plotting convenience).
    """

    def _calc_path_lengths(paths: dict) -> dict:
        """
        Calculate the length of each path in the graph.
        """
        path_lengths = {}
        # Calculate the lengths of the paths
        for kmer, path in paths.items():
            for mb in path:
                k_mb = len(mb)
                # If the current kmer is BEGIN, initialise the path length to 0
                if mb == "BEGIN":
                    path_len = 0
                # If it's the first k-mer, add the length of the k-mer
                elif ma == "BEGIN":
                    path_len += k_mb
                # For all additional kmers in the path, add the length of the new amino acids between the k-mers
                else:
                    min_k = min(len(ma), k_mb)
                    overlap = min_k - 1
                    path_len += k_mb - overlap
                # Define the previous k-mer as the current k-mer
                ma = mb
            path_lengths[kmer] = path_len
        return path_lengths

    for node in G.nodes:
        G.nodes[node]["len"] = len(node)
    if not aligned:
        paths = nx.shortest_path(G, source="BEGIN", weight="len")
        path_lengths = _calc_path_lengths(paths)
        pos = {
            n: (d, f(G, n))
            for n, d in path_lengths.items()
            if n != "BEGIN" and n != "END"
        }
        nx.set_node_attributes(G, pos, "pos")
    # Ensure the 'END' node is always at the end of the graph
    end_pos = (
        max([pos[0] for pos in list(nx.get_node_attributes(G, "pos").values())]) + 1
    )
    nx.set_node_attributes(G, {"END": (end_pos, 0)}, "pos")
    return G


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
    edges = kmerise_simple(cycle, ks=[2])
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
