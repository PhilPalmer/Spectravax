#!/usr/bin/env python

import networkx as nx
import pandas as pd

from tqdm.notebook import tqdm
from tvax.config import EpitopeGraphConfig
from tvax.graph import argmax, f, P
from tvax.eval import compute_population_coverage
from tvax.score import load_haplotypes, load_overlap


"""
Design a vaccine (cocktail).
"""


def design_vaccines(G: nx.Graph, config: EpitopeGraphConfig) -> list:
    """
    Design a vaccine cocktail from a graph of epitopes
    """
    # Find the optimal path(s) through the graph of epitopes
    Q = cocktail(G, config.m, config=config)
    return Q


###############################################
# Find optimal path through a graph of epitopes
###############################################


def find_optimal_path(G: nx.Graph, config) -> list:
    """
    Returns the optimal path through a graph of epitopes
    :param G: Directed Graph containing epitopes
    :returns: List of epitope strings on the optimal path
    """
    # Load the haplotypes and overlap data
    peptides = [e for e in G.nodes if e not in ["BEGIN", "END"]]
    hap_freq_path_mhc1 = config.hap_freq_mhc1_path
    hap_freq_path_mhc2 = config.hap_freq_mhc2_path
    hap_freq_mhc1, average_frequency_mhc1 = load_haplotypes(hap_freq_path_mhc1)
    hap_freq_mhc2, average_frequency_mhc2 = load_haplotypes(hap_freq_path_mhc2)
    overlap_haplotypes_mhc1 = load_overlap(peptides, hap_freq_mhc1, config, "mhc1")
    overlap_haplotypes_mhc2 = load_overlap(peptides, hap_freq_mhc2, config, "mhc2")
    # Forward loop - compute F(e)
    epitopes = sorted(G.nodes, key=lambda e: G.nodes[e]["pos"][0])
    for e in tqdm(epitopes):
        F(
            G,
            e,
            config,
            average_frequency_mhc1,
            average_frequency_mhc2,
            overlap_haplotypes_mhc1,
            overlap_haplotypes_mhc2,
        )
    # Backward loop - build the path that achieves the maximal score
    path = backward(G)
    return path


def get_optimal_p(G, e, path, config):
    """
    For a given node e, returns the predecessor nodes with the highest F(e) recursively until the start node
    """
    predecessors = P(G, e)
    if predecessors == ["BEGIN"]:
        return path
    else:
        predecessors_Fe = [F(G, pe, config) for pe in predecessors]
        i = argmax(predecessors_Fe)
        path.insert(0, predecessors[i])
        return get_optimal_p(G, predecessors[i], path, config)


def F(
    G: nx.Graph,
    e: str,
    config,
    average_frequency_mhc1: pd.DataFrame = None,
    average_frequency_mhc2: pd.DataFrame = None,
    overlap_haplotypes_mhc1: pd.DataFrame = None,
    overlap_haplotypes_mhc2: pd.DataFrame = None,
) -> float:
    """
    Returns the maximum total score over all paths that end in e
    :param G: Directed Graph containing epitopes
    :param e: String for a given potential T-cell epitope (PTE)
    :returns: Float for the maximum total epitope score
    """
    # Use precomputed F(e) if it already exists for the epitope
    if "F(e)" not in G.nodes[e]:
        if e == "BEGIN":
            Fe = 0
        elif e == "END":
            Fe_dict = nx.get_node_attributes(G, "F(e)")
            Fe = max(list(Fe_dict.values()))
        else:
            predecessors = P(G, e)
            if predecessors == ["BEGIN"]:
                # If the set of predecessors P(e) is empty, then F(e) = compute_score(e)
                Fe = compute_score(
                    G,
                    e,
                    [e],
                    config,
                    average_frequency_mhc1,
                    average_frequency_mhc2,
                    overlap_haplotypes_mhc1,
                    overlap_haplotypes_mhc2,
                )
            else:
                # Get all nodes on the optimal path from the start node to the current node
                path = get_optimal_p(G, e, [e], config)
                # Then compute the F(e) i.e. score for all of the nodes in the path
                Fe = compute_score(
                    G,
                    e,
                    path,
                    config,
                    average_frequency_mhc1,
                    average_frequency_mhc2,
                    overlap_haplotypes_mhc1,
                    overlap_haplotypes_mhc2,
                )
                print(f"Computed F({e}) = {Fe} using {len(path)} peptides")

        # Save F(e) to the graph for this epitope
        nx.set_node_attributes(G, {e: Fe}, "F(e)")
    return f(G, e, f="F(e)")


def compute_score(
    G: nx.Graph,
    e: str,
    path: list,
    config: EpitopeGraphConfig,
    average_frequency_mhc1: pd.DataFrame = None,
    average_frequency_mhc2: pd.DataFrame = None,
    overlap_haplotypes_mhc1: pd.DataFrame = None,
    overlap_haplotypes_mhc2: pd.DataFrame = None,
) -> float:
    """
    Returns the maximum total score over all paths that end in e
    """
    # If population coverage weights are set compute these scores
    if config.weights.population_coverage_mhc1:
        nx.set_node_attributes(
            G,
            {
                e: compute_population_coverage(
                    path,
                    config.n_target,
                    config,
                    "mhc1",
                    average_frequency_mhc1,
                    overlap_haplotypes_mhc1,
                )
            },
            "population_coverage_mhc1",
        )
    if config.weights.population_coverage_mhc2:
        nx.set_node_attributes(
            G,
            {
                e: compute_population_coverage(
                    path,
                    config.n_target,
                    config,
                    "mhc2",
                    average_frequency_mhc2,
                    overlap_haplotypes_mhc2,
                )
            },
            "population_coverage_mhc2",
        )
    # Compute the total score by taking a weighted average of the scores
    score = sum(
        # Get all of the node attribute scores for this epitope
        [G.nodes[e][name] * val for name, val in config.weights if val > 0]
    ) / sum([val for name, val in config.weights if val > 0])
    # Multiply the score by the clade weight
    score *= G.nodes[e]["clade_weight"]
    return score


def backward(G: nx.Graph, path: list = []) -> list:
    """
    Returns the path that achieves the maximal score
    :param G: Directed Graph containing epitopes
    :param path: List of epitope strings to complete (deafult=[])
    :returns: List of epitope strings on path that achieve maximum score
    """
    # Get the precomputed F(e) from the graph for all epitopes
    Fe_dict = nx.get_node_attributes(G, "F(e)")
    if not path:
        # Get the epitope with the maximum F(e) as the final epitope in our optimal path
        end_nodes = {e: Fe_dict[e] for e in P(G, "END")}
        path = [max(end_nodes, key=end_nodes.get)]
    # Get the most recently added epitope e and it's predecessors P(e)
    e = path[0]
    predecessors = P(G, e)
    if predecessors[0] != "BEGIN":
        # Add the best (highest F(e)) predecessor P(e) of epitope e to our path
        i = argmax([Fe_dict[pe] for pe in predecessors])
        path.insert(0, predecessors[i])
        # Repeat until you get to the start
        backward(G, path)
    return path


###########################################################
# Cocktail: Find (and iteratively refine) a set of antigens
###########################################################


def cocktail(
    G: nx.Graph, m: int, refine: bool = True, score: str = "score", config=None
) -> list:
    """
    Returns a list of m antigens
    :param G: Directed Graph containing epitopes
    :param m: Integer for number of antigens
    :param refine: Boolean for if the antigens should be iteratively refined
    :returns: List containing m antigens
    """
    Q = []  # vaccine
    # Save original epitope score so it can be reset later
    score_dict = nx.get_node_attributes(G, score)
    # TODO: If m == 1, then just return the optimal path for computational efficiency
    for n in range(0, m):
        # Compute and save next antigen sequence
        q = find_optimal_path(G, config)
        # Add q to vaccine
        Q.append(q)
        # No credit for including e in subsequent antigens
        for e in q:
            nx.set_node_attributes(G, {e: 0}, score)
        # Remove F(e) so it's recomputed using the updated scores
        for e, d in G.nodes(data=True):
            del d["F(e)"]
    # Reset to the original scores
    nx.set_node_attributes(G, score_dict, score)
    # Optional: Repeat - iterative refinement
    if refine and m > 1:
        Q = iterative_refinement(G, Q, config=config)
    return Q


def iterative_refinement(
    G: nx.Graph, Q: list, score: str = "score", config=None
) -> list:
    """
    Returns a list of iteratively refined antigens
    :param G: Directed Graph containing epitopes
    :param Q: List containing antigens
    :returns: List containing iteratively refined antigens
    """
    m = len(Q)
    # Save original epitope score so it can be reset later
    score_dict = nx.get_node_attributes(G, score)
    while True:
        for n in range(0, m):
            prev_Q = Q
            # Remove sequence q from vaccine Q
            q = Q[n]
            Q.remove(q)
            # No credit for including e in existing antigens
            for q in Q:
                for e in q:
                    nx.set_node_attributes(G, {e: 0}, score)
            # Compute replacement for old sequence q
            q = find_optimal_path(G, config)
            # Add q to vaccine
            Q.insert(n, q)
            # Reset to the original scores
            nx.set_node_attributes(G, score_dict, score)
            # Remove F(e) so it's recomputed using the updated scores
            for e, d in G.nodes(data=True):
                del d["F(e)"]
            if Q == prev_Q:
                return Q
