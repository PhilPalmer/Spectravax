#!/usr/bin/env python

import networkx as nx

from tvax.config import EpitopeGraphConfig
from tvax.graph import argmax, f, P


#######################################################
# Design: Wrapper function to design a vaccine cocktail
#######################################################


def design_vaccines(G: nx.Graph, config: EpitopeGraphConfig) -> list:
    """
    Design a vaccine cocktail from a graph of epitopes
    """
    # Find the optimal path(s) through the graph of epitopes
    Q = cocktail(G, config.m)
    # Convert the paths to AA strings
    vaccine_designs = [path_to_seq(path) for path in Q]
    return vaccine_designs


###############################################
# Find optimal path through a graph of epitopes
###############################################


def find_optimal_path(G: nx.Graph) -> list:
    """
    Returns the optimal path through a graph of epitopes
    :param G: Directed Graph containing epitopes
    :returns: List of epitope strings on the optimal path
    """
    # Forward loop - compute F(e)
    for e in G.nodes:
        F(G, e)
    # Backward loop - build the path that achieves the maximal score
    path = backward(G)
    return path


def F(G: nx.Graph, e: str) -> float:
    """
    Returns the maximum total score over all paths that end in e
    :param G: Directed Graph containing epitopes
    :param e: String for a given potential T-cell epitope (PTE)
    :returns: Float for the maximum total epitope score
    """
    # Use precomputed F(e) if it already exists for the epitope
    if "F(e)" not in G.nodes[e]:
        predecessors = P(G, e)
        if not predecessors:
            # If the set of predecessors P(e) is empty, then F(e) = f(e)
            Fe = f(G, e)
        else:
            # If the set of predecessors P(e) is not empty, then F(e) = f(e) + max(F(P(e)))
            Fe = f(G, e) + max([F(G, pe) for pe in predecessors])
        # Save F(e) to the graph for this epitope
        nx.set_node_attributes(G, {e: Fe}, "F(e)")
    return f(G, e, f="F(e)")


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


def cocktail(G: nx.Graph, m: int, refine: bool = True, score: str = "score") -> list:
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
    for n in range(0, m):
        # Compute and save next antigen sequence
        q = find_optimal_path(G)
        # Add q to vaccine
        Q.append(q)
        # No credit for including e in subsequent antigens
        for e in q:
            nx.set_node_attributes(G, {e: 0}, score)
        # Remove F(e) so it's recomputed using the updated scores
        for (e, d) in G.nodes(data=True):
            del d["F(e)"]
    # Reset to the original scores
    nx.set_node_attributes(G, score_dict, score)
    # Optional: Repeat - iterative refinement
    if refine:
        Q = iterative_refinement(G, Q)
    return Q


def iterative_refinement(G: nx.Graph, Q: list, score: str = "score") -> list:
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
            q = find_optimal_path(G)
            # Add q to vaccine
            Q.insert(n, q)
            # Reset to the original scores
            nx.set_node_attributes(G, score_dict, score)
            # Remove F(e) so it's recomputed using the updated scores
            for (e, d) in G.nodes(data=True):
                del d["F(e)"]
            if Q == prev_Q:
                return Q


########################################################
# Process output from the vaccine design into a sequence
########################################################


def path_to_seq(path: list) -> str:
    """
    Returns an AA string for a list of epitopes (path)
    """
    seq = [path[0]] + [e[-1] for e in path[1:]]
    return "".join(seq)
