#!/usr/bin/env python

import igviz as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tvax.config import EpitopeGraphConfig, Weights
from tvax.graph import load_fasta
from tvax.pca_protein_rank import pca_protein_rank, plot_pca
from tvax.seq import msa, path_to_seq
from scipy import stats


"""
Generate various plots to visualise the epitope graph and vaccine design(s).
"""


def plot_epitope_graph(
    G: nx.Graph,
    paths: list = None,
    node_size: int = 50,
    with_labels: bool = False,
    ylim: list = ([0, 1]),
    interactive: bool = False,
    colour_by: str = "score",
) -> plt.figure:
    """
    Plot the epitope graph
    :param G: Directed Graph containing epitopes
    :param paths: List of lists of epitope strings (deafult=None)
    :param node_size: Integer for size of nodes in non-interactive plot (default=150)
    :param with_labels: Boolean for if epitope labels should be displayed in the non-interactive plot (default=False)
    :param ylim: List for y-axis limits (default=[0,1])
    :param interactive: Boolean for if the plot should be interactive (default=False)
    :returns: None
    """

    if interactive:
        e = list(G.nodes)[0]
        attrs = list(G.nodes[e].keys())
        fig = ig.plot(
            G, color_method=colour_by, node_text=attrs
        )  # layout='spectral','spiral','spring'
        return fig.show()
    else:
        # Define vars
        node_color = list(nx.get_node_attributes(G, colour_by).values())
        pos = nx.get_node_attributes(G, "pos")
        if paths:
            for path in paths:
                path = ["BEGIN"] + path + ["END"]
                for i in range(0, len(path) - 1):
                    G.edges[path[i], path[i + 1]]["colour"] = "red"
        edge_colours = nx.get_edge_attributes(G, "colour")
        # Plot
        fig, ax = plt.subplots(1, figsize=(16, 8))
        nx.draw(
            G,
            node_color=node_color,
            pos=pos,
            node_size=node_size,
            with_labels=with_labels,
            edge_color=edge_colours.values(),
            width=2,
            font_color="white",
            ax=ax,
        )
        limits = plt.axis("on")
        max_pos = max([p[0] for p in pos.values()]) + 0.5
        ax.set_xlim([-0.5, max_pos])
        ax.set_ylim(ylim)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.tick_params(axis="both", which="major", labelsize=14)
        plt.ylabel("Epitope Score", fontsize=18)


def format_title(title: str) -> str:
    title = title.replace("_", " ").replace("mhc", "MHC")
    title = " ".join([w[0].upper() + w[1:] for w in title.split(" ")])
    return title


def plot_score(G: nx.Graph, score: str = "frequency") -> plt.figure:
    """
    Plot the distribution of a given score for the nodes in the graph.
    :param G: NetworkX graph
    :param score: String for the score to be plotted (default='frequency')
    :returns: Figure
    """
    scores = list(nx.get_node_attributes(G, score).values())
    # Plot
    fig, ax = plt.subplots(1, figsize=(16, 8))
    ax.hist(scores, bins=100)
    ax.set_xlabel(format_title(score), fontsize=18)
    ax.set_ylabel("Number of PTEs", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    return fig


def plot_scores(
    G: nx.Graph, weights: Weights, paths: list = None, percent: bool = False
) -> plt.figure:
    """
    Plot the scores of the nodes in the graph
    :param G: The graph of epitopes
    :param paths: The optimal paths through the graph, if specified the scores of the nodes in the path will be plotted
    :return: A figure of the scores of the nodes in the graph
    """

    def _get_mean_score(
        G: nx.Graph,
        weight_val: float,
        score: str = None,
    ):
        """
        Get the mean score of the nodes in the graph
        """
        return [
            np.mean(
                [
                    G.nodes[node][score] * weight_val * G.nodes[node]["clade_weight"]
                    for node in G.nodes
                    if G.nodes[node]["pos"][0] == p
                ]
            )
            for p in pos
        ]

    # Get scores
    score_dict = {}
    y_label = "Score"
    for score_name, score_weight in weights:
        if paths:
            score_dict[score_name] = [
                G.nodes[node][score_name] * score_weight * G.nodes[node]["clade_weight"]
                for node in paths[0]
            ]
            pos = list(range(0, len(paths[0])))
        else:
            score_dict[score_name] = _get_mean_score(G, score_weight, score_name)
            pos = list(set([G.nodes[node]["pos"][0] for node in G.nodes]))
    score_arrays = [score_dict[score] for score in score_dict.keys()]
    # For each position, for each score, express as a proportion of the total score
    if percent:
        y_label = "% Contribution to Total Score"
        for i in range(0, len(score_arrays[0])):
            total = sum([score_arrays[j][i] for j in range(0, len(score_arrays))])
            for j in range(0, len(score_arrays)):
                score_arrays[j][i] = score_arrays[j][i] / total * 100
        # Compute the average contribution of each score to the total score
        for i, score in enumerate(score_dict.keys()):
            score_dict[score] = np.mean(score_arrays[i])
    labels = [format_title(score) for score in score_dict.keys()]

    # Plot stacked area
    fig, ax = plt.subplots(1, figsize=(16, 8))
    ax.stackplot(pos, *score_arrays, labels=labels)
    ax.set_xlabel("Position", fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.legend(loc="upper left", fontsize=14)
    ax.set_xlim([0, max(pos)])
    return fig, score_dict


def plot_corr(
    G: nx.Graph, x: str = "frequency", y: str = "population_coverage"
) -> plt.figure:
    df = pd.DataFrame(
        {
            x: [G.nodes[node][x] for node in G.nodes],
            y: [G.nodes[node][y] for node in G.nodes],
        }
    )
    corr_plot = sns.jointplot(x=x, y=y, data=df, kind="reg", height=10)  #
    r, p = stats.pearsonr(df[x], df[y])
    corr_plot.ax_joint.annotate(
        f"$R_\\rho = {r:.2f}$",
        xy=(0.3, 0.3),
        xycoords=corr_plot.ax_joint.transAxes,
        fontsize=16,
    )
    corr_plot.set_axis_labels(format_title(x), format_title(y), fontsize=16)
    return corr_plot.fig


def plot_vaccine_design_pca(
    Q: list,
    config: EpitopeGraphConfig,
    n_clusters: int = None,
    plot_type: str = "2D",
    interactive: bool = False,
    plot: bool = False,
) -> plt.figure:
    """
    Plot the PCA of the vaccine design.
    :param Q: List of the optimal vaccine cocktail(s).
    :param n_clusters: Number of clusters to use for clustering the sequences.
    :param plot: Boolean indicating whether to generate all plots.
    :param plot_type: Type of plot to generate.
    :return: Plot of the PCA of the vaccine design.
    """
    # Define vars
    fasta_base = ".".join(str(config.fasta_path).split(".")[:-1])
    fasta_path = fasta_base + "_designs.fasta"
    msa_path = fasta_base + "_designs.msa"
    n_clusters = n_clusters if n_clusters else config.n_clusters
    # Load the sequences
    seqs_dict = load_fasta(config.fasta_path)
    # Add design to sequences dictionary
    seqs_dict["vaccine_design"] = path_to_seq(Q[0])
    # Write the sequences to a fasta file
    SeqIO.write(
        [
            SeqRecord(Seq(seqs_dict[seq_id]), id=seq_id, description="")
            for seq_id in seqs_dict
        ],
        fasta_path,
        "fasta",
    )
    # Perform MSA
    msa(fasta_path, msa_path, overwrite=True)
    # Perform PCA on the MSA
    clusters_dict, comp_df = pca_protein_rank(
        msa_path, n_clusters=n_clusters, plot=plot
    )
    # Plot the PCA results
    pca_plot = plot_pca(comp_df, plot_type=plot_type, interactive=interactive)
    return pca_plot, comp_df


#######################
# Parameter sweep plots
#######################


def plot_param_sweep_heatmap(
    df: pd.DataFrame, x: str = "n_cluster", y: str = "pop_cov_weight", z: str = "av_cov"
) -> go.Figure:
    """
    Plot a heatmap of the parameter sweep results.
    """
    fig = go.Figure(
        data=go.Heatmap(
            x=df[x],
            y=df[y],
            z=df[z],
            colorscale="RdBu",
            hovertemplate="Number of clusters: %{x}<br>Population coverage weight: %{y}<br>Population coverage: %{customdata[0]:.1f}%<br>Pathogen coverage: %{customdata[1]:.1f}%<br>Mean coverage: %{z:.1f}%",
            customdata=df[["pop_cov", "path_cov"]].values,
            colorbar=dict(title="Mean coverage (%)"),
        )
    )

    fig.update_layout(
        xaxis_title="Number of clusters", yaxis_title="Population coverage weight"
    )

    return fig


def plot_param_sweep_scatter(
    df: pd.DataFrame, x: str = "pop_cov", y: str = "path_cov", z: str = "av_cov"
) -> px.imshow:
    """
    Plot a scatter plot of the parameter sweep results.
    """

    pio.templates.default = "plotly_white"

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=z,
        trendline="ols",
        hover_data=["pop_cov_weight", "n_cluster"],
        labels={
            "pop_cov": "Population coverage (%)",
            "path_cov": "Pathogen coverage (%)",
            "av_cov": "Mean coverage (%)",
            "n_cluster": "Number of clusters",
            "pop_cov_weight": "Population coverage weight",
        },
        color_continuous_midpoint=50,
        # color_continuous_scale="RdBu",
    )

    fig.update_layout(
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
    )

    return fig
