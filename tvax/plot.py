#!/usr/bin/env python

import igviz as ig
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import string

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from dna_features_viewer import GraphicFeature, GraphicRecord
from matplotlib import cm
from tvax.config import EpitopeGraphConfig, Weights
from tvax.graph import load_fasta
from tvax.pca_protein_rank import pca_protein_rank, plot_pca
from tvax.score import load_haplotypes, load_overlap, optivax_robust
from tvax.seq import msa, path_to_seq, kmerise_simple
from scipy import stats
from scipy.interpolate import griddata


"""
Generate various plots to visualise the epitope graph and vaccine design(s).
"""


def plot_kmer_graph(
    G: nx.Graph,
    paths: list = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    score: str = "score",
    score_min: float = 0,
    score_max: float = 1,
    colour_palette: str = "viridis",
    jitter_amount: float = 0.02,
    path_edge_colour: str = "red",
    edge_colour: str = "#BFBFBF",
    node_size: int = 15,
    with_labels: bool = False,
    xlim: list = None,
    ylim: list = ([-0.03, 1.05]),
):
    """
    Plot the k-mer graph
    """
    # Compute necessary variables
    scores = nx.get_node_attributes(G, score)
    score_values = scores.values()
    cmap = sns.color_palette(colour_palette, as_cmap=True)
    norm = mcolors.Normalize(vmin=score_min, vmax=score_max)
    node_color = [cmap(norm(score)) for score in score_values]

    # Position of the nodes with jitter
    pos = nx.get_node_attributes(G, "pos")
    pos = {node: (x, scores[node]) for node, (x, score) in pos.items()}
    pos_jittered = {
        node: (x, y + np.random.uniform(-jitter_amount, jitter_amount))
        for node, (x, y) in pos.items()
    }

    # Edge colours
    if paths is not None:
        path_edges = []
        paths[0] = ["BEGIN"] + paths[0] + ["END"]
        for i in range(len(paths[0]) - 1):
            path_edges.append((paths[0][i], paths[0][i + 1]))

        # Get the edge colours
        edge_colours_non_path = [
            (u, v) for u, v, d in G.edges(data=True) if (u, v) not in path_edges
        ]
        edge_colours_path = [
            (u, v) for u, v, d in G.edges(data=True) if (u, v) in path_edges
        ]

    # Compute the max_pos value
    max_pos = max([p[0] for p in pos.values()]) + 0.5

    # Plot the graph
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, figsize=(16, 8))

    # Draw the non-path edges
    nx.draw_networkx_edges(
        G,
        pos=pos_jittered,
        edgelist=edge_colours_non_path,
        edge_color=edge_colour,
        node_size=0,  # Hide the nodes for now
        width=0.5,
        ax=ax,
    )

    # Draw the nodes
    nx.draw_networkx_nodes(
        G, node_color=node_color, pos=pos_jittered, node_size=node_size, ax=ax
    )

    # Draw the path edges
    nx.draw_networkx_edges(
        G,
        pos=pos_jittered,
        edgelist=edge_colours_path,
        edge_color=path_edge_colour,
        node_size=0,  # Hide the nodes for now
        width=2,
        ax=ax,
    )

    # Label the path nodes
    if with_labels:
        path_nodes = sum(paths, [])
        for i, node in enumerate(path_nodes):
            # offset = 0.03 if i % 2 == 0 else -0.03
            plt.text(pos[node][0], pos[node][1], node, fontsize=10, ha="center")

    # Set plot properties
    if xlim == None:
        ax.set_xlim([-0.1, max_pos])
    else:
        ax.set_xlim(xlim)
    # ax.set_ylim(ylim)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel("Approximate K-mer Position", fontsize=18)
    ax.set_ylabel("K-mer Score", fontsize=18)

    # Only show whole numbers on the x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add a colorbar to illustrate the score
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="vertical")
    return fig


def plot_kmer_graphs(
    G: nx.Graph,
    paths: list,
    out_path: str,
    fig_size: tuple = (16, 16),
    xlim1: tuple = None,
    ylim1: tuple = ([-0.03, 1.05]),
    xlim2: tuple = ([180, 200]),
    ylim2: tuple = ([-0.03, 0.3]),
) -> None:
    """
    Plot two k-mer graphs with different x and y limits
    """
    # Create the figure
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size)

    # Plot the graphs
    fig1 = plot_kmer_graph(G, paths, fig=fig, ax=ax1, xlim=xlim1, ylim=ylim1)
    fig2 = plot_kmer_graph(G, paths, fig=fig, ax=ax2, xlim=xlim2, ylim=ylim2)

    # Save fig
    plt.savefig(out_path, bbox_inches="tight")


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
        # ax.set_ylim(ylim)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.tick_params(axis="both", which="major", labelsize=14)
        plt.ylabel("K-mer Score", fontsize=18)
        plt.xlabel("K-mer Position", fontsize=18)


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


def plot_scores_distribution(
    scores_dict: dict, out_path: str = "data/figures/scores_distribution.svg"
) -> None:
    df = pd.DataFrame(scores_dict)
    score_names = {
        "f_cons": "Conservation score ($f_{cons}$)",
        "f_mhc1": "MHC-I host coverage score ($f_{mhc1}$)",
        "f_mhc2": "MHC-II host coverage score ($f_{mhc2}$)",
        "f": "Total weighted score ($f$)",
        "f_clade_adjusted": "Clade adjusted score ($f_{clade\_adjusted}$)",
        "f_scaled": "Scaled score ($f_{scaled}$)",
    }

    # Set seaborn theme and style
    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")

    # Create subplots
    fig, axes = plt.subplots(
        nrows=len(df.index), ncols=1, figsize=(10, 15), sharex=False, sharey=False
    )

    # Define a function to generate x values for the KDE plot
    def _generate_x(data, num_points=1000):
        return np.linspace(min(data), max(data), num_points)

    # Iterate over each score (row in the DataFrame)
    for row, score in enumerate(df.index):
        # Iterate over each antigen (column in the DataFrame)
        for antigen in df.columns:
            # Get data
            data = df.loc[score, antigen]
            # Compute KDE
            kde = stats.gaussian_kde(data)
            # Generate x values
            x_values = _generate_x(data)
            # Compute density values
            density = kde(x_values)
            # Compute the width of the interval between x-values
            dx = x_values[1] - x_values[0]
            # Convert density to count
            count = density * dx * len(data)
            # Plot count
            axes[row].plot(x_values, count, label=antigen)
            axes[row].fill_between(x_values, count, alpha=0.5)

        # Set title, labels and x-limits
        # axes[row].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[row].set_xlabel(score_names.get(score, score))
        axes[row].set_ylabel("Number of k-mers")
        axes[row].legend()
        axes[row].set_xlim([0, 1])

    # Save the plot
    plt.tight_layout()
    plt.savefig(out_path)


def plot_kmer_filtering(
    n_filtered_kmers_df: pd.DataFrame, out_path: str = None
) -> None:
    """
    Plot stacked bar chart showing the number of k-mers that pass each filter
    """
    # Create a figure
    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_context("paper")

    # Plot the bars
    sns.barplot(
        x="antigen",
        y="n_host_kmers",
        data=n_filtered_kmers_df,
        color=palette[3],
        label="Host",
        ax=ax,
    )
    sns.barplot(
        x="antigen",
        y="n_rare_kmers",
        data=n_filtered_kmers_df,
        color=palette[1],
        label="Rare",
        ax=ax,
    )
    sns.barplot(
        x="antigen",
        y="n_passed_kmers",
        data=n_filtered_kmers_df,
        color=palette[2],
        label="Passed",
        ax=ax,
    )

    for index, row in n_filtered_kmers_df.iterrows():
        y_pos = (
            row["n_passed_kmers"] + 0.02 * row["n_passed_kmers"]
            if row["n_rare_kmers"] == 0
            else row["n_rare_kmers"] + 0.02 * row["n_rare_kmers"]
        )
        ax.text(
            index,
            y_pos,
            str(row["n_host_kmers"]),
            color=palette[3],
            ha="center",
            fontsize=12,
        )

    # Set plot properties
    ax.set_ylabel("Number of K-mers", fontsize=18)
    ax.set_xlabel("Antigen", fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.legend(fontsize=14)

    # Save the figure
    plt.savefig(out_path, bbox_inches="tight")


def plot_corr(
    G: nx.Graph, x: str = "frequency", y: str = "population_coverage_mhc1"
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
    base_path = f"{config.results_dir}/MSA/{config.prefix}_designs"
    fasta_path = f"{base_path}.fasta"
    msa_path = f"{base_path}.msa"
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


###################
# Pathogen coverage
###################


def plot_path_cov(path_cov_df: pd.DataFrame):
    """
    Plot the pathogen coverage for each target input sequence
    """
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    # Plot the boxes and the points
    sns.boxplot(
        x="clade",
        y="pathogen_coverage",
        data=path_cov_df,
        palette=path_cov_df["colour"].unique(),
        ax=ax,
    )
    sns.stripplot(
        x="clade",
        y="pathogen_coverage",
        data=path_cov_df,
        jitter=True,
        dodge=True,
        size=4,
        alpha=0.5,
        color="black",
        ax=ax,
    )
    # Set the plot title and labels
    ax.set_xlabel("Cluster", fontsize=14)
    ax.set_ylabel("Pathogen Coverage (%)", fontsize=14)
    # ax.set_ylim(0, 18)


##########################
# Plot MHC binding heatmap
##########################


def plot_mhc_heatmap(
    seq: str,
    epitope_graph: nx.Graph,
    config: EpitopeGraphConfig,
    alleles: list = None,
    mhc_type: str = "mhc1",
    binding_criteria: str = "EL-score",
    hspace: float = 0.05,
    cbar_kws: dict = {"orientation": "vertical", "shrink": 0.8, "aspect": 20},
):
    """
    Plot the MHC binding heatmap for a given sequence.
    """
    if mhc_type == "mhc1":
        affinity_cutoff = config.affinity_cutoff_mhc1
        raw_affinity_path = config.raw_affinity_netmhc_path
        k = min(config.k)
        hla_loci = ["HLA-A", "HLA-B", "HLA-C"]
    else:
        affinity_cutoff = config.affinity_cutoff_mhc2
        raw_affinity_path = config.raw_affinity_netmhcii_path
        k = max(config.k)
        hla_loci = ["DRB1", "HLA-DP", "HLA-DQ"]

    pmhc_aff_pivot = pd.read_pickle(raw_affinity_path)
    if binding_criteria == "transformed_affinity":
        pmhc_aff_pivot = pmhc_aff_pivot.applymap(
            lambda x: 1 if x > affinity_cutoff else 0
        )
    path = kmerise_simple(seq, [k])
    pos = [i + 1 for i in range(len(path))]
    try:
        df = pmhc_aff_pivot.loc[path, :].T
    except KeyError:
        # T-reg epitopes are not in the binding data - set them to 0
        df = pd.DataFrame(
            np.zeros((len(pmhc_aff_pivot.columns), len(path))), columns=path
        )
        kmers = [kmer for kmer in path if kmer in pmhc_aff_pivot.index]
        df.index = pmhc_aff_pivot.columns
        df.loc[:, kmers] = pmhc_aff_pivot.loc[kmers, :].T
    df.columns = pos
    # Reshape dataframe to long format
    df = df.reset_index().melt(
        id_vars=["loci", "allele"], var_name="position", value_name="binding"
    )
    # Filter to only include the specified alleles
    if alleles:
        df = df.loc[df["allele"].isin(alleles), :]
    # Filter to only keep the specified HLA loci
    df = df.loc[df["loci"].isin(hla_loci), :]
    # Rename the loci DRB1 -> HLA-DRB1
    df["allele"] = df["allele"].apply(
        lambda x: x.replace("DRB1", "HLA-DRB1").replace("_", "-")
    )

    # Generate dataframe of number of peptide-HLA hits per allele
    if not alleles:
        alleles = df["allele"].unique().tolist()
    peptide_hits_df = (
        df.loc[df["binding"] == 1, :]
        if binding_criteria == "transformed_affinity"
        else df
    )
    peptide_hits_df = df.loc[df["binding"] == 1, :]
    peptide_hits_df = (
        peptide_hits_df.groupby(["allele"])
        .size()
        .reset_index(name="n_peptide_hla_hits")
    )
    # Add missing alleles
    missing_alleles = [a for a in alleles if a not in peptide_hits_df["allele"].values]
    missing_df = pd.DataFrame(
        {"allele": missing_alleles, "n_peptide_hla_hits": [0] * len(missing_alleles)}
    )
    # Concat and reset index
    peptide_hits_df = pd.concat([peptide_hits_df, missing_df], axis=0, sort=False)
    # Sort by allele
    peptide_hits_df = peptide_hits_df.sort_values(by="allele").reset_index(drop=True)

    # Update the positions to extend to the length of the k-mer
    k_minus_1 = k - 1
    df["position"] = pd.to_numeric(df["position"])
    start_df = df.loc[df["position"] <= k_minus_1, :].copy()
    start_df["binding"] = 0
    df["position"] += k_minus_1
    df = pd.concat([start_df, df], sort=False)

    # Update the binding to extend to the length of the k-mer
    df["binding"] = pd.to_numeric(df["binding"])
    previous_binding_cols = [f"binding_previous{i+1}" for i in range(k_minus_1)]
    for i, col in enumerate(previous_binding_cols):
        df[col] = df.groupby(["loci", "allele"])["binding"].transform(
            lambda x: x.shift(-(i + 1)).fillna(0)
        )
    df["binding"] = df[["binding"] + previous_binding_cols].max(axis=1)
    df = df.drop(columns=previous_binding_cols)

    # Plot heatmap of all loci on the same plot with same color scale
    # fig, ax = plt.subplots(figsize=(25, 15))
    # sns.heatmap(df.pivot_table(values='binding', index=['loci', 'allele'], columns=['position']), cmap="Blues", ax=ax, cbar=False)
    # ax.set_xlabel('Amino acid position', fontsize=14)
    # ax.set_ylabel('MHC allele', fontsize=14)

    # Define vars
    n_loci = len(df["loci"].unique())
    colors = ["Reds", "Greens", "Blues"]

    # Create subplots for each loci, share the x-axis
    fig, axes = plt.subplots(n_loci, 1, figsize=(25, 15), sharex=True)
    # Reduce the space between the subplots on the y-axis
    fig.subplots_adjust(hspace=hspace)

    # Plot a heatmap for each loci
    for i, loci in enumerate(df["loci"].unique()):
        cmap = (
            cm.get_cmap(colors[i], 2)
            if binding_criteria == "transformed_affinity"
            else cm.get_cmap(colors[i])
        cmap.set_under(color="white")

        # Plot the heatmap, capture the returned colorbar
        cbar_ax = (
            sns.heatmap(
                df.loc[df["loci"] == loci, :].pivot_table(
                    values="binding", index=["allele"], columns=["position"]
                ),
                cmap=cmap,
                ax=axes[i],
                cbar=True,
                cbar_kws=cbar_kws,
                vmin=0,
                vmax=1,
            )
            .collections[0]
            .colorbar.ax
        )

        # Set the title for the colorbar
        cbar_title = loci.replace("DRB1", "HLA-DRB1")
        cbar_ax.set_title(cbar_title, position=(0.5, 1.05))
        axes[i].tick_params(axis="both", which="both", length=0, labelsize=14)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        if i == n_loci - 1:
            axes[i].set_xlabel("Amino acid position", fontsize=20)
        if i == 1:
            axes[i].set_ylabel("MHC allele", fontsize=20)

    return fig, peptide_hits_df


#####################
# Population coverage
#####################


def plot_population_coverage(
    vaccine_design: list = None,
    n_targets: list = list(range(0, 11)),
    mhc_types: list = ["mhc1", "mhc2"],
    config: EpitopeGraphConfig = None,
    G: nx.Graph = None,
):
    """
    Plot the population coverage of the vaccine designs.
    """
    peptides = vaccine_design
    pop_cov_dict = {"ancestry": [], "mhc_type": [], "n_target": [], "pop_cov": []}
    kmers_dict = None if G is None else dict(G.nodes(data=True))

    # Preprocessing
    for mhc_type in mhc_types:
        hap_freq_path = (
            config.hap_freq_mhc1_path
            if mhc_type == "mhc1"
            else config.hap_freq_mhc2_path
        )
        hap_freq, average_frequency = load_haplotypes(hap_freq_path)
        if "Asians" in hap_freq.index:
            hap_freq.index = hap_freq.index.str.replace("Asians", "Asian")
        ancestries = hap_freq.index.tolist() + ["Average"]
        for anc in ancestries:
            if anc == "Average":
                anc_freq = average_frequency
            else:
                anc_freq = hap_freq.loc[anc].copy()
            overlap_haplotypes = load_overlap(peptides, anc_freq, config, mhc_type)
            for n_target in n_targets:
                pop_cov = optivax_robust(
                    overlap_haplotypes, anc_freq, n_target, peptides, kmers_dict
                )
                pop_cov_dict["ancestry"].append(anc)
                pop_cov_dict["mhc_type"].append(mhc_type)
                pop_cov_dict["n_target"].append(n_target)
                pop_cov_dict["pop_cov"].append(pop_cov)

    pop_cov_df = pd.DataFrame(pop_cov_dict)
    pop_cov_df["pop_cov"] = pop_cov_df["pop_cov"] * 100
    pop_cov_df["n_target"] = "n ≥ " + pop_cov_df["n_target"].astype(str)

    # Sort by ancestry but put average last
    ancestries = sorted(ancestries)
    ancestries.remove("Average")
    ancestries.append("Average")
    pop_cov_df["ancestry"] = pd.Categorical(
        pop_cov_df["ancestry"], categories=ancestries, ordered=True
    )

    # Plot
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    pop_cov_plots = []

    for mhc_type in mhc_types:
        fig, ax = plt.subplots(1, figsize=(14, 6))
        sns.barplot(
            x="n_target",
            y="pop_cov",
            hue="ancestry",
            data=pop_cov_df[pop_cov_df["mhc_type"] == mhc_type],
            ax=ax,
            palette="Set2",
        )
        ax.set_xlabel("Minimum number of peptide-HLA hits cutoff", fontsize=18)
        mhc_class = "I" if mhc_type == "mhc1" else "II"
        ax.set_ylabel(f"Population Coverage for MHC Class {mhc_class} (%)", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.legend(loc="upper left", fontsize=14)
        # ax.set_xlim([1, 8])
        ax.set_ylim([0, 100])
        # Move legend to right side of plot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc="lower left", fontsize=14)
        pop_cov_plots.append(fig)
    return pop_cov_plots, pop_cov_df


#######################
# Parameter sweep plots
#######################


def plot_parameter_sweep_surface(
    antigens=["n", "nsp12", "h3", "n1"],
    x: str = "w_mhc1",
    y: str = "w_mhc2",
    zs=["av_cov", "path_cov", "pop_cov_mhc1_n_5", "pop_cov_mhc2_n_5"],
    z_labels=[
        "Average Coverage (%)",
        "Pathogen Coverage (%)",
        "Population Coverage MHC-I n ≥ 5 (%)",
        "Population Coverage MHC-II n ≥ 5 (%)",
    ],
    out_path="data/figures/param_sweep_surface.png",
) -> None:
    """
    Plot parameter sweep 3D surface plot
    """
    # Plot a grid of 3D surface plots
    fig, ax = plt.subplots(
        len(zs), len(antigens), figsize=(20, 20), subplot_kw={"projection": "3d"}
    )
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # Generate the surface plot for each antigen
    for i, antigen in enumerate(antigens):
        for j, z in enumerate(zs):
            param_sweep_path = f"data/param_sweep_{antigen}.csv"

            # Load data
            df = pd.read_csv(param_sweep_path)
            covs = ["pop_cov_mhc1_n_5", "pop_cov_mhc2_n_5", "path_cov"]
            df["av_cov"] = df[covs].mean(axis=1)

            # Increase the resolution of the grid
            x_unique = np.linspace(df[x].min(), df[x].max(), 100)
            y_unique = np.linspace(df[y].min(), df[y].max(), 100)
            x_grid, y_grid = np.meshgrid(x_unique, y_unique)
            z_grid = griddata((df[x], df[y]), df[z], (x_grid, y_grid), method="linear")
            df_grid = pd.DataFrame(
                {x: x_grid.ravel(), y: y_grid.ravel(), z: z_grid.ravel()}
            )

            # Plot the chosen point, surface, and contour
            ax[j, i].plot(
                [20],
                [10],
                [z_grid[10, 20]],
                markerfacecolor="black",
                markeredgecolor="black",
                marker="o",
                alpha=1,
                markersize=7.5,
                zorder=20,
            )
            ax[j, i].plot_surface(
                x_grid,
                y_grid,
                z_grid,
                cmap=cm.viridis,
                linewidth=0,
                vmin=0,
                vmax=100,
                antialiased=True,
                zorder=1,
            )
            ax[j, i].contour(
                x_grid,
                y_grid,
                z_grid,
                10,
                cmap="viridis",
                linestyles="solid",
                offset=-1,
                vmin=0,
                vmax=100,
                zorder=0,
            )
            # Set the axis limits, labels, and title
            ax[j, i].set_zlim(0, 100)
            ax[j, i].set_xlabel("$w_{mhc1}$")
            ax[j, i].set_ylabel("$w_{mhc2}$")
            if i == len(antigens) - 1:
                ax[j, i].set_zlabel(z_labels[j])
            if j == 0:
                ax[j, i].set_title(f"{antigen.upper()}")

    # Save the plot
    plt.savefig(out_path, dpi=300, bbox_inches="tight")


#########################
# Compare vaccine designs
#########################


def plot_antigens_comparison(
    path_cov_df: pd.DataFrame,
    pop_cov_df: pd.DataFrame,
    out_path: str,
    fig_size: tuple = (16, 24),
) -> None:
    """
    Compare the pathogen and population coverage for different antigens.
    """
    # Create the figure
    sns.set_style("whitegrid")
    gridspec = dict(hspace=0.3, height_ratios=[1, 0, 1, 1])
    fig, (ax1, ax4, ax2, ax3) = plt.subplots(
        nrows=4, ncols=1, figsize=fig_size, gridspec_kw=gridspec
    )
    ax4.set_visible(False)

    # Plot the graphs
    fig1 = plot_path_cov_swarmplot(path_cov_df, fig=fig, ax=ax1)
    fig2 = plot_pop_cov_lineplot(pop_cov_df, mhc_type="mhc1", fig=fig, ax=ax2)
    fig3 = plot_pop_cov_lineplot(pop_cov_df, mhc_type="mhc2", fig=fig, ax=ax3)

    # Annotate the subplots with letters
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.text(
            -0.05,
            1.1,
            string.ascii_uppercase[i],
            transform=ax.transAxes,
            size=24,
            weight="bold",
        )

    # Save fig
    plt.savefig(out_path, bbox_inches="tight")


def plot_pop_cov_lineplot(
    pop_cov_df: pd.DataFrame,
    mhc_type: str = "mhc1",
    x: str = "n_target",
    y: str = "pop_cov",
    hue: str = "antigen",
    style: str = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Plot lineplot of the population coverage results for different vaccine designs.
    """
    mhc_class = "I" if mhc_type == "mhc1" else "II"
    new_y = f"Coverage for MHC-{mhc_class}"
    pop_cov_df["mhc_type"] = (
        pop_cov_df["mhc_type"]
        .replace("mhc1", "Coverage for MHC-I")
        .replace("mhc2", "Coverage for MHC-II")
    )
    df = pop_cov_df[pop_cov_df["mhc_type"] == new_y]
    df = df.rename(columns={y: new_y})
    y = new_y

    # Set the font size and style
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Set the figure size
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # Plot the lineplot
    sns.lineplot(
        data=df,
        x=x,
        y=new_y,
        hue=hue,
        style=style,
        markers=True,
        dashes=False,
        palette="colorblind",
        ax=ax,
    )
    # Set the axis labels
    ax.set_xlabel("Minimum number of peptide-HLA hits cutoff", fontsize=18)
    ax.set_ylabel(f"{y} (%)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Set axis limits
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 10)
    # Display the legend outside of the plot (right middle)
    ax.legend(bbox_to_anchor=(1.0, 0.8), loc=2, borderaxespad=0.0)
    return fig


def plot_path_cov_swarmplot(
    path_cov_df: pd.DataFrame, fig: plt.Figure = None, ax: plt.Axes = None
) -> plt.Figure:
    """
    Plot swarmplot of the pathogen coverage results for different vaccine designs.
    """
    # Set the font size and style
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Set the figure size
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # Plot the swarmplot with data points
    sns.violinplot(
        data=path_cov_df,
        x="antigen",
        y="pathogen_coverage",
        hue="antigen",
        palette="colorblind",
        ax=ax,
    )
    sns.swarmplot(
        data=path_cov_df,
        x="antigen",
        y="pathogen_coverage",
        hue="antigen",
        palette="colorblind",
        ax=ax,
        dodge=True,
        size=3,
        alpha=0.5,
        legend=False,
    )

    # Set the axis labels
    ax.set_xlabel("Target Antigen", fontsize=18)
    ax.set_ylabel("Pathogen Coverage (%)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    # Set axis limits
    ax.set_ylim(0, 100)
    # Rotate the x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    # Display the legend outside of the plot (right middle)
    ax.legend(bbox_to_anchor=(1.0, 0.8), loc=2, borderaxespad=0.0)
    return fig


###########################################################################
# Plot the predicted the number of hits in mice for each experimental group
###########################################################################


def plot_predicted_hits_barplot(
    binders_df: pd.DataFrame, out_path: str = "data/figures/predicted_mice_mhc_hits.png"
) -> None:
    """
    Plot the predicted number of peptide-MHC hits for different sequences
    """
    # Define the style
    sns.set(style="whitegrid")
    sns.set_palette("colorblind")

    # Generate the plot
    g = sns.FacetGrid(
        binders_df,
        col="mhc_class",
        hue="binding_threshold",
        hue_order=[
            "Weak",
            "Strong",
            "Very Strong (<50nM)",
            "Experimentally Observed (IEDB)",
        ],
        col_wrap=2,
        height=4,
        aspect=1.5,
    )
    g.map(sns.barplot, "ID", "hits", order=sorted(binders_df.ID.unique())).add_legend()

    # Add labels
    g.fig.subplots_adjust(wspace=0.1)
    g.despine(left=True)
    mhc = "H-2" if "mice" in out_path else "HLA"
    g.set_axis_labels("", f"Number of peptide-{mhc} hits")
    g.legend.set_title("Binding threshold")
    g.set_titles("{col_name}")
    # Rotate the x-axis labels
    for ax in g.axes.flat:
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

    # Save the plot
    g.savefig(out_path, dpi=300)


###################################################
# Plot the annotation + coverage scores by position
###################################################


def plot_annot_cov_by_pos(
    record: GraphicRecord,
    kmer_scores_df: pd.DataFrame,
    out_path: str = "data/figures/cov_by_pos.png",
) -> None:
    """
    Plot the annotation + coverage scores by position
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 3]}
    )
    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")

    # Annotation
    record.plot(ax=ax1, with_ruler=False, strand_in_label_threshold=4)

    # Coverage
    sns.lineplot(
        data=kmer_scores_df,
        x="position",
        y="conservation_total",
        label="Conservation (Total)",
        ax=ax2,
        # color="red",
    )
    if (
        "conservation_sarbeco" in kmer_scores_df.columns
        and "conservation_merbeco" in kmer_scores_df.columns
    ):
        sns.lineplot(
            data=kmer_scores_df,
            x="position",
            y="conservation_sarbeco",
            label="Conservation (Sarbeco)",
            ax=ax2,
            color=sns.xkcd_rgb["lightish blue"],
            linestyle="dashed",
        )
        sns.lineplot(
            data=kmer_scores_df,
            x="position",
            y="conservation_merbeco",
            label="Conservation (Merbeco)",
            ax=ax2,
            color=sns.xkcd_rgb["darkish blue"],
            linestyle="dashed",
        )
    sns.lineplot(
        data=kmer_scores_df,
        x="position",
        y="population_coverage_mhc1",
        label="Population Coverage (MHC-I)",
        ax=ax2,
    )
    sns.lineplot(
        data=kmer_scores_df,
        x="position",
        y="population_coverage_mhc2",
        label="Population Coverage (MHC-II)",
        ax=ax2,
    )
    # Set axis limits and labels
    plt.xlim(0, kmer_scores_df["position"].max())
    plt.ylim(0, 100)
    plt.xlabel("Amino Acid Position in CITVax Vaccine Design")
    plt.ylabel("K-mer Score (%)")
    plt.legend(bbox_to_anchor=(1.01, 0.8), loc=2, borderaxespad=0.0)
    plt.tight_layout()

    # Save the plot
    plt.savefig(out_path, dpi=300)
