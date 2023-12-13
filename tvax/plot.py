#!/usr/bin/env python

import geopandas as gpd
import igviz as ig
import matplotlib.axes as axes
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
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
from tvax.seq import msa, path_to_seq, kmerise_simple
from typing import List, Optional
from scipy import stats
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde as kde
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss


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

    n_filtered_kmers_df = n_filtered_kmers_df.sort_values(by="antigen").reset_index()

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


#####################################
# Plot evaluation of binding criteria
#####################################


def compute_calibration_curve(y_true: list, y_prob: list, n_bins: int = 10):
    """
    Compute calibration curve from true labels and predicted probabilities.
    """

    # Bin the predicted probabilities
    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_limits[:-1] + bin_limits[1:]) / 2
    true_proportions = []
    pred_means = []

    for i in range(n_bins):
        left, right = bin_limits[i], bin_limits[i + 1]
        mask = (y_prob >= left) & (y_prob < right)

        # Compute the mean true and predicted values in each bin
        if mask.sum() > 0:
            true_proportions.append(y_true[mask].mean())
            pred_means.append(y_prob[mask].mean())

    return np.array(true_proportions), np.array(pred_means)


def plot_binding_criteria_eval(
    concat_df: str, out_path: str, mhc2_allele: str = "DRB1_0401"
):
    """Evaluate models and plot results."""

    def compute_precision_recall(y_true, y_prob, threshold=0.5):
        y_pred = (y_prob >= threshold).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return precision, recall

    scores = [
        "NetMHCpan4.1_BA_score",
        "NetMHCpan4.1_EL_score",
        "MHCFlurry2.0.6_BA_score",
        "MHCFlurry2.0.6_processing_score",
        "MHCFlurry2.0.6_presentation_score",
        "Ensemble_BA_score",
        "Ensemble_EL_score",
    ]

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind")

    mhc2_scores = [score for score in scores if "NetMHC" in score]
    mhc_dfs = {
        "Class I": (concat_df[concat_df["allele"] != mhc2_allele], scores),
        "Class II": (concat_df[concat_df["allele"] == mhc2_allele], mhc2_scores),
    }

    summary_data = []

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    for mhc_idx, (mhc_name, (mhc_df, mhc_scores)) in enumerate(mhc_dfs.items()):
        for idx, score in enumerate(mhc_scores):
            model_name = score.split("_")[0]
            if mhc_name == "Class II" and "NetMHCpan" in model_name:
                model_name = "NetMHCIIpan4.0"
            colour = palette[idx]
            score_name = score.split("_")[1]
            if score_name == "processing":
                score_name = "AP"
            if score_name == "presentation":
                score_name = "EL"
            if "BA" in score_name:
                if mhc_name == "Class I":
                    thresh = 50
                elif mhc_name == "Class II":
                    thresh = 500
                pr_threshold = 1 - np.log(thresh) / np.log(50000)
                pr_threshold_text = f"≤ {thresh}nM"
            else:
                pr_threshold = 0.5
                pr_threshold_text = "≥ 0.5"

            mhc_df[score] = mhc_df[score].replace([np.inf, -np.inf], np.nan)
            mhc_df.dropna(subset=[score], inplace=True)

            predicted_probs = mhc_df[score]
            true_labels = mhc_df["binder"]

            auc_roc = roc_auc_score(true_labels, predicted_probs)
            fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
            precision, recall = compute_precision_recall(
                true_labels, predicted_probs, threshold=pr_threshold
            )
            brier = brier_score_loss(true_labels, predicted_probs)
            logloss = log_loss(true_labels, predicted_probs)

            summary_data.append(
                {
                    "MHC": mhc_name,
                    "Model": model_name,
                    "Score Name": score_name,
                    "Binding Criterion": pr_threshold_text,
                    "AUROC": auc_roc,
                    "Precision": precision,
                    "Recall": recall,
                    "Brier Score": brier,
                    "Log Loss": logloss,
                }
            )

            ax[mhc_idx, 0].plot(
                fpr,
                tpr,
                color=colour,
                label=f"{model_name} {score_name} (AUC = {auc_roc:.2f})",
            )
            ax[mhc_idx, 0].set_title(f"ROC Curve for {mhc_name}")
            ax[mhc_idx, 0].set_xlabel("False Positive Rate")
            ax[mhc_idx, 0].set_ylabel("True Positive Rate")
            ax[mhc_idx, 0].legend()

            fraction_of_positives, mean_predicted_value = compute_calibration_curve(
                true_labels, predicted_probs
            )
            ax[mhc_idx, 1].plot(
                mean_predicted_value,
                fraction_of_positives,
                marker=".",
                color=colour,
                label=f"{model_name} {score_name}",
            )
            ax[mhc_idx, 1].set_title(f"Calibration Plot for {mhc_name}")
            ax[mhc_idx, 1].set_xlabel("Mean Predicted Probability")
            ax[mhc_idx, 1].set_ylabel("Fraction of Positives")

    for subplot_idx, subplot_ax in enumerate(ax.ravel()):
        subplot_ax.text(
            -0.05,
            1.1,
            string.ascii_uppercase[subplot_idx],
            transform=subplot_ax.transAxes,
            size=24,
            weight="bold",
        )
        subplot_ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(out_path, bbox_inches="tight")

    summary_df = pd.DataFrame(summary_data)

    return summary_df


def plot_calibration_curves(
    netmhc_df: pd.DataFrame,
    out_path: str = None,
    palette: list = sns.color_palette("colorblind"),
) -> None:
    """
    Plot calibration curves for MHC-I and MHC-II.
    """

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(18, 10))

    for mhc_idx, mhc_type in enumerate(["mhc1", "mhc2"]):
        mhc_name = "MHC-I" if mhc_type == "mhc1" else "MHC-II"

        # Subset data
        mhc_df = netmhc_df[netmhc_df["mhc_type"] == mhc_type]

        # Rename the DRB1 alleles for consistency
        mhc_df["allele"] = (
            mhc_df["allele"].str.replace("DRB1", "HLA-DRB1").str.replace("_", "-")
        )

        # Get sorted alleles
        sorted_alleles = sorted(mhc_df["allele"].unique())

        # Plot calibration for each allele
        for allele_idx, allele in enumerate(sorted_alleles):
            allele_df = mhc_df[mhc_df["allele"] == allele]
            predicted_probs = allele_df["EL-score"]
            true_labels = allele_df["binder"]

            frac_pos, mean_pred_value = compute_calibration_curve(
                true_labels, predicted_probs
            )

            ax[mhc_idx].plot(
                mean_pred_value,
                frac_pos,
                marker=".",
                label=allele,
                color=palette[allele_idx],
            )

        ax[mhc_idx].set_title(f"Calibration Plot for {mhc_name}", fontsize=18)
        ax[mhc_idx].set_xlabel("Mean Eluted Ligand (EL) Score", fontsize=16)
        ax[mhc_idx].set_ylabel("Fraction of Positives", fontsize=16)
        ax[mhc_idx].set_xlim([0, 1])
        ax[mhc_idx].set_ylim([0, 1])
        ax[mhc_idx].legend(title="Allele", bbox_to_anchor=(1.05, 1), loc="upper left")

        ax[mhc_idx].tick_params(axis="x", labelsize=14)
        ax[mhc_idx].tick_params(axis="y", labelsize=14)

    # Add subplot labels
    for subplot_idx, subplot_ax in enumerate(ax.ravel()):
        subplot_ax.text(
            -0.05,
            1.1,
            string.ascii_uppercase[subplot_idx],
            transform=subplot_ax.transAxes,
            size=28,
            weight="bold",
        )
        subplot_ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
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


def plot_pca_on_ax(df, ax):
    plot_pca(df, plot_type="2D", ax=ax)


def plot_boxplot_on_ax(df, ax, y="E(#DPs)_vax_cov_mhc1+2"):
    """
    Plot the coverage for each pathogen in the target sequences as a boxplot.
    """
    # Remove the vaccine design
    df = df[df["cluster"] != "vaccine design"]
    # Sort by the cluster
    df = df.sort_values(by="cluster")
    color_dict = dict(zip(df["cluster"].unique(), df["colour"].unique()))
    # Plot the boxes and the points
    sns.boxplot(
        x="cluster",
        y=y,
        data=df,
        palette=df["colour"].unique(),
        ax=ax,
    )
    sns.stripplot(
        x="cluster",
        y=y,
        data=df,
        jitter=True,
        dodge=True,
        size=6,
        linewidth=1,
        # alpha=0.5,
        hue="cluster",
        # color="black",
        palette=color_dict,
        edgecolor="white",
        ax=ax,
    )
    # Set the plot title and labels
    ax.set_xlabel("Cluster", fontsize=14)
    ax.set_ylabel("E(#DPs)", fontsize=14)
    ax.set_ylim(0)
    ax.legend().set_visible(False)


def plot_2d_kde_on_ax(
    x, y, z, citvax_i, ax, vmin=0, vmax=None, levels=15, fill=False, show_clb=False
):
    if vmax == None:
        vmax = z.max()
    citvax_x = x.values[citvax_i]
    citvax_y = y.values[citvax_i]
    x = x.drop(citvax_i)
    y = y.drop(citvax_i)
    z = z.drop(citvax_i)
    xy = np.vstack([x, y])
    z_kde = kde(xy, weights=z)
    x_grid, y_grid = np.meshgrid(
        np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    )
    xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z_grid_kde = z_kde(xy_grid).reshape(x_grid.shape)
    z_max = z.max()
    z_grid_kde = z_grid_kde * z_max / z_grid_kde.max()
    c = ax.scatter(x, y, c=z, cmap="viridis", vmin=vmin, vmax=vmax, edgecolors="white")
    if fill:
        ax.contourf(
            x_grid,
            y_grid,
            z_grid_kde,
            levels=levels,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            zorder=0,
        )
    else:
        ax.contour(
            x_grid,
            y_grid,
            z_grid_kde,
            levels=levels,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
    ax.scatter(
        citvax_x,
        citvax_y,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=100,
        marker="X",
        edgecolors="white",
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if show_clb:
        fig = ax.figure
        clb = fig.colorbar(c, ax=ax)
        clb.ax.set_title("E(#DPs)", fontsize=14)


def plot_3d_kde_on_ax(x, y, z, citvax_i, ax, vmin=0, vmax=None, levels=15):
    if vmax == None:
        vmax = z.max()
    citvax_x = x.values[citvax_i]
    citvax_y = y.values[citvax_i]
    citvax_z = z.values[citvax_i]
    x = x.drop(citvax_i)
    y = y.drop(citvax_i)
    z = z.drop(citvax_i)
    xy = np.vstack([x, y])
    z_kde = kde(xy, weights=z)
    x_grid, y_grid = np.meshgrid(
        np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    )
    xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z_grid_kde = z_kde(xy_grid).reshape(x_grid.shape)
    z_max = z.max()
    z_grid_kde = z_grid_kde * z_max / z_grid_kde.max()
    c = ax.scatter(x, y, z, c=z, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.scatter(
        citvax_x,
        citvax_y,
        citvax_z,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=100,
        marker="X",
        zorder=10,
    )
    ax.contour(
        x_grid, y_grid, z_grid_kde, levels=levels, cmap="viridis", vmin=vmin, vmax=vmax
    )
    ax.contour(
        x_grid, y_grid, z_grid_kde, zdir="y", offset=np.max(y_grid), colors="grey"
    )
    ax.contour(
        x_grid, y_grid, z_grid_kde, zdir="x", offset=np.min(x_grid), colors="grey"
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("E(#DPs)")
    ax.view_init(elev=40, azim=-70)


def plot_path_exp_dps(
    comp_df: pd.DataFrame,
    svg_path: str = None,
    levels=15,
):
    """
    Plot the expected number of displayed peptides for each target input sequence.
    """
    # Define variables such as x, y and z
    x = comp_df["PCA1"]
    y = comp_df["PCA2"]
    z_vax_col = "E(#DPs)_vax_cov_mhc1+2"
    z_wts_col = "E(#DPs)_wts_cov_mhc1+2"
    comp_df[z_vax_col] = (
        comp_df["E(#DPs)_vax_cov_mhc1"] + comp_df["E(#DPs)_vax_cov_mhc2"]
    )
    comp_df[z_wts_col] = (
        comp_df["E(#DPs)_wts_cov_mhc1"] + comp_df["E(#DPs)_wts_cov_mhc2"]
    )
    z_vax = comp_df[z_vax_col]
    z_wts = comp_df[z_wts_col]
    vmax = max(z_vax.max(), z_wts.max())
    vmin = 0
    citvax_i = comp_df[comp_df["Sequence_id"] == "vaccine_design"].index[0]

    # Define the figure
    sns.set(style="whitegrid", font_scale=1.3)
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 8)

    # Add axes and plots to the figure
    ax1 = fig.add_subplot(gs[0, 0:3])
    plot_pca_on_ax(comp_df, ax1)
    ax2 = fig.add_subplot(gs[0, 4:8])
    plot_boxplot_on_ax(comp_df, ax2)
    ax3 = fig.add_subplot(gs[1, 0:4])
    plot_2d_kde_on_ax(
        x,
        y,
        z_vax,
        citvax_i,
        ax3,
        fill=True,
        vmax=vmax,
        vmin=vmin,
        levels=levels,
        show_clb=True,
    )
    ax4 = fig.add_subplot(gs[1, 4:6])
    plot_2d_kde_on_ax(x, y, z_wts, citvax_i, ax4, vmax=vmax, vmin=vmin, levels=levels)
    ax5 = fig.add_subplot(gs[1, 6:8], projection="3d")
    plot_3d_kde_on_ax(x, y, z_wts, citvax_i, ax5, vmax=vmax, vmin=vmin, levels=levels)

    # Annotate the subplots with letters
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
        if ax != ax5:
            # For 2D axes
            ax.text(
                -0.05,
                1.1,
                string.ascii_uppercase[i],
                transform=ax.transAxes,
                size=24,
                weight="bold",
            )

    # Save the figure
    fig.subplots_adjust(hspace=0.3, wspace=0.4)
    if svg_path:
        plt.savefig(svg_path)


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


def plot_pmhc_heatmaps(seq, epitope_graph, config, alleles, out_path):
    # Set up the grid layout
    fig = plt.figure(figsize=(25, 40))
    gs = GridSpec(6, 1)

    axes_mhc1 = [fig.add_subplot(gs[i]) for i in range(3)]
    axes_mhc2 = [fig.add_subplot(gs[i + 3]) for i in range(3)]

    plot_mhc_heatmap(
        seq, epitope_graph, config, alleles, mhc_type="mhc1", fig=fig, axes=axes_mhc1
    )
    plot_mhc_heatmap(
        seq, epitope_graph, config, alleles, mhc_type="mhc2", fig=fig, axes=axes_mhc2
    )

    # Add letters to the subplots
    for i, ax in enumerate(axes_mhc1 + axes_mhc2):
        ax.text(
            -0.05,
            1.05,
            string.ascii_uppercase[i],
            transform=ax.transAxes,
            size=36,
            weight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path)


def plot_mhc_heatmap(
    seq: str,
    epitope_graph: nx.Graph,
    config: EpitopeGraphConfig,
    alleles: list = None,
    mhc_type: str = "mhc1",
    binding_criteria: str = "EL-score",
    hspace: float = 0.05,
    cbar_kws: dict = {"orientation": "vertical", "shrink": 0.8, "aspect": 20},
    fig=None,
    axes=None,
    xticklabels: int = 20,
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
    colors = ["Blues", "Oranges", "Greens"]

    # Create subplots for each loci, share the x-axis
    if fig is None and axes is None:
        fig, axes = plt.subplots(n_loci, 1, figsize=(25, 15), sharex=True)
        fig.subplots_adjust(hspace=hspace)

    # Plot a heatmap for each loci
    for i, loci in enumerate(df["loci"].unique()):
        cmap = (
            cm.get_cmap(colors[i], 2)
            if binding_criteria == "transformed_affinity"
            else cm.get_cmap(colors[i])
        )
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
                rasterized=True,
            )
            .collections[0]
            .colorbar.ax
        )

        # Update x-ticks
        max_pos = df.loc[df["loci"] == loci, "position"].max()
        new_xticks = list(range(0, max_pos + 1, xticklabels))
        axes[i].set_xticks(new_xticks)
        axes[i].set_xticklabels(new_xticks, rotation=0)

        # Set the title for the colorbar
        cbar_title = loci.replace("DRB1", "HLA-DRB1")
        cbar_ax.set_title(cbar_title, position=(0.5, 1.05), fontsize=20)
        cbar_ax.tick_params(labelsize=15)
        axes[i].tick_params(axis="both", which="both", length=0, labelsize=14)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        if i == n_loci - 1 and mhc_type == "mhc2":
            axes[i].set_xlabel("Amino acid position", fontsize=20)
        if i == 1:
            axes[i].set_ylabel("MHC allele", fontsize=20)

    return fig, peptide_hits_df


#####################
# Population coverage
#####################


def plot_population_coverage(
    trans_host_cov_df: pd.DataFrame,
    trans_cov_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    svg_path: str,
) -> None:
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(nrows=5, ncols=4, figure=fig)

    ax1 = np.array([fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4])])
    ax2 = np.array(
        [
            [
                fig.add_subplot(gs[2, 0]),
                fig.add_subplot(gs[2, 1]),
                fig.add_subplot(gs[2, 2]),
                fig.add_subplot(gs[2, 3]),
            ],
            [
                fig.add_subplot(gs[3, 0]),
                fig.add_subplot(gs[3, 1]),
                fig.add_subplot(gs[3, 2]),
                fig.add_subplot(gs[3, 3]),
            ],
        ]
    )
    ax3 = fig.add_subplot(gs[4, 0:2])
    ax4 = fig.add_subplot(gs[4, 2:4])

    plot_dual_population_coverage_histogram(trans_host_cov_df, trans_cov_df, axs=ax1)
    plot_population_coverage_histogram(trans_cov_df, axs=ax2)
    plot_population_coverage_barplot(cov_df, "mhc1", ax=ax3)
    plot_population_coverage_barplot(cov_df, "mhc2", ax=ax4)

    fig.subplots_adjust(hspace=0.5, wspace=0.2)

    all_axes = list(ax1) + list(ax2.ravel()) + [ax3, ax4]
    for subplot_idx, subplot_ax in enumerate(all_axes):
        subplot_ax.text(
            -0.05,
            1.05,
            string.ascii_uppercase[subplot_idx],
            transform=subplot_ax.transAxes,
            size=24,
            weight="bold",
        )

    fig.savefig(svg_path, bbox_inches="tight")


def plot_population_coverage_barplot(
    data: pd.DataFrame,
    mhc_type: str,
    ancestries_order: List[str] = ["Asian", "Black", "White", "Average"],
    max_n_target: int = 15,
    ax: axes.Axes = None,
) -> None:
    """
    Generate a barplot for the population coverage data.
    """
    # Filter data to only include the desired MHC type and only n_target <= max_n_target
    data = data[data["mhc_type"] == mhc_type]
    data = data[data["n_target"] <= max_n_target]

    sns.barplot(
        x="n_target",
        y="pop_cov",
        hue="ancestry",
        data=data,
        ax=ax,
        palette="Set2",
    )
    ax.set_xlabel("Minimum # of displayed peptides cutoff", fontsize=14)
    mhc_str = "MHC-I" if mhc_type == "mhc1" else "MHC-II"
    ax.set_title(mhc_str, fontsize=14)
    ax.set_ylabel(f"Coverage (%)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.legend(loc="upper left", fontsize=14)
    ax.set_ylim([0, 100])
    # Move legend to right side of plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc="lower left", fontsize=14)


def plot_single_ancestry_histogram(
    ax, data, ancestry, mhc_type, color, label_suffix="", dual_histogram=False
):
    """
    Plots a single histogram for a given MHC type and ancestry.
    """
    count = data["n_target"].values
    freq = data["pop_cov"].values
    if dual_histogram:
        if label_suffix:
            label = "Host coverage"
        else:
            label = "Coverage"
    else:
        label = None
    ax.bar(count, freq, width=1, color=color, label=label)
    mhc_type_str = "MHC-I" if mhc_type == "mhc1" else "MHC-II"
    ax.set_title(f"{mhc_type_str} {ancestry}", fontsize=14)
    ax.set_xlabel("# displayed peptides", fontsize=14)
    ax.set_ylabel(f"Coverage (%)", fontsize=14)
    if label_suffix == "" and dual_histogram:
        ax.yaxis.set_visible(False)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}".format(x)))
    exp = (count * freq / 100).sum()
    ax.axvline(
        x=exp, c=color, ls="--", lw=2, label="$\mathbb{E}(\# DPs)$" + f"={exp:.0f}"
    )

    if dual_histogram:
        if label_suffix == "":
            ax.legend(fontsize=12, loc="upper right")
        else:
            ax.legend(fontsize=12, bbox_to_anchor=(1, 0.7), loc="upper right")
    else:
        ax.legend(fontsize=12)

    return ax


def plot_dual_population_coverage_histogram(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    mhc_types: Optional[List[str]] = ["mhc1", "mhc2"],
    ancestry: str = "Average",
    palette: str = "Set2",
    axs: np.ndarray = None,
) -> None:
    """
    Plot two population coverage histograms based on the provided dataframes.
    """

    ancestry_colours = {
        "Asian": sns.color_palette(palette)[0],
        "Black": sns.color_palette(palette)[1],
        "White": sns.color_palette(palette)[2],
        "Average": sns.color_palette(palette)[3],
    }

    for i, mhc_type in enumerate(mhc_types):
        ancestry_data_1 = df1[
            (df1["ancestry"] == ancestry) & (df1["mhc_type"] == mhc_type)
        ]
        ancestry_data_2 = df2[
            (df2["ancestry"] == ancestry) & (df2["mhc_type"] == mhc_type)
        ]

        plot_single_ancestry_histogram(
            axs[i],
            ancestry_data_1,
            ancestry,
            mhc_type,
            sns.color_palette(palette)[6],
            label_suffix="(Host)",
            dual_histogram=True,
        )
        ax2 = axs[i].twinx()
        plot_single_ancestry_histogram(
            ax2,
            ancestry_data_2,
            ancestry,
            mhc_type,
            ancestry_colours[ancestry],
            dual_histogram=True,
        )


def plot_population_coverage_histogram(
    df: pd.DataFrame,
    mhc_types: Optional[List[str]] = ["mhc1", "mhc2"],
    ancestries_order: list = ["Asian", "Black", "White", "Average"],
    palette: str = "Set2",
    axs: np.ndarray = None,
) -> None:
    """
    Plot the population coverage histogram based on the provided dataframe.
    """

    ancestry_colours = {
        "Asian": sns.color_palette(palette)[0],
        "Black": sns.color_palette(palette)[1],
        "White": sns.color_palette(palette)[2],
        "Average": sns.color_palette(palette)[3],
    }

    for i, mhc_type in enumerate(mhc_types):
        for j, ancestry in enumerate(ancestries_order):
            ancestry_data = df[
                (df["ancestry"] == ancestry) & (df["mhc_type"] == mhc_type)
            ]
            plot_single_ancestry_histogram(
                axs[i, j], ancestry_data, ancestry, mhc_type, ancestry_colours[ancestry]
            )


########################################################################################
# Plot the expected number of displayed peptides E(#DPs) by country for a single antigen
########################################################################################


def filter_hla_groups(
    group, loci: dict = {"HLA-A", "HLA-B", "HLA-C", "HLA-DP", "HLA-DQ", "DRB1"}
):
    if loci.issubset(set(group["locus_name"].unique())):
        return group
    else:
        return group.head(0)


def plot_global_distribution(
    exp_dps_df: pd.DataFrame,
    ax: plt.Axes,
    mhc_type: str = "mhc1",
    vmax: int = 24,
    country_name_map: dict = None,
):
    if mhc_type == "mhc1":
        loci = {"DRB1", "HLA-DP", "HLA-DQ"}
    elif mhc_type == "mhc2":
        loci = {"HLA-A", "HLA-B", "HLA-C"}
    mhc_str = "MHC-I" if mhc_type == "mhc1" else "MHC-II"

    # Load the world map
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Summing the E(#DPs) for each country
    filtered_exp_dps_df = (
        exp_dps_df.groupby("country")
        .apply(filter_hla_groups, loci=loci)
        .reset_index(drop=True)
    )
    country_sum_exp_dps = (
        filtered_exp_dps_df.groupby("country")["E(#DPs)"].sum().reset_index()
    )

    # Renaming the countries to match the world map
    country_sum_exp_dps["country"] = country_sum_exp_dps["country"].replace(
        country_name_map
    )

    # Merge the world map with the aggregated E(#DPs) data
    world = world.merge(
        country_sum_exp_dps, how="left", left_on="name", right_on="country"
    )

    # Plotting
    ax.set_facecolor("w")
    world.boundary.plot(ax=ax, linewidth=1, edgecolor="grey")
    world.plot(
        column="E(#DPs)",
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        legend=True,
        legend_kwds={
            "label": "E(#DPs) by Country",
            "orientation": "horizontal",
            "shrink": 0.5,
            "aspect": 30,
            "pad": 0.01,
        },
    )
    ax.set_title(
        f"{mhc_str} Global Distribution of Expected Number of Displayed Peptides",
        fontsize=16,
        y=1.05,
    )
    ax.set_ylim([-60, 90])
    ax.set_axis_off()


def prepare_country_data(exp_dps_df: pd.DataFrame, world: pd.DataFrame):
    grouped_exp_dps_df = (
        exp_dps_df.groupby(["country", "locus_name"])["E(#DPs)"].sum().reset_index()
    )
    grouped_exp_dps_df["locus_name"] = grouped_exp_dps_df["locus_name"].replace(
        {"DRB1": "HLA-DRB1"}
    )
    pivot_df = grouped_exp_dps_df.pivot(
        index="country", columns="locus_name", values="E(#DPs)"
    ).fillna(0)
    pivot_df["Total"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.merge(
        world[["name", "pop_est"]], how="left", left_index=True, right_on="name"
    )
    top_countries_df = pivot_df.sort_values(by="pop_est", ascending=False).head(30)
    top_countries_df = pd.concat(
        [top_countries_df, pivot_df[pivot_df["name"] == "Global"]], axis=0
    )
    top_countries_df = (
        top_countries_df.drop(columns="pop_est")
        .rename(columns={"name": "country"})
        .set_index("country")
    )
    top_countries_df = top_countries_df.drop(columns="Total")
    return top_countries_df


def prepare_country_plot_data(top_countries_df: pd.DataFrame):
    long_df = top_countries_df.reset_index().melt(id_vars="country")
    long_df.columns = ["country", "locus", "E(#DPs)"]
    long_df["sort_key"] = (long_df["country"] == "Global").astype(int)
    long_df = long_df.sort_values(by=["sort_key", "country", "locus"]).drop(
        columns="sort_key"
    )
    return long_df


def create_stacked_bar_plot(long_df: pd.DataFrame, ax: plt.Axes):
    sns.set(style="whitegrid", palette="colorblind", font_scale=1.2)
    palette = sns.color_palette("colorblind", n_colors=long_df["locus"].nunique())
    color_dict = {
        locus: color for locus, color in zip(long_df["locus"].unique(), palette)
    }

    bottom = [0] * len(long_df["country"].unique())
    countries = long_df["country"].unique()
    for locus, color in sorted(color_dict.items(), reverse=True):
        data = long_df[long_df["locus"] == locus]
        ax.bar(countries, data["E(#DPs)"], bottom=bottom, label=locus, color=color)
        bottom = [b + d for b, d in zip(bottom, data["E(#DPs)"])]

    ax.set_title(
        "Expected Number of Displayed Peptides for The Top 30 Countries by Population Size"
    )
    ax.set_xlabel("Country")
    ax.set_ylabel("E(#DPs)")
    ax.set_xlim(-0.5, len(countries) - 0.5)
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries, rotation=45, ha="right")
    handles = [
        mpatches.Patch(color=color, label=locus) for locus, color in color_dict.items()
    ]
    ax.legend(handles=handles, title="Locus", bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray")

    plt.tight_layout()


def plot_exp_dps_by_country(
    exp_dps_df: pd.DataFrame,
    out_path: Path,
    country_name_mapping: dict,
):
    """
    Creates the figure for the global distribution of expected number of displayed peptides
    """
    # Set up the figure and GridSpec layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])

    # First Global Distribution Plot (A)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_global_distribution(
        exp_dps_df, ax1, mhc_type="mhc1", vmax=24, country_name_map=country_name_mapping
    )

    # Second Global Distribution Plot (B)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_global_distribution(
        exp_dps_df, ax2, mhc_type="mhc2", vmax=24, country_name_map=country_name_mapping
    )

    # Stacked Bar Plot (C)
    ax3 = fig.add_subplot(gs[1, :])  # Spanning across both columns
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    top_countries_df = prepare_country_data(exp_dps_df, world)
    long_df = prepare_country_plot_data(top_countries_df)
    create_stacked_bar_plot(long_df, ax3)

    # Add letter labels to each subplot
    for subplot_idx, subplot_ax in enumerate([ax1, ax2, ax3]):
        subplot_ax.text(
            -0.05,
            1.1,
            string.ascii_uppercase[subplot_idx],
            transform=subplot_ax.transAxes,
            size=24,
            weight="bold",
        )

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(
        fig
    )  # Close the figure to avoid displaying it in non-interactive environments


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


def plot_mhc_data(mhc_class, ax1, ax2, ax3, df, vmax=None):
    """
    Plot the number of epitopes for each experimental group.
    """
    subset_df = df[df["mhc_class"] == mhc_class]
    mhc_str = "MHC-I" if mhc_class == "mhc1" else "MHC-II"
    netmhc = "NetMHCpan" if mhc_class == "mhc1" else "NetMHCIIpan"

    # Plot the number of epitopes on ax1
    sns.barplot(
        data=subset_df,
        x="seq_id",
        y="is_mhc1_epitope",
        hue="test_seq_id",
        estimator=sum,
        ax=ax1,
        ci=None,
        palette="colorblind",
    )
    ax1.set_title(
        f"Number of {mhc_str} Epitopes Observed Experimentally (Source: IEDB)"
    )
    if vmax:
        ax1.set_ylim(bottom=0, top=vmax)
    ax1.set_xlabel("")
    ax1.set_ylabel("#Epitopes")
    ax1.legend().set_visible(False)

    # Plot expected number of displayed peptides on ax2 subplot
    sns.barplot(
        data=subset_df,
        x="seq_id",
        y="EL-score",
        hue="test_seq_id",
        estimator=sum,
        ax=ax2,
        ci=None,
        palette="colorblind",
    )
    ax2.set_title(
        f"Expected Number of {mhc_str} Displayed Peptides (Prediction via {netmhc})"
    )
    if vmax:
        ax2.set_ylim(bottom=0, top=vmax)
    ax2.set_xlabel("")
    ax2.set_ylabel("E(#DPs)")
    ax2.legend().set_visible(False)

    # Plot the main data on ax3 subplot
    sns.violinplot(
        data=subset_df,
        x="seq_id",
        y="EL-score",
        hue="test_seq_id",
        ax=ax3,
        palette="colorblind",
    )
    sns.stripplot(
        data=subset_df,
        x="seq_id",
        y="EL-score",
        hue="test_seq_id",
        dodge=True,
        jitter=True,
        ax=ax3,
        alpha=0.5,
        palette="colorblind",
    )
    ax3.set_title(
        f"Probability Distribution for {mhc_str} Peptide Presentation (Predicted using {netmhc})"
    )
    ax3.set_ylim(bottom=0, top=1)
    ax3.set_xlabel("Antigen")
    ax3.set_ylabel("Eluted Ligand (EL) Score")
    ax3.legend().set_visible(False)


def plot_experimental_predictions(results_df: pd.DataFrame, out_path: str):
    """
    Plot the experimental and predicted number of displayed peptides for each antigen.
    """
    fig = plt.figure(figsize=(21, 10))

    # Set up the grid layout
    gs = GridSpec(3, 2, height_ratios=[2, 2, 6], width_ratios=[1, 1])

    sns.set(style="whitegrid", palette="colorblind", font_scale=1.2)

    # Sort by seq_id and test_seq_id
    results_df = results_df.sort_values(by=["seq_id", "test_seq_id"])
    grouped = (
        results_df.groupby(["seq_id", "test_seq_id", "mhc_class"])["EL-score"]
        .sum()
        .reset_index()
    )
    vmax = grouped["EL-score"].max() + 1

    # Create subplots for MHC-I
    ax1_mhc1 = fig.add_subplot(gs[0, 0])
    ax2_mhc1 = fig.add_subplot(gs[1, 0])
    ax3_mhc1 = fig.add_subplot(gs[2, 0])

    # Create subplots for MHC-II
    ax1_mhc2 = fig.add_subplot(gs[0, 1])
    ax2_mhc2 = fig.add_subplot(gs[1, 1])
    ax3_mhc2 = fig.add_subplot(gs[2, 1])

    # Plot the data
    plot_mhc_data("mhc1", ax1_mhc1, ax2_mhc1, ax3_mhc1, results_df, vmax)
    plot_mhc_data("mhc2", ax1_mhc2, ax2_mhc2, ax3_mhc2, results_df, vmax)

    # Place legend outside the plot
    handles, labels = ax3_mhc1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        title="Peptide Pool",
        fontsize=14,
        title_fontsize=16,
    )

    # Label each subplot with a letter
    all_axes = [ax1_mhc1, ax1_mhc2, ax2_mhc1, ax2_mhc2, ax3_mhc1, ax3_mhc2]
    for subplot_idx, subplot_ax in enumerate(all_axes):
        subplot_ax.text(
            -0.05,
            1.05,
            string.ascii_uppercase[subplot_idx],
            transform=subplot_ax.transAxes,
            size=24,
            weight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path)


###################################################
# Plot the annotation + coverage scores by position
###################################################


def plot_annot_cov_by_pos(
    record: GraphicRecord,
    kmer_scores_df: pd.DataFrame,
    out_path: str = "data/figures/cov_by_pos.png",
    figsize=(12, 6),
    height_ratios=[3, 3],
) -> None:
    """
    Plot the annotation + coverage scores by position
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": height_ratios}
    )
    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")

    # Annotation
    record.plot(ax=ax1, with_ruler=False, strand_in_label_threshold=4)

    # Coverage
    conservation_label = "Conservation"
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
        conservation_label = "Conservation (Total)"
    sns.lineplot(
        data=kmer_scores_df,
        x="position",
        y="conservation_total",
        label=conservation_label,
        ax=ax2,
        # color="red",
    )
    sns.lineplot(
        data=kmer_scores_df,
        x="position",
        y="population_coverage_mhc1",
        label="Host Coverage (MHC-I)",
        ax=ax2,
    )
    sns.lineplot(
        data=kmer_scores_df,
        x="position",
        y="population_coverage_mhc2",
        label="Host Coverage (MHC-II)",
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
