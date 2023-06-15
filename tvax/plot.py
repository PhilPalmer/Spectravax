#!/usr/bin/env python

import igviz as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

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
    hspace: float = 0.05,
):
    """ """
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
    pmhc_aff_pivot = pmhc_aff_pivot.applymap(lambda x: 1 if x > affinity_cutoff else 0)
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
        cmap = cm.get_cmap(colors[i], 2)
        cmap.set_under(color="white")
        sns.heatmap(
            df.loc[df["loci"] == loci, :].pivot_table(
                values="binding", index=["allele"], columns=["position"]
            ),
            cmap=cmap,
            ax=axes[i],
            cbar=False,
            vmin=0.1,
        )
        axes[i].tick_params(axis="both", which="both", length=0)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        if i == n_loci - 1:
            axes[i].set_xlabel("Amino acid position", fontsize=16)
        if i == 1:
            axes[i].set_ylabel("MHC allele", fontsize=16)

    return fig, peptide_hits_df


#####################
# Population coverage
#####################


def plot_population_coverage(
    vaccine_design: list = None,
    n_targets: list = list(range(0, 11)),
    mhc_types: list = ["mhc1", "mhc2"],
    config: EpitopeGraphConfig = None,
):
    """
    Plot the population coverage of the vaccine designs.
    """
    peptides = vaccine_design
    pop_cov_dict = {"ancestry": [], "mhc_type": [], "n_target": [], "pop_cov": []}

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
                    overlap_haplotypes, anc_freq, n_target, peptides
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
    target_protein: str = "n",
    x: str = "w_mhc1",
    y: str = "w_mhc2",
    z: str = "pop_cov_mhc1_n_5",
    z_label: str = "Population coverage MHC Class I n ≥ 5 (%)",
    param_sweep_path: str = None,
    out_path: str = None,
) -> None:
    """
    Plot parameter sweep 3D surface plot
    """
    if param_sweep_path is None:
        param_sweep_path = f"data/param_sweep_betacov_{target_protein}.csv"
    if out_path is None:
        out_path = f"data/figures/param_sweep_surface_{target_protein}_{z}.png"

    # Load data
    df = pd.read_csv(param_sweep_path)
    covs = ["pop_cov_mhc1_n_5", "pop_cov_mhc2_n_5", "path_cov"]
    df["av_cov"] = df[covs].mean(axis=1)

    # Increase the resolution of the grid
    x_unique = np.linspace(df[x].min(), df[x].max(), 100)
    y_unique = np.linspace(df[y].min(), df[y].max(), 100)
    x_grid, y_grid = np.meshgrid(x_unique, y_unique)
    z_grid = griddata((df[x], df[y]), df[z], (x_grid, y_grid), method="linear")
    df_grid = pd.DataFrame({x: x_grid.ravel(), y: y_grid.ravel(), z: z_grid.ravel()})

    # Generate plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    # Trisurf plot
    # ax.plot_trisurf(
    #     df[x],
    #     df[y],
    #     df[z],
    #     cmap=cm.viridis,
    #     linewidth=0.2,
    #     antialiased=True,
    # )
    ax.plot_surface(
        x_grid, y_grid, z_grid, cmap="viridis", linewidth=0.5, vmin=0, vmax=100
    )
    ax.contour(
        x_grid,
        y_grid,
        z_grid,
        10,
        lw=3,
        cmap="viridis",
        linestyles="solid",
        offset=-1,
        vmin=0,
        vmax=100,
    )
    ax.set_zlim(0, 100)
    ax.set_xlabel("Population Coverage MHC Class I Weight ($w_{mhc1}$)")
    ax.set_ylabel("Population Coverage MHC Class II Weight ($w_{mhc2}$)")
    ax.set_zlabel(z_label)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")


#########################
# Compare vaccine designs
#########################


def plot_pop_cov_lineplot(
    pop_cov_df: pd.DataFrame, mhc_type: str = "mhc1", hue: str = "antigen"
) -> None:
    """
    Plot lineplot of the population coverage results for different vaccine designs.
    """
    mhc_class = "I" if mhc_type == "mhc1" else "II"
    y = f"Population Coverage for MHC Class {mhc_class}"
    pop_cov_df["mhc_type"] = (
        pop_cov_df["mhc_type"]
        .replace("mhc1", "Population Coverage for MHC Class I")
        .replace("mhc2", "Population Coverage for MHC Class II")
    )
    df = pop_cov_df[pop_cov_df["mhc_type"] == y]

    # Set the font size and style
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Set the figure size
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot the lineplot
    sns.lineplot(
        data=df,
        x="n_target",
        y="pop_cov",
        hue=hue,
        markers=True,
        dashes=False,
        palette="colorblind",
        ax=ax,
    )
    # Set the axis labels
    ax.set_xlabel("Minimum number of peptide-HLA hits cutoff")
    ax.set_ylabel(f"{y} (%)")
    # Set axis limits
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 10)
    # Display the legend outside of the plot (right middle)
    ax.legend(bbox_to_anchor=(1.0, 0.8), loc=2, borderaxespad=0.0)


def plot_path_cov_swarmplot(path_cov_df: pd.DataFrame) -> None:
    """
    Plot swarmplot of the pathogen coverage results for different vaccine designs.
    """
    # Set the font size and style
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Set the figure size
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

    # Set the x-axis label
    ax.set_xlabel("Target Antigen")
    ax.set_ylabel("Pathogen Coverage (%)")
    # Set axis limits
    ax.set_ylim(0, 100)
    # Rotate the x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    # Display the legend outside of the plot (right middle)
    ax.legend(bbox_to_anchor=(1.0, 0.8), loc=2, borderaxespad=0.0)


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
        hue_order=["Weak", "Strong", "Very Strong (<50nM)"],
        col_wrap=2,
        height=4,
        aspect=1.5,
    )
    g.map(sns.barplot, "ID", "hits").add_legend()

    # Add labels
    g.despine(left=True)
    g.set_axis_labels("", "Number of peptide-H-2 hits")
    g.legend.set_title("Binding threshold")
    g.set_titles("{col_name}")
    # Rotate the x-axis labels
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

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
