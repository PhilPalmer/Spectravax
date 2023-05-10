#!/usr/bin/env python

# Original author: Sneha Viswanathan (19th Oct 2021)
# Modified by: Phil Palmer (9th Feb 2023)

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#################
# Data processing
#################


def parse_msa(msa_pr_path):
    """
    Get the MSA from FASTA file and convert to dictionary
    :param msa_pr_path: path to the protein MSA FASTA file
    :return: MSA dictionary and MSA dataframe
    """
    seqs = SeqIO.parse(msa_pr_path, "fasta")
    msa_dict = {}
    msa_dict["Sequence_id"] = []
    for seq in seqs:
        msa_dict["Sequence_id"].append(seq.id)
        for i in range(len(seq.seq)):
            pos = f"pos{i+1}"
            if pos not in msa_dict:
                msa_dict[pos] = []
            msa_dict[pos].append(seq.seq[i])
    msa_df = pd.DataFrame(msa_dict)
    return msa_dict, msa_df


def convert_to_rank_matrix(msa_dict):
    """
    Convert the MSA dictionary to a ranked MSA dataframe
    :param msa_dict: MSA dictionary
    :return: ranked MSA dataframe
    """
    ranked_msa_dict = {}
    for (
        keys,
        values,
    ) in msa_dict.items():  # {'Sequence_id':['ASDW..',''],'Pos1':['E','K',.....],....}
        if keys == "Sequence_id":
            ranked_msa_dict[keys] = values
        else:
            d = {}
            for aa in values:
                d[aa] = d.get(aa, 0) + 1  # calculating the number of occurrence
            d_sorted = dict(sorted(d.items(), key=lambda x: x[1]))  # in ascending order
            c1 = 1
            for keys1 in d_sorted:
                if keys1 == "-":
                    d_sorted[keys1] = 0
                else:
                    d_sorted[keys1] = c1
                    c1 = c1 + 1
            rank = [d_sorted[aa1] for aa1 in values]
            ranked_msa_dict[keys] = rank
    ranked_msa_df = pd.DataFrame(ranked_msa_dict)
    ranked_msa_df.set_index("Sequence_id", inplace=True)
    return ranked_msa_df


def fit_pca(df, threshold=0.8):
    """
    Get PCA scores from a dataframe
    :param df: dataframe
    :param threshold: threshold used for the cumulative variance to determine the number of components to retain
    :return: list of sequence names, PCA scores and PCA object
    """
    pca = PCA()
    pca.fit(df)
    var = (
        pca.explained_variance_ratio_.cumsum()
    )  # calculating the contribution of each components
    c = 0
    for a in var:
        if a <= threshold:
            c = c + 1
        else:
            continue
    pca = PCA(n_components=c)
    pca.fit(df)
    return df.index.tolist(), pca.transform(df), pca


def kmeans_clustering(seq_names, pca_scores):
    """
    Compute the within cluster sum of squares (WCSS) using kmeans clustering
    :param seq_names: list of sequence names
    :param pca_scores: PCA scores
    :return: list of WCSS
    """
    wcss = []
    for i in range(1, min(len(seq_names), 31)):
        kmeans_pca = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans_pca.fit(pca_scores)
        wcss.append(kmeans_pca.inertia_)
    return wcss


def elbow_method(wcss):
    """
    Find the elbow point of the WCSS
    :param wcss: list of WCSS
    :return: elbow point
    """
    x1, y1 = 1, wcss[0]
    x2, y2 = len(wcss), wcss[-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i + 1
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)
    return distances.index(max(distances)) + 1


def get_clusters(msa_df, pca_scores, n_clusters):
    """
    Get the clusters from the MSA dataframe
    :param msa_df: MSA dataframe
    :param pca_scores: PCA scores
    :param n_clusters: number of clusters
    :return: Components dataframe
    """
    kmeans_pca = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans_pca.fit(pca_scores)
    pca_vector = pd.DataFrame(pca_scores)
    pca = pca_vector[pca_vector.columns[0:3]]
    # TODO: Add an option to specify the metadata file
    metadata_df = pd.DataFrame(
        {
            "Sequence_id": msa_df.Sequence_id.tolist(),
            "colour": "black",
            "type": "A",
            "size": 40,
        }
    )
    comp_df = msa_df.merge(
        metadata_df
    )  # used merge instead of concat to make sure the dataframe is matched at the sequence id level
    comp_df = pd.concat([comp_df.reset_index(), pca], axis=1)
    comp_df.columns.values[-3:] = ["PCA1", "PCA2", "PCA3"]
    comp_df["cluster"] = kmeans_pca.labels_
    comp_df["cluster"] = comp_df["cluster"] + 1
    comp_df["cluster_rank"] = comp_df["cluster"].map(
        {1: "first", 2: "second", 3: "third", 4: "fourth"}
    )
    return comp_df


def get_loadings(pca, msa_df, ranked_msa_df):
    """
    Get the loadings from the PCA
    :param pca: PCA object
    :param msa_df: MSA dataframe
    :param ranked_msa_df: ranked MSA dataframe
    :return: loadings dataframe
    """
    loadings_df = pd.DataFrame(pca.components_.T, index=ranked_msa_df.columns[:])
    dist = {}
    for index, rows in loadings_df.iterrows():
        d1 = (
            ((float(rows[0])) ** 2) + ((float(rows[1])) ** 2) + ((float(rows[2])) ** 2)
        ) ** 0.5
        dist[index] = d1
    m = max(dist.values())
    list1 = ["Sequence_id"]
    for pos1, dist1 in dist.items():
        if dist1 > (0.01 * m):
            list1.append(pos1)
        else:
            continue
    loadings_df = msa_df.loc[:, list1]
    return loadings_df


##########
# Plotting
##########


def plot_elbow_graph(wcss, figsize=(10, 6)):
    """
    Plot the elbow graph
    :param wcss: list of WCSS
    :param figsize: figure size
    :return: figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(range(1, len(wcss) + 1), wcss)
    plt.xlabel("Number of clusters")
    plt.ylabel("Within cluster sum of squares (WCSS)")
    return fig


def plot_pca(
    df,
    plot_type="3D",
    interactive=False,
    group_by="cluster",
    colours=list(sns.color_palette("colorblind").as_hex()),
    figsize=(10, 6),
):
    """
    Plot PCA
    :param df: dataframe
    :param type: string for '2D' or '3D' plot (default: '3D')
    :param group_by: column to group by
    :param colours: list of colours
    :param figsize: figure size
    :return: figure
    """
    # Define clusters
    df.loc[df["Sequence_id"].str.contains("design"), "cluster"] = "vaccine design"
    df["cluster"] = df["cluster"].astype(int, errors="ignore")
    # Define colours for each cluster
    df["colour"] = df[group_by].map(
        {
            cluster: colours[i % len(colours)]
            for i, cluster in enumerate(df[group_by].unique())
        }
    )
    df.loc[df["Sequence_id"].str.contains("design"), "colour"] = "black"
    if interactive:
        px.defaults.template = "plotly_white"
        df["cluster"] = df["cluster"].astype(str)
        color_discrete_map = (
            df[["cluster", "colour"]]
            .drop_duplicates()
            .set_index("cluster")
            .to_dict()["colour"]
        )
        fig = px.scatter_3d(
            df,
            x="PCA1",
            y="PCA2",
            z="PCA3",
            color="cluster",
            hover_name="Sequence_id",
            color_discrete_map=color_discrete_map,
            opacity=0.8,
        )
        fig.update_traces(marker=dict(line=dict(width=1, color="#808080")))
        return fig
    fig = plt.figure(figsize=figsize)
    if plot_type == "3D":
        ax = fig.add_subplot(111, projection="3d")
    for t1 in list(set(df[group_by])):
        group_df = df.loc[df[group_by] == t1]
        X = group_df["PCA1"]
        Y = group_df["PCA2"]
        Z = group_df["PCA3"]
        C = group_df["colour"]
        S = group_df["size"]
        if plot_type == "2D":
            plt.scatter(X, Y, c=C, edgecolor="black", s=S, label=t1)
        elif plot_type == "3D":
            ax.scatter(X, Y, Z, c=C, edgecolor="black", s=S, label=t1)
    plt.xlabel("PC 1", fontsize="12")
    plt.ylabel("PC 2", rotation="90", fontsize="12")
    plt.xticks(size="10")
    plt.yticks(size="10")
    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=12)
    plt.tight_layout()
    if plot_type == "3D":
        ax.set_xlim(min(df["PCA1"]), max(df["PCA1"]))
        ax.set_ylim(min(df["PCA2"]), max(df["PCA2"]))
        ax.set_zlim(min(df["PCA3"]), max(df["PCA3"]))
        ax.set_ylabel("PC 2", rotation="45", fontsize="12")
        ax.set_zlabel("PC 3", rotation="90", fontsize="12")
        ax.tick_params(labelsize="10")
    return fig


def plot_loadings(loadings, figsize=(10, 6)):
    """
    Plot the loadings
    :param loadings: loadings dataframe
    :param figsize: figure size
    :return: figure
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")
    X1 = loadings[0].values
    Y1 = loadings[1].values
    Z1 = loadings[2].values
    ax.scatter3D(X1, Y1, Z1)
    ax.set_xlabel("PC 1", fontsize="12")
    ax.set_ylabel("PC 2", rotation="45", fontsize="12")
    ax.set_zlabel("PC 3", rotation="90", fontsize="12")
    ax.tick_params(labelsize="10")
    plt.tight_layout()
    return fig


###############
# Main function
###############


def pca_protein_rank(msa_pr_path, n_clusters=None, plot=True):
    """
    Main function to rank, perform PCA and K-means clustering on a multiple sequence alignment of a protein
    :param msa_pr_path: path to the multiple sequence alignment
    :param n_clusters: number of clusters to use (default: None)
    :param plot: boolean to plot the results (default: True)
    :return: dictionary of clusters
    """
    # Data processing
    msa_dict, msa_df = parse_msa(msa_pr_path)
    ranked_msa_df = convert_to_rank_matrix(msa_dict)
    seq_names, pca_scores, pca = fit_pca(ranked_msa_df)
    wcss = kmeans_clustering(seq_names, pca_scores)

    if not n_clusters:
        print("Performing elbow method to determine the number of clusters to use...")
        n_clusters = elbow_method(wcss)
        print(f"Number of clusters to use: {n_clusters}")

    # TODO: Investigate NaN values in the 'cluster_rank' column
    comp_df = get_clusters(msa_df, pca_scores, n_clusters)
    loadings_df = get_loadings(pca, msa_df, ranked_msa_df)
    clusters_dict = (
        comp_df[["Sequence_id", "cluster"]]
        .set_index("Sequence_id")
        .to_dict()["cluster"]
    )

    if not plot:
        return clusters_dict, comp_df

    # Plotting
    elbow_plot = plot_elbow_graph(wcss)
    pca_2d_plot = plot_pca(comp_df, plot_type="2D")
    pca_3d_plot = plot_pca(comp_df, plot_type="3D")
    pca_interactive_plot = plot_pca(comp_df, interactive=True)
    loadings_plot = plot_loadings(loadings_df)

    return (
        clusters_dict,
        comp_df,
        elbow_plot,
        pca_2d_plot,
        pca_3d_plot,
        pca_interactive_plot,
        loadings_plot,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform PCA and clustering on a multiple sequence alignment"
    )
    parser.add_argument(
        "msa_pr_path", type=str, help="Path to the multiple sequence alignment"
    )
    parser.add_argument("--n_clusters", type=int, help="Number of clusters to use")
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    args = parser.parse_args()
    pca_protein_rank(args.msa_pr_path, args.n_clusters, args.plot)
