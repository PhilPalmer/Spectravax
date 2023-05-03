import os
import pandas as pd

from tvax.config import EpitopeGraphConfig, Weights
from tvax.design import design_vaccines
from tvax.eval import compute_population_coverage, compute_av_pathogen_coverage
from tvax.graph import build_epitope_graph

"""
Run different analyses including a parameter sweep of potential vaccine designs.
"""


def run_parameter_sweep(
    n_clusters: list,
    pop_cov_weights: list,
    results_path: str,
    config: EpitopeGraphConfig,
) -> pd.DataFrame:
    """
    Run a parameter sweep over the number of clusters and the population coverage weights.
    """
    if not os.path.exists(results_path):
        param_sweep_dict = {
            "n_cluster": [],
            "pop_cov_weight": [],
            "pop_cov": [],
            "path_cov": [],
        }

        for n in n_clusters:
            for weight in pop_cov_weights:
                config.weights = Weights(frequency=1, population_coverage_mhc1=weight)
                config.n_clusters = n
                epitope_graph = build_epitope_graph(config)
                vaccine_designs = design_vaccines(epitope_graph, config)
                pop_cov = compute_population_coverage(vaccine_designs[0], 5, config)
                path_cov = compute_av_pathogen_coverage(vaccine_designs[0], config)
                param_sweep_dict["n_cluster"].append(n)
                param_sweep_dict["pop_cov_weight"].append(weight)
                param_sweep_dict["pop_cov"].append(pop_cov)
                param_sweep_dict["path_cov"].append(path_cov)

        # Process the data
        param_sweep_df = pd.DataFrame(param_sweep_dict)
        param_sweep_df["pop_cov"] = param_sweep_df["pop_cov"] * 100
        param_sweep_df["path_cov"] = param_sweep_df["path_cov"] * 100
        param_sweep_df["av_cov"] = (
            param_sweep_df["pop_cov"] + param_sweep_df["path_cov"]
        ) / 2
        param_sweep_df.to_csv(results_path, index=False)

    else:
        # TODO: Check that the results are the same as the ones in the file
        param_sweep_df = pd.read_csv(results_path)

    return param_sweep_df
