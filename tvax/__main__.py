#!/usr/bin/env python
"""
Spectravax: Design broad-spectrum T-cell-inducing vaccines.

Spectravax scores pathogen protein k-mers using two coverage metrics
— pathogen coverage and host coverage — then uses a graph-based algorithm
to select the optimal overlapping subset of k-mers to generate a contiguous
vaccine antigen sequence that maximises the total coverage score.
"""

import sys
import tempfile
from pathlib import Path

import click


# Standard file names expected inside a data directory
DATA_DIR_FILES = {
    "human_proteome_path": "human_proteome.fasta.gz",
    "mhc1_alleles_path": "mhc1_alleles.txt",
    "mhc2_alleles_path": "mhc2_alleles.txt",
    "hap_freq_mhc1_path": "hap_freq_mhc1.pkl",
    "hap_freq_mhc2_path": "hap_freq_mhc2.pkl",
}


def _seqs_to_temp_fasta(seqs):
    """Write amino acid sequences to a temporary FASTA file."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", prefix="spectravax_", delete=False
    )
    for i, seq in enumerate(seqs):
        tmp.write(f">seq_{i + 1}\n{seq}\n")
    tmp.flush()
    return Path(tmp.name)


def _resolve_data_dir(data_dir):
    """Resolve reference file paths from a data directory."""
    data_dir = Path(data_dir).expanduser()
    params = {}
    for config_key, filename in DATA_DIR_FILES.items():
        path = data_dir / filename
        if path.exists():
            params[config_key] = str(path)
    return params


@click.group()
@click.version_option(package_name="spectravax", prog_name="spectravax")
def cli():
    """Spectravax: Design broad-spectrum T-cell-inducing vaccines."""
    pass


@cli.command()
@click.option("--fasta", type=click.Path(exists=True),
              help="Protein FASTA file.")
@click.option("--seq", multiple=True,
              help="Amino acid sequence (repeat for multiple: --seq SEQ1 --seq SEQ2).")
@click.option("--fasta-nt", type=click.Path(exists=True),
              help="Nucleotide FASTA (triggers full preprocessing pipeline).")
@click.option("--output", "-o", type=click.Path(),
              help="Output FASTA path (default: stdout).")
@click.option("--results-dir", type=click.Path(),
              help="Directory for intermediate results (default: results_<prefix>).")
@click.option("--data-dir", type=click.Path(),
              help="Directory containing reference data files.")
@click.option("--k", multiple=True, type=int,
              help="K-mer length (repeat for multiple: --k 9 --k 15). Default: 9 15.")
@click.option("--m", type=int,
              help="Number of antigens in cocktail. Default: 1.")
@click.option("--n-target", type=int,
              help="Minimum peptide-HLA hits for host coverage. Default: 1.")
@click.option("--skip-host-coverage", is_flag=True, default=False,
              help="Skip MHC binding predictions (pathogen coverage only).")
@click.option("--fast/--robust", default=None,
              help="Scoring mode: --fast (sum scores) or --robust (population coverage).")
@click.option("--equalise-clades/--no-equalise-clades", default=None,
              help="Balance representation across pathogen clades.")
@click.option("--affinity-predictor", type=click.Choice(["netmhcpan", "mhcflurry"]),
              help="MHC binding predictor. Default: netmhcpan.")
@click.option("--w-cons", type=float,
              help="Pathogen coverage weight. Default: 1.0.")
@click.option("--w-mhc1", type=float,
              help="MHC-I host coverage weight. Default: 1.0.")
@click.option("--w-mhc2", type=float,
              help="MHC-II host coverage weight. Default: 1.0.")
@click.option("--w-clade", type=float,
              help="Clade balance weight. Default: 1.0.")
@click.option("--human-proteome", type=click.Path(exists=True),
              help="Human proteome FASTA for self-tolerance filter.")
@click.option("--mhc1-alleles", type=click.Path(exists=True),
              help="MHC-I alleles file.")
@click.option("--mhc2-alleles", type=click.Path(exists=True),
              help="MHC-II alleles file.")
@click.option("--hap-freq-mhc1", type=click.Path(exists=True),
              help="MHC-I haplotype frequencies (pickle).")
@click.option("--hap-freq-mhc2", type=click.Path(exists=True),
              help="MHC-II haplotype frequencies (pickle).")
def design(fasta, seq, fasta_nt, output, results_dir, data_dir, k, m, n_target,
           skip_host_coverage, fast, equalise_clades, affinity_predictor,
           w_cons, w_mhc1, w_mhc2, w_clade,
           human_proteome, mhc1_alleles, mhc2_alleles,
           hap_freq_mhc1, hap_freq_mhc2):
    """Design vaccine antigen(s) from pathogen sequences."""
    from tvax.config import EpitopeGraphConfig, Weights
    from tvax.design import design_vaccines
    from tvax.graph import build_epitope_graph
    from tvax.seq import path_to_seq

    # Require at least one input
    if not fasta and not seq and not fasta_nt:
        raise click.UsageError("Provide --fasta, --seq, or --fasta-nt.")

    # Build params — only include explicitly provided values
    # so EpitopeGraphConfig defaults are the source of truth
    params = {}

    # Input
    if fasta:
        params["fasta_path"] = fasta
    if seq:
        params["fasta_path"] = str(_seqs_to_temp_fasta(seq))
    if fasta_nt:
        params["fasta_nt_path"] = fasta_nt

    # Output
    if results_dir:
        params["results_dir"] = results_dir

    # Reference data: resolve from --data-dir first, then allow overrides
    if data_dir:
        params.update(_resolve_data_dir(data_dir))
    if human_proteome:
        params["human_proteome_path"] = human_proteome
    if mhc1_alleles:
        params["mhc1_alleles_path"] = mhc1_alleles
    if mhc2_alleles:
        params["mhc2_alleles_path"] = mhc2_alleles
    if hap_freq_mhc1:
        params["hap_freq_mhc1_path"] = hap_freq_mhc1
    if hap_freq_mhc2:
        params["hap_freq_mhc2_path"] = hap_freq_mhc2

    # Algorithm parameters
    if k:
        params["k"] = list(k)
    if m is not None:
        params["m"] = m
    if n_target is not None:
        params["n_target"] = n_target
    if fast is not None:
        params["robust"] = not fast
    if equalise_clades is not None:
        params["equalise_clades"] = equalise_clades
    if affinity_predictor:
        params["affinity_predictors"] = [affinity_predictor]

    # Weights — only set if explicitly provided
    weight_kwargs = {}
    if skip_host_coverage:
        weight_kwargs["population_coverage_mhc1"] = 0
        weight_kwargs["population_coverage_mhc2"] = 0
    if w_cons is not None:
        weight_kwargs["frequency"] = w_cons
    if w_mhc1 is not None:
        weight_kwargs["population_coverage_mhc1"] = w_mhc1
    if w_mhc2 is not None:
        weight_kwargs["population_coverage_mhc2"] = w_mhc2
    if w_clade is not None:
        weight_kwargs["clade"] = w_clade
    if weight_kwargs:
        params["weights"] = Weights(**weight_kwargs)

    config = EpitopeGraphConfig(**params)

    # Pipeline
    click.echo("Building epitope graph...", err=True)
    G = build_epitope_graph(config)
    click.echo(
        f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        err=True,
    )

    click.echo("Designing vaccine antigen(s)...", err=True)
    designs = design_vaccines(G, config)

    # Output vaccine sequences
    lines = []
    for i, path in enumerate(designs):
        lines.append(f">spectravax_antigen_{i + 1}")
        lines.append(path_to_seq(path))
    output_text = "\n".join(lines) + "\n"

    if output:
        Path(output).write_text(output_text)
        click.echo(f"Wrote {len(designs)} antigen(s) to {output}", err=True)
    else:
        click.echo(output_text, nl=False)


@cli.command("download-data")
@click.option("--data-dir", type=click.Path(), required=True,
              help="Directory to save reference files.")
def download_data(data_dir):
    """Download reference data files from Zenodo."""
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Replace with actual Zenodo DOI/URL once published
    click.echo(f"Data directory: {data_dir}")
    click.echo("Zenodo download not yet configured. Please place reference files manually:")
    for config_key, filename in DATA_DIR_FILES.items():
        path = data_dir / filename
        status = "found" if path.exists() else "MISSING"
        click.echo(f"  {filename} [{status}]")


if __name__ == "__main__":
    cli()
