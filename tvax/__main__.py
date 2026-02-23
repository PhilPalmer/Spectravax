#!/usr/bin/env python
"""
Spectravax: Design broad-spectrum T-cell-inducing vaccines.

Spectravax scores pathogen protein k-mers using two coverage metrics
— pathogen coverage and host coverage — then uses a graph-based algorithm
to select the optimal overlapping subset of k-mers to generate a contiguous
vaccine antigen sequence that maximises the total coverage score.

Usage:
    python -m tvax --fasta INPUT.fasta [options]
    python -m tvax --seq "MSDNGPQNQRNAP" "MSDNGPQNQRSAP" [options]
"""

import argparse
import sys
import tempfile
from pathlib import Path

from tvax.config import EpitopeGraphConfig, Weights
from tvax.design import design_vaccines
from tvax.graph import build_epitope_graph
from tvax.seq import path_to_seq


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="spectravax",
        description="Design broad-spectrum T-cell-inducing vaccine antigens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Design from a FASTA file
  python -m tvax --fasta sequences.fasta --k 9 15 --output vaccine.fasta

  # Quick test with amino acid sequences
  python -m tvax --seq "MSDNGPQNQRNAP" "MSDNGPQNQRSAP" --k 9

Citation:
  Palmer et al. (2025). bioRxiv, 2025.04.14.648815
  https://doi.org/10.1101/2025.04.14.648815
""",
    )

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--fasta", type=Path, help="Path to input protein FASTA file"
    )
    input_group.add_argument(
        "--seq", nargs="+", metavar="SEQ",
        help="Amino acid sequences (space-separated strings)",
    )

    # Optional: nucleotide FASTA for preprocessing
    parser.add_argument(
        "--fasta-nt", type=Path, default=None,
        help="Path to nucleotide FASTA (triggers full preprocessing pipeline)",
    )

    # Output
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Directory for intermediate results (default: results_<prefix>)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output FASTA path (default: stdout)",
    )

    # Algorithm parameters
    parser.add_argument(
        "--k", type=int, nargs="+", default=[9, 15],
        help="K-mer lengths (default: 9 15)",
    )
    parser.add_argument(
        "--m", type=int, default=1,
        help="Number of antigens in cocktail (default: 1)",
    )
    parser.add_argument(
        "--n-target", type=int, default=1,
        help="Minimum peptide-HLA hits for host coverage (default: 1)",
    )
    parser.add_argument(
        "--robust", action="store_true", default=True,
        help="Use robust optimisation (default)",
    )
    parser.add_argument(
        "--fast", dest="robust", action="store_false",
        help="Use fast optimisation (sum scores instead of host coverage)",
    )
    parser.add_argument(
        "--equalise-clades", action="store_true", default=True,
        help="Balance representation across pathogen clades (default)",
    )
    parser.add_argument(
        "--no-equalise-clades", dest="equalise_clades", action="store_false",
    )
    parser.add_argument(
        "--affinity-predictor", choices=["netmhcpan", "mhcflurry"],
        default="netmhcpan", help="MHC binding affinity predictor (default: netmhcpan)",
    )

    # Scoring weights
    weights_group = parser.add_argument_group("scoring weights")
    weights_group.add_argument("--w-cons", type=float, default=1.0, help="Pathogen coverage weight (default: 1.0)")
    weights_group.add_argument("--w-mhc1", type=float, default=1.0, help="MHC-I host coverage weight (default: 1.0)")
    weights_group.add_argument("--w-mhc2", type=float, default=1.0, help="MHC-II host coverage weight (default: 1.0)")
    weights_group.add_argument("--w-clade", type=float, default=1.0, help="Clade balance weight (default: 1.0)")

    # MHC binding data paths
    data_group = parser.add_argument_group("data paths")
    data_group.add_argument("--human-proteome", type=Path, default=None, help="Human proteome FASTA for self-tolerance filter")
    data_group.add_argument("--mhc1-alleles", type=Path, default=None, help="MHC-I alleles file")
    data_group.add_argument("--mhc2-alleles", type=Path, default=None, help="MHC-II alleles file")
    data_group.add_argument("--hap-freq-mhc1", type=Path, default=None, help="MHC-I haplotype frequencies (pickle)")
    data_group.add_argument("--hap-freq-mhc2", type=Path, default=None, help="MHC-II haplotype frequencies (pickle)")

    return parser.parse_args(argv)


def seqs_to_temp_fasta(seqs):
    """Write amino acid sequences to a temporary FASTA file."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", prefix="spectravax_", delete=False
    )
    for i, seq in enumerate(seqs):
        tmp.write(f">seq_{i + 1}\n{seq}\n")
    tmp.flush()
    return Path(tmp.name)


def main(argv=None):
    args = parse_args(argv)

    # Build FASTA from --seq if provided
    fasta_path = args.fasta
    if args.seq:
        fasta_path = seqs_to_temp_fasta(args.seq)

    # Construct config
    params = {
        "fasta_path": str(fasta_path),
        "k": args.k,
        "m": args.m,
        "n_target": args.n_target,
        "robust": args.robust,
        "equalise_clades": args.equalise_clades,
        "affinity_predictors": [args.affinity_predictor],
        "weights": Weights(
            frequency=args.w_cons,
            population_coverage_mhc1=args.w_mhc1,
            population_coverage_mhc2=args.w_mhc2,
            clade=args.w_clade,
        ),
    }
    if args.results_dir:
        params["results_dir"] = str(args.results_dir)
    if args.fasta_nt:
        params["fasta_nt_path"] = str(args.fasta_nt)
    if args.human_proteome:
        params["human_proteome_path"] = str(args.human_proteome)
    if args.mhc1_alleles:
        params["mhc1_alleles_path"] = str(args.mhc1_alleles)
    if args.mhc2_alleles:
        params["mhc2_alleles_path"] = str(args.mhc2_alleles)
    if args.hap_freq_mhc1:
        params["hap_freq_mhc1_path"] = str(args.hap_freq_mhc1)
    if args.hap_freq_mhc2:
        params["hap_freq_mhc2_path"] = str(args.hap_freq_mhc2)

    config = EpitopeGraphConfig(**params)

    # Core workflow
    print("Building epitope graph...", file=sys.stderr)
    G = build_epitope_graph(config)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)

    print("Designing vaccine antigen(s)...", file=sys.stderr)
    designs = design_vaccines(G, config)

    # Output vaccine sequences
    output_lines = []
    for i, path in enumerate(designs):
        seq = path_to_seq(path)
        output_lines.append(f">spectravax_antigen_{i + 1}")
        output_lines.append(seq)

    output_text = "\n".join(output_lines) + "\n"
    if args.output:
        args.output.write_text(output_text)
        print(f"Wrote {len(designs)} antigen(s) to {args.output}", file=sys.stderr)
    else:
        print(output_text, end="")


if __name__ == "__main__":
    main()
