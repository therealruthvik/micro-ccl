#!/usr/bin/env python3
"""Plot micro-ccl benchmark CSVs (bench_allreduce / bench_mpi_allreduce output).

Usage:
    python3 scripts/plot_results.py results1.csv results2.csv ... -o plots/

Reads one or more CSVs sharing the schema:
    algo,dtype,op,world_size,size_bytes,count,iters,min_us,median_us,p99_us,bandwidth_gbps

and writes two PNGs: latency vs. message size (log-log, one line per algo)
and bandwidth vs. message size (log-x), so the ring/recursive-doubling/
OpenMPI comparison and the crossover between them are visible directly,
not just as rows in a CSV.

Not part of the C++ build -- this is an offline plotting convenience the
project's "no exotic dependencies" constraint does not apply to (it never
runs on the RDMA data path, or even on the benchmarked machines
necessarily). Needs matplotlib: `pip install matplotlib`.
"""
import argparse
import csv
import sys
from collections import defaultdict


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["size_bytes"] = int(row["size_bytes"])
                row["median_us"] = float(row["median_us"])
                row["min_us"] = float(row["min_us"])
                row["p99_us"] = float(row["p99_us"])
                row["bandwidth_gbps"] = float(row["bandwidth_gbps"])
                rows.append(row)
    return rows


def group_by_algo(rows):
    by_algo = defaultdict(list)
    for r in rows:
        by_algo[r["algo"]].append(r)
    for algo, rs in by_algo.items():
        rs.sort(key=lambda r: r["size_bytes"])
    return by_algo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="+")
    parser.add_argument("-o", "--out-dir", default=".")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    rows = load_rows(args.csv_files)
    if not rows:
        sys.exit("no rows found in the given CSV file(s)")
    by_algo = group_by_algo(rows)

    # Latency vs. size -- this is the plot the algorithm crossover shows up
    # in: at small sizes, whichever algorithm has fewer round trips (fewer
    # sequential steps) should sit lower; at large sizes, whichever moves
    # less data per rank should sit lower.
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, rs in sorted(by_algo.items()):
        ax.plot([r["size_bytes"] for r in rs], [r["median_us"] for r in rs],
                marker="o", label=algo)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("median latency (us)")
    ax.set_title("Allreduce latency vs. message size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/latency_vs_size.png", dpi=150)

    # Bandwidth vs. size -- the mirror image of the latency plot, makes the
    # same crossover visible from the "effective throughput" side.
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, rs in sorted(by_algo.items()):
        ax.plot([r["size_bytes"] for r in rs],
                [r["bandwidth_gbps"] for r in rs], marker="o", label=algo)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title("Allreduce effective bandwidth vs. message size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/bandwidth_vs_size.png", dpi=150)

    print(f"wrote {args.out_dir}/latency_vs_size.png and "
          f"{args.out_dir}/bandwidth_vs_size.png")


if __name__ == "__main__":
    main()
