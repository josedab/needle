#!/usr/bin/env python3
"""ann-benchmarks runner for Needle.

Usage:
    python run_benchmark.py --dataset sift-128-euclidean --k 10
    python run_benchmark.py --dataset glove-200-angular --k 10
"""

import argparse
import json
import subprocess
import sys
import time
import os

DATASETS = {
    "sift-128-euclidean": {"dims": 128, "metric": "euclidean", "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5"},
    "glove-200-angular": {"dims": 200, "metric": "angular", "url": "http://ann-benchmarks.com/glove-200-angular.hdf5"},
    "fashion-mnist-784-euclidean": {"dims": 784, "metric": "euclidean", "url": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"},
}

M_VALUES = [12, 16, 24]
EF_CONSTRUCTION = [100, 200, 500]
EF_SEARCH = [10, 20, 40, 80, 120, 200, 400, 800]


def run_benchmark(dataset, k=10):
    """Run benchmark for a dataset with parameter sweep."""
    info = DATASETS.get(dataset)
    if not info:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    print(f"Dataset: {dataset} (dims={info['dims']}, metric={info['metric']})")
    print(f"K={k}")
    print()

    results = []
    for m in M_VALUES:
        for ef_c in EF_CONSTRUCTION:
            # Build phase
            build_start = time.time()
            db_path = f"/tmp/needle_bench_{m}_{ef_c}.needle"

            subprocess.run(
                ["needle", "create", db_path],
                capture_output=True,
            )
            subprocess.run(
                ["needle", "create-collection", db_path,
                 "-n", "bench", "-d", str(info["dims"])],
                capture_output=True,
            )
            build_time = time.time() - build_start

            # Query phase (parameter sweep)
            for ef_s in EF_SEARCH:
                result = {
                    "algorithm": f"needle-hnsw",
                    "parameters": {"M": m, "ef_construction": ef_c, "ef_search": ef_s},
                    "dataset": dataset,
                    "k": k,
                    "build_time": build_time,
                    # Placeholder - real implementation would compute actual recall/QPS
                    "recall": 0.0,
                    "qps": 0.0,
                }
                results.append(result)
                print(f"  M={m} ef_c={ef_c} ef_s={ef_s}: build={build_time:.2f}s")

            # Cleanup
            try:
                os.remove(db_path)
            except FileNotFoundError:
                pass

    # Output results
    output_file = f"results_{dataset}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_file}")
    print(f"Total configurations tested: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Needle ann-benchmark runner")
    parser.add_argument("--dataset", default="sift-128-euclidean", help="Dataset name")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    args = parser.parse_args()
    run_benchmark(args.dataset, args.k)
