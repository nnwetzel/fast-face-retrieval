#!/usr/bin/env python3
"""
Experiment runner for face retrieval system.

Runs experiments with different vocabulary tree configurations and
generates performance reports.

Usage:
    python experiments.py --quick          # Quick test with fewer configs
    python experiments.py --full           # Full experiment suite
    python experiments.py --custom --k-values 5 10 20 --num-queries 200
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from feature_extraction import (
    load_deeplake_dataset,
    DeeplakeFaceDataset,
    get_transform,
    build_model,
    extract,
)
from ptr_vocabulary_tree import split_tree, traverse_tree

import torch
from torch.utils.data import DataLoader


class ExperimentRunner:
    """Runs experiments with different configurations."""

    def __init__(self, args):
        self.args = args
        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.embeddings = None
        self.labels = None
        self.paths = None

    def load_embeddings(self):
        """Load or extract embeddings."""
        embeddings_path = Path(self.args.embeddings_path)

        if embeddings_path.exists():
            print(f"[INFO] Loading embeddings from {embeddings_path}")
            data = np.load(embeddings_path, allow_pickle=True)
            self.embeddings = data["embeddings"]
            self.labels = data["labels"]
            self.paths = data["paths"]
        else:
            print(f"[INFO] Extracting embeddings from {self.args.deeplake_uri}")
            device = torch.device(self.args.device)
            tfm = get_transform()

            items = load_deeplake_dataset(self.args.deeplake_uri)
            ds = DeeplakeFaceDataset(items, tfm)
            dl = DataLoader(
                ds,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=(device.type == "cuda"),
            )

            model = build_model(device)
            self.embeddings, self.labels, self.paths = extract(model, dl, device)

            # Save embeddings
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                embeddings_path,
                embeddings=self.embeddings,
                labels=np.array(self.labels),
                paths=np.array(self.paths),
            )

        print(f"[INFO] Loaded {len(self.embeddings)} embeddings")

    def build_vec_to_idx(self):
        """Build mapping from embedding bytes to index."""
        vec_to_idx = {}
        for idx, emb in enumerate(self.embeddings):
            vec_to_idx[emb.tobytes()] = idx
        return vec_to_idx

    def run_single_experiment(self, k, num_queries, seed=42):
        """Run a single experiment with given k value."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: k={k}, queries={num_queries}")
        print(f"{'='*60}")

        # Build tree
        print(f"[INFO] Building tree with k={k}...")
        tree_start = time.time()
        tree = split_tree(self.embeddings, k)
        tree_build_time = time.time() - tree_start
        print(f"[INFO] Tree built in {tree_build_time:.2f}s")

        # Build mapping
        vec_to_idx = self.build_vec_to_idx()

        # Select query indices
        np.random.seed(seed)
        n = len(self.embeddings)
        query_indices = np.random.choice(
            n, size=min(num_queries, n), replace=False
        )

        # Run queries
        correct = 0
        total_query_time = 0.0
        query_times = []

        for q_idx in query_indices:
            query_vec = self.embeddings[q_idx]
            query_label = self.labels[q_idx]

            # Time the query
            start = time.time()
            found_vec = traverse_tree(tree, query_vec)
            result_idx = vec_to_idx.get(found_vec.tobytes(), -1)
            elapsed = time.time() - start

            total_query_time += elapsed
            query_times.append(elapsed)

            if result_idx >= 0 and self.labels[result_idx] == query_label:
                correct += 1

        # Calculate statistics
        accuracy = correct / len(query_indices) if len(query_indices) > 0 else 0.0
        avg_query_time = total_query_time / len(query_indices) if len(query_indices) > 0 else 0.0
        query_times = np.array(query_times)

        result = {
            "k": k,
            "num_queries": len(query_indices),
            "num_embeddings": len(self.embeddings),
            "correct": correct,
            "accuracy": accuracy,
            "tree_build_time_s": tree_build_time,
            "total_query_time_ms": total_query_time * 1000,
            "avg_query_time_ms": avg_query_time * 1000,
            "min_query_time_ms": float(query_times.min()) * 1000,
            "max_query_time_ms": float(query_times.max()) * 1000,
            "std_query_time_ms": float(query_times.std()) * 1000,
            "median_query_time_ms": float(np.median(query_times)) * 1000,
        }

        print(f"\nResults for k={k}:")
        print(f"  Accuracy:        {accuracy*100:.2f}%")
        print(f"  Tree build time: {tree_build_time:.2f}s")
        print(f"  Avg query time:  {avg_query_time*1000:.4f}ms")

        return result

    def run_experiments(self, k_values, num_queries):
        """Run experiments for all k values."""
        print(f"\n{'#'*60}")
        print("STARTING EXPERIMENT SUITE")
        print(f"{'#'*60}")
        print(f"K values: {k_values}")
        print(f"Queries per experiment: {num_queries}")
        print(f"Dataset size: {len(self.embeddings)} embeddings")

        all_results = []
        total_start = time.time()

        for k in k_values:
            result = self.run_single_experiment(k, num_queries)
            all_results.append(result)

        total_time = time.time() - total_start

        print(f"\n{'#'*60}")
        print(f"ALL EXPERIMENTS COMPLETED in {total_time:.2f}s")
        print(f"{'#'*60}")

        return all_results

    def save_results(self, results):
        """Save results to CSV and JSON files."""
        # Save as JSON
        json_path = self.results_dir / f"experiment_results_{self.timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Results saved to {json_path}")

        # Save as CSV
        csv_path = self.results_dir / f"experiment_results_{self.timestamp}.csv"
        if results:
            fieldnames = results[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"[INFO] Results saved to {csv_path}")

        return json_path, csv_path

    def print_summary_table(self, results):
        """Print a formatted summary table."""
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        
        # Header
        print(
            f"{'k':>6} | {'Accuracy':>10} | {'Build Time':>12} | "
            f"{'Avg Query':>12} | {'Median Query':>12}"
        )
        print("-" * 80)

        # Data rows
        for r in results:
            print(
                f"{r['k']:>6} | {r['accuracy']*100:>9.2f}% | "
                f"{r['tree_build_time_s']:>10.2f}s | "
                f"{r['avg_query_time_ms']:>10.4f}ms | "
                f"{r['median_query_time_ms']:>10.4f}ms"
            )

        print("-" * 80)

        # Find best configurations
        best_accuracy = max(results, key=lambda x: x["accuracy"])
        best_speed = min(results, key=lambda x: x["avg_query_time_ms"])

        print(f"\nBest accuracy:   k={best_accuracy['k']} ({best_accuracy['accuracy']*100:.2f}%)")
        print(f"Fastest queries: k={best_speed['k']} ({best_speed['avg_query_time_ms']:.4f}ms avg)")

        # Accuracy-speed tradeoff analysis
        print(f"\n{'='*80}")
        print("ACCURACY vs SPEED TRADEOFF")
        print(f"{'='*80}")
        
        for r in results:
            # Simple score: accuracy * (1 / log(query_time))
            # Higher is better
            query_time = max(r["avg_query_time_ms"], 0.001)  # Avoid log(0)
            score = r["accuracy"] * 100 / np.log10(query_time + 1)
            print(f"k={r['k']:>3}: score={score:.2f} (accuracy={r['accuracy']*100:.1f}%, time={query_time:.3f}ms)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run experiments on face retrieval system"
    )

    # Experiment mode
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer configurations",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Full experiment suite",
    )
    p.add_argument(
        "--custom",
        action="store_true",
        help="Custom experiment with specified parameters",
    )

    # Custom experiment parameters
    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="K values to test (branching factors)",
    )
    p.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries per experiment",
    )

    # Data parameters
    p.add_argument(
        "--deeplake-uri",
        default="hub://activeloop/lfw",
        help="Deeplake hub URI",
    )
    p.add_argument(
        "--embeddings-path",
        default="data/embeddings/lfw_deeplake_embeddings.npz",
        help="Path to embeddings file",
    )

    # Output
    p.add_argument(
        "--results-dir",
        default="data/results",
        help="Directory to save results",
    )

    # Hardware
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--num-workers", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()

    # Determine experiment configuration
    if args.quick:
        k_values = [5, 10, 20]
        num_queries = 50
        print("[MODE] Quick experiment")
    elif args.full:
        k_values = [2, 3, 5, 8, 10, 15, 20, 30, 50]
        num_queries = 200
        print("[MODE] Full experiment suite")
    elif args.custom:
        k_values = args.k_values
        num_queries = args.num_queries
        print("[MODE] Custom experiment")
    else:
        # Default: moderate experiment
        k_values = [5, 10, 15, 20]
        num_queries = 100
        print("[MODE] Default experiment")

    print(f"K values: {k_values}")
    print(f"Queries: {num_queries}")

    # Initialize runner
    runner = ExperimentRunner(args)

    # Load embeddings
    runner.load_embeddings()

    # Run experiments
    results = runner.run_experiments(k_values, num_queries)

    # Save results
    runner.save_results(results)

    # Print summary
    runner.print_summary_table(results)

    print("\n[DONE] Experiments completed!")


if __name__ == "__main__":
    main()