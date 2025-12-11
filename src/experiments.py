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
from collections import defaultdict
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


def stratified_train_test_split(embeddings, labels, test_ratio=0.2, min_samples_per_class=2, seed=42):
    """
    Split data into train and test sets, stratified by label.
    
    For face retrieval, we need:
    - Train set: images used to build the vocabulary tree (database)
    - Test set: query images (must have same identity in train set to be retrievable)
    
    Only includes classes that have at least min_samples_per_class samples,
    so that each class can have at least one sample in both train and test.
    """
    np.random.seed(seed)
    
    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    
    train_indices = []
    test_indices = []
    skipped_classes = 0
    
    for label, indices in label_to_indices.items():
        if len(indices) < min_samples_per_class:
            # Skip classes with too few samples - can't split them meaningfully
            skipped_classes += 1
            continue
        
        # Shuffle indices for this class
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        # Split: at least 1 in test, rest in train
        n_test = max(1, int(len(indices) * test_ratio))
        n_test = min(n_test, len(indices) - 1)  # Ensure at least 1 in train
        
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # Shuffle the final indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    print(f"[INFO] Stratified split complete:")
    print(f"       Total classes: {len(label_to_indices)}")
    print(f"       Classes used: {len(label_to_indices) - skipped_classes}")
    print(f"       Classes skipped (< {min_samples_per_class} samples): {skipped_classes}")
    print(f"       Train samples: {len(train_indices)}")
    print(f"       Test samples: {len(test_indices)}")
    
    return train_indices, test_indices


def random_train_test_split(embeddings, labels, test_ratio=0.2, seed=42):
    """
    Simple random split (not stratified).
    
    Note: This may result in some query images having no matching identity
    in the database, which will always be incorrect.
    """
    np.random.seed(seed)
    
    n = len(embeddings)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    n_test = int(n * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    print(f"[INFO] Random split complete:")
    print(f"       Train samples: {len(train_indices)}")
    print(f"       Test samples: {len(test_indices)}")
    
    return train_indices, test_indices


class ExperimentRunner:
    """Runs experiments with different configurations."""

    def __init__(self, args):
        self.args = args
        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Full dataset
        self.embeddings = None
        self.labels = None
        self.paths = None
        
        # Train/test splits
        self.train_indices = None
        self.test_indices = None
        self.train_embeddings = None
        self.train_labels = None
        self.test_embeddings = None
        self.test_labels = None

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

    def create_train_test_split(self):
        """Create train/test split for proper evaluation."""
        print(f"\n[INFO] Creating train/test split (test_ratio={self.args.test_ratio})...")

        # Convert labels to numpy array if it's a list
        if isinstance(self.labels, list):
            self.labels = np.array(self.labels)

        if self.args.split_strategy == "stratified":
            self.train_indices, self.test_indices = stratified_train_test_split(
                self.embeddings,
                self.labels,
                test_ratio=self.args.test_ratio,
                min_samples_per_class=self.args.min_samples_per_class,
                seed=self.args.seed,
            )
        else:
            self.train_indices, self.test_indices = random_train_test_split(
                self.embeddings,
                self.labels,
                test_ratio=self.args.test_ratio,
                seed=self.args.seed,
            )

        # Create train/test arrays
        self.train_embeddings = self.embeddings[self.train_indices]
        self.train_labels = self.labels[self.train_indices]
        self.test_embeddings = self.embeddings[self.test_indices]
        self.test_labels = self.labels[self.test_indices]

        # Analyze split quality
        train_label_set = set(self.train_labels)
        test_label_set = set(self.test_labels)
        overlap = train_label_set & test_label_set

        print(f"[INFO] Split analysis:")
        print(f"       Unique labels in train: {len(train_label_set)}")
        print(f"       Unique labels in test: {len(test_label_set)}")
        print(f"       Labels in both sets: {len(overlap)}")
        print(f"       Test queries with matching train identity: {sum(1 for l in self.test_labels if l in train_label_set)}")
    
    def build_vec_to_idx(self, embeddings):
        """Build mapping from embedding bytes to index."""
        vec_to_idx = {}
        for idx, emb in enumerate(embeddings):
            vec_to_idx[emb.tobytes()] = idx
        return vec_to_idx

    def run_single_experiment(self, k, num_queries):
        """Run a single experiment with given k value."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: k={k}, queries={num_queries}")
        print(f"{'='*60}")

        # Build tree using TRAIN embeddings only
        print(f"[INFO] Building tree with k={k} on {len(self.train_embeddings)} train embeddings...")
        tree_start = time.time()
        tree = split_tree(self.train_embeddings, k)
        tree_build_time = time.time() - tree_start
        print(f"[INFO] Tree built in {tree_build_time:.2f}s")

        # Build mapping for train embeddings
        vec_to_idx = self.build_vec_to_idx(self.train_embeddings)

        # Select query indices from TEST set
        n_test = len(self.test_embeddings)
        num_queries = min(num_queries, n_test)
        
        np.random.seed(self.args.seed)
        query_indices = np.random.choice(n_test, size=num_queries, replace=False)

        # Run queries
        correct = 0
        total_query_time = 0.0
        query_times = []
        
        # Track retrievable queries (those with matching identity in train set)
        train_label_set = set(self.train_labels)
        retrievable_queries = 0
        retrievable_correct = 0

        for q_idx in query_indices:
            query_vec = self.test_embeddings[q_idx]
            query_label = self.test_labels[q_idx]
            
            # Check if this query has a matching identity in train set
            is_retrievable = query_label in train_label_set
            if is_retrievable:
                retrievable_queries += 1

            # Time the query
            start = time.time()
            found_vec = traverse_tree(tree, query_vec)
            result_idx = vec_to_idx.get(found_vec.tobytes(), -1)
            elapsed = time.time() - start

            total_query_time += elapsed
            query_times.append(elapsed)

            # Check if retrieved image has same label as query
            if result_idx >= 0:
                result_label = self.train_labels[result_idx]
                if result_label == query_label:
                    correct += 1
                    if is_retrievable:
                        retrievable_correct += 1

        # Calculate statistics
        accuracy = correct / num_queries if num_queries > 0 else 0.0
        retrievable_accuracy = (
            retrievable_correct / retrievable_queries 
            if retrievable_queries > 0 else 0.0
        )
        avg_query_time = total_query_time / num_queries if num_queries > 0 else 0.0
        query_times = np.array(query_times)

        result = {
            "k": k,
            "num_queries": num_queries,
            "num_train_embeddings": len(self.train_embeddings),
            "num_test_embeddings": len(self.test_embeddings),
            "correct": correct,
            "accuracy": accuracy,
            "retrievable_queries": retrievable_queries,
            "retrievable_correct": retrievable_correct,
            "retrievable_accuracy": retrievable_accuracy,
            "tree_build_time_s": tree_build_time,
            "total_query_time_ms": total_query_time * 1000,
            "avg_query_time_ms": avg_query_time * 1000,
            "min_query_time_ms": float(query_times.min()) * 1000 if len(query_times) > 0 else 0,
            "max_query_time_ms": float(query_times.max()) * 1000 if len(query_times) > 0 else 0,
            "std_query_time_ms": float(query_times.std()) * 1000 if len(query_times) > 0 else 0,
            "median_query_time_ms": float(np.median(query_times)) * 1000 if len(query_times) > 0 else 0,
        }

        print(f"\nResults for k={k}:")
        print(f"  Overall Accuracy:     {accuracy*100:.2f}% ({correct}/{num_queries})")
        print(f"  Retrievable Accuracy: {retrievable_accuracy*100:.2f}% ({retrievable_correct}/{retrievable_queries})")
        print(f"  Tree build time:      {tree_build_time:.2f}s")
        print(f"  Avg query time:       {avg_query_time*1000:.4f}ms")

        return result

    def run_experiments(self, k_values, num_queries):
        """Run experiments for all k values."""
        print(f"\n{'#'*60}")
        print("STARTING EXPERIMENT SUITE")
        print(f"{'#'*60}")
        print(f"K values: {k_values}")
        print(f"Queries per experiment: {num_queries}")
        print(f"Train set size: {len(self.train_embeddings)} embeddings")
        print(f"Test set size: {len(self.test_embeddings)} embeddings")

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
        print(f"\n{'='*100}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*100}")
        
        # Header
        print(
            f"{'k':>6} | {'Accuracy':>10} | {'Retr. Acc':>10} | {'Build Time':>12} | "
            f"{'Avg Query':>12} | {'Median Query':>12}"
        )
        print("-" * 100)

        # Data rows
        for r in results:
            print(
                f"{r['k']:>6} | {r['accuracy']*100:>9.2f}% | "
                f"{r['retrievable_accuracy']*100:>9.2f}% | "
                f"{r['tree_build_time_s']:>10.2f}s | "
                f"{r['avg_query_time_ms']:>10.4f}ms | "
                f"{r['median_query_time_ms']:>10.4f}ms"
            )

        print("-" * 100)

        # Find best configurations
        best_accuracy = max(results, key=lambda x: x["accuracy"])
        best_retrievable = max(results, key=lambda x: x["retrievable_accuracy"])
        best_speed = min(results, key=lambda x: x["avg_query_time_ms"])

        print(f"\nBest overall accuracy:     k={best_accuracy['k']} ({best_accuracy['accuracy']*100:.2f}%)")
        print(f"Best retrievable accuracy: k={best_retrievable['k']} ({best_retrievable['retrievable_accuracy']*100:.2f}%)")
        print(f"Fastest queries:           k={best_speed['k']} ({best_speed['avg_query_time_ms']:.4f}ms avg)")

        # Accuracy-speed tradeoff analysis
        print(f"\n{'='*100}")
        print("ACCURACY vs SPEED TRADEOFF")
        print(f"{'='*100}")
        
        for r in results:
            query_time = max(r["avg_query_time_ms"], 0.001)
            score = r["retrievable_accuracy"] * 100 / np.log10(query_time + 1)
            print(
                f"k={r['k']:>3}: score={score:.2f} "
                f"(retr_acc={r['retrievable_accuracy']*100:.1f}%, time={query_time:.3f}ms)"
            )


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

    # Train/test split parameters
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    p.add_argument(
        "--split-strategy",
        choices=["stratified", "random"],
        default="stratified",
        help="How to split data: stratified (by label) or random",
    )
    p.add_argument(
        "--min-samples-per-class",
        type=int,
        default=2,
        help="Minimum samples per class for stratified split (default: 2)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
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
    print(f"Split strategy: {args.split_strategy}")
    print(f"Test ratio: {args.test_ratio}")

    # Initialize runner
    runner = ExperimentRunner(args)

    # Load embeddings
    runner.load_embeddings()

    # Create train/test split
    runner.create_train_test_split()

    # Run experiments
    results = runner.run_experiments(k_values, num_queries)

    # Save results
    runner.save_results(results)

    # Print summary
    runner.print_summary_table(results)

    print("\n[DONE] Experiments completed!")


if __name__ == "__main__":
    main()