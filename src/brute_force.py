#!/usr/bin/env python3
"""
Brute-force nearest neighbor baseline for face retrieval.
Compares against vocabulary tree performance.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from collections import defaultdict


def load_embeddings(path):
    """Load embeddings from npz file."""
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["labels"], data["paths"]


def stratified_split(embeddings, labels, test_ratio=0.2, min_samples=2, seed=42):
    """Stratified train/test split."""
    np.random.seed(seed)
    
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    
    train_indices = []
    test_indices = []
    
    for label, indices in label_to_indices.items():
        if len(indices) < min_samples:
            continue
        
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        n_test = max(1, int(len(indices) * test_ratio))
        n_test = min(n_test, len(indices) - 1)
        
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
    
    return np.array(train_indices), np.array(test_indices)


def brute_force_search(query_vec, database, k=1):
    """Find k nearest neighbors by brute force."""
    distances = np.linalg.norm(database - query_vec, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices, distances[nearest_indices]


def evaluate(train_embeddings, train_labels, test_embeddings, test_labels, num_queries=500, seed=42):
    """Evaluate brute-force retrieval."""
    np.random.seed(seed)
    
    n_test = len(test_embeddings)
    num_queries = min(num_queries, n_test)
    query_indices = np.random.choice(n_test, size=num_queries, replace=False)
    
    correct = 0
    total_time = 0.0
    
    for q_idx in query_indices:
        query_vec = test_embeddings[q_idx]
        query_label = test_labels[q_idx]
        
        start = time.time()
        nearest_idx, _ = brute_force_search(query_vec, train_embeddings, k=1)
        elapsed = time.time() - start
        
        total_time += elapsed
        
        if train_labels[nearest_idx[0]] == query_label:
            correct += 1
    
    accuracy = correct / num_queries
    avg_time = total_time / num_queries
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": num_queries,
        "avg_query_time_ms": avg_time * 1000,
        "total_time_ms": total_time * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Brute-force NN baseline")
    parser.add_argument("--embeddings-path", default="data/embeddings/lfw_deeplake_embeddings.npz")
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Load embeddings
    print(f"[INFO] Loading embeddings from {args.embeddings_path}")
    embeddings, labels, paths = load_embeddings(args.embeddings_path)
    labels = np.array(labels)
    print(f"[INFO] Loaded {len(embeddings)} embeddings")
    
    # Split
    print(f"[INFO] Creating stratified split (test_ratio={args.test_ratio})")
    train_idx, test_idx = stratified_split(embeddings, labels, args.test_ratio, seed=args.seed)
    
    train_embeddings = embeddings[train_idx]
    train_labels = labels[train_idx]
    test_embeddings = embeddings[test_idx]
    test_labels = labels[test_idx]
    
    print(f"[INFO] Train: {len(train_embeddings)}, Test: {len(test_embeddings)}")
    
    # Evaluate
    print(f"\n[INFO] Running brute-force evaluation with {args.num_queries} queries...")
    results = evaluate(train_embeddings, train_labels, test_embeddings, test_labels, 
                       num_queries=args.num_queries, seed=args.seed)
    
    # Print results
    print(f"\n{'='*50}")
    print("BRUTE-FORCE BASELINE RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:        {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"Avg query time:  {results['avg_query_time_ms']:.4f}ms")
    print(f"Total time:      {results['total_time_ms']:.2f}ms")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()