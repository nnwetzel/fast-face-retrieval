#!/usr/bin/env python3
"""
Main pipeline for face retrieval using deep features and vocabulary trees.

Usage:
    python main.py --build --deeplake-uri hub://activeloop/lfw
    python main.py --query --query-idx 0
    python main.py --evaluate --num-queries 100
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from feature_extraction import (
    load_deeplake_dataset,
    DeeplakeFaceDataset,
    get_transform,
    build_model,
    extract,
    save,
)
from ptr_vocabulary_tree import split_tree, traverse_tree

import torch
from torch.utils.data import DataLoader


def load_or_extract_embeddings(args):
    """Load embeddings from file or extract them from dataset."""
    embeddings_path = Path(args.embeddings_path)
    
    if embeddings_path.exists() and not args.force_extract:
        print(f"[INFO] Loading existing embeddings from {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"]
        paths = data["paths"]
        print(f"[INFO] Loaded {len(embeddings)} embeddings")
        return embeddings, labels, paths
    
    print(f"[INFO] Extracting embeddings from {args.deeplake_uri}")
    device = torch.device(args.device)
    tfm = get_transform()
    
    items = load_deeplake_dataset(args.deeplake_uri)
    ds = DeeplakeFaceDataset(items, tfm)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    model = build_model(device)
    embeddings, labels, paths = extract(model, dl, device)
    save(args.embeddings_path, embeddings, labels, paths)
    
    return embeddings, labels, paths


def build_vocabulary_tree(embeddings, k):
    """Build vocabulary tree from embeddings."""
    print(f"[INFO] Building vocabulary tree with k={k}")
    start = time.time()
    tree = split_tree(embeddings, k)
    elapsed = time.time() - start
    print(f"[INFO] Tree built in {elapsed:.2f}s")
    return tree


def save_tree(tree, path):
    """Save vocabulary tree to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tree, f)
    print(f"[INFO] Tree saved to {path}")


def load_tree(path):
    """Load vocabulary tree from file."""
    with open(path, "rb") as f:
        tree = pickle.load(f)
    print(f"[INFO] Tree loaded from {path}")
    return tree


def build_vec_to_idx(embeddings):
    """Build mapping from embedding bytes to index."""
    vec_to_idx = {}
    for idx, emb in enumerate(embeddings):
        vec_to_idx[emb.tobytes()] = idx
    return vec_to_idx


def query_single(tree, vec_to_idx, query_vec):
    """Query the tree for a single vector, return matched index."""
    found_vec = traverse_tree(tree, query_vec)
    return vec_to_idx.get(found_vec.tobytes(), -1)


def query_with_timing(tree, vec_to_idx, query_vec):
    """Query with timing information."""
    start = time.time()
    result_idx = query_single(tree, vec_to_idx, query_vec)
    elapsed = time.time() - start
    return result_idx, elapsed


def evaluate_retrieval(tree, vec_to_idx, embeddings, labels, num_queries=100, seed=42):
    """
    Evaluate retrieval performance.
    
    For each query, we check if the retrieved image has the same label.
    """
    np.random.seed(seed)
    n = len(embeddings)
    query_indices = np.random.choice(n, size=min(num_queries, n), replace=False)
    
    correct = 0
    total_time = 0.0
    results = []
    
    print(f"\n[EVAL] Running {len(query_indices)} queries...")
    
    for q_idx in query_indices:
        query_vec = embeddings[q_idx]
        query_label = labels[q_idx]
        
        result_idx, elapsed = query_with_timing(tree, vec_to_idx, query_vec)
        total_time += elapsed
        
        if result_idx >= 0:
            result_label = labels[result_idx]
            is_correct = (result_label == query_label)
        else:
            result_label = None
            is_correct = False
        
        if is_correct:
            correct += 1
        
        results.append({
            "query_idx": q_idx,
            "query_label": query_label,
            "result_idx": result_idx,
            "result_label": result_label,
            "correct": is_correct,
            "time": elapsed,
        })
    
    accuracy = correct / len(query_indices) if query_indices.size > 0 else 0.0
    avg_time = total_time / len(query_indices) if query_indices.size > 0 else 0.0
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total queries:     {len(query_indices)}")
    print(f"Correct matches:   {correct}")
    print(f"Accuracy:          {accuracy*100:.2f}%")
    print(f"Total time:        {total_time*1000:.2f}ms")
    print(f"Avg query time:    {avg_time*1000:.4f}ms")
    print(f"{'='*50}\n")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(query_indices),
        "avg_query_time_ms": avg_time * 1000,
        "total_time_ms": total_time * 1000,
        "details": results,
    }


def run_single_query(args, tree, vec_to_idx, embeddings, labels, paths):
    """Run a single query and display results."""
    query_idx = args.query_idx
    
    if query_idx < 0 or query_idx >= len(embeddings):
        print(f"[ERROR] Invalid query index {query_idx}. Must be 0-{len(embeddings)-1}")
        return
    
    query_vec = embeddings[query_idx]
    query_label = labels[query_idx]
    query_path = paths[query_idx]
    
    print(f"\n{'='*50}")
    print("QUERY")
    print(f"{'='*50}")
    print(f"Index: {query_idx}")
    print(f"Label: {query_label}")
    print(f"Path:  {query_path}")
    
    result_idx, elapsed = query_with_timing(tree, vec_to_idx, query_vec)
    
    print(f"\n{'='*50}")
    print("RESULT")
    print(f"{'='*50}")
    
    if result_idx >= 0:
        result_label = labels[result_idx]
        result_path = paths[result_idx]
        match = "YES" if result_label == query_label else "NO"
        
        print(f"Index: {result_idx}")
        print(f"Label: {result_label}")
        print(f"Path:  {result_path}")
        print(f"Match: {match}")
    else:
        print("No result found")
    
    print(f"Time:  {elapsed*1000:.4f}ms")
    print(f"{'='*50}\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Face retrieval pipeline using vocabulary trees"
    )
    
    # Mode selection
    p.add_argument("--build", action="store_true", help="Build tree from embeddings")
    p.add_argument("--query", action="store_true", help="Run a single query")
    p.add_argument("--evaluate", action="store_true", help="Evaluate retrieval performance")
    
    # Paths
    p.add_argument(
        "--deeplake-uri",
        default="hub://activeloop/lfw",
        help="Deeplake hub URI for dataset",
    )
    p.add_argument(
        "--embeddings-path",
        default="data/embeddings/lfw_deeplake_embeddings.npz",
        help="Path to save/load embeddings",
    )
    p.add_argument(
        "--tree-path",
        default="data/trees/vocabulary_tree.pkl",
        help="Path to save/load vocabulary tree",
    )
    
    # Tree parameters
    p.add_argument(
        "--tree-k",
        type=int,
        default=10,
        help="Branching factor for vocabulary tree",
    )
    
    # Query parameters
    p.add_argument(
        "--query-idx",
        type=int,
        default=0,
        help="Index of image to use as query",
    )
    
    # Evaluation parameters
    p.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries for evaluation",
    )
    
    # Feature extraction parameters
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of embeddings",
    )
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Check that at least one mode is selected
    if not (args.build or args.query or args.evaluate):
        print("[INFO] No mode selected. Use --build, --query, or --evaluate")
        print("[INFO] Running full pipeline: build + evaluate")
        args.build = True
        args.evaluate = True
    
    # Load or extract embeddings
    embeddings, labels, paths = load_or_extract_embeddings(args)
    
    tree_path = Path(args.tree_path)
    
    # Build mode: create and save tree
    if args.build:
        tree = build_vocabulary_tree(embeddings, args.tree_k)
        save_tree(tree, tree_path)
    else:
        # Load existing tree
        if not tree_path.exists():
            print(f"[ERROR] Tree not found at {tree_path}. Run with --build first.")
            return
        tree = load_tree(tree_path)
    
    # Build vector-to-index mapping
    vec_to_idx = build_vec_to_idx(embeddings)
    
    # Query mode: run single query
    if args.query:
        run_single_query(args, tree, vec_to_idx, embeddings, labels, paths)
    
    # Evaluate mode: run evaluation
    if args.evaluate:
        evaluate_retrieval(
            tree, vec_to_idx, embeddings, labels, num_queries=args.num_queries
        )


if __name__ == "__main__":
    main()