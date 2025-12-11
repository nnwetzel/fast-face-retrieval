# Fast Face Retrieval using Deep Features on Vocabulary Trees

A face retrieval system that combines deep neural network feature extraction with vocabulary tree indexing for efficient and accurate image search.

## Overview

This project implements a face retrieval pipeline that:
1. Extracts deep feature embeddings from face images using a pretrained CNN
2. Indexes these features using a vocabulary tree structure for fast similarity search
3. Retrieves matching faces from a database given a query image

> **Note:** Check `k_ary_vocab_tree_bugged.py` for an experimental implementation of a k-ary vocabulary tree. This version contains known bugs and is kept for reference/debugging purposes.

## Installation

````bash
# Clone the repository
git clone <repository-url>
cd face-retrieval

# Install dependencies
pip install -r requirements.txt
````

## Usage

### Building the Tree and Running Evaluation

````bash
# Build vocabulary tree and evaluate (default behavior)
python main.py --build --evaluate

# Specify custom parameters
python main.py --build --tree-k 10 --deeplake-uri hub://activeloop/lfw
````

### Running a Single Query

````bash
# Query with a specific image index
python main.py --query --query-idx 42
````

### Running Experiments

````bash
# Quick test with fewer configurations
python experiments.py --quick

# Full experiment suite
python experiments.py --full

# Custom experiment
python experiments.py --custom --k-values 5 10 20 --num-queries 200
````

### Brute-Force Baseline

````bash
# Run brute-force nearest neighbor baseline for comparison
python brute_force.py --num-queries 500
````

## Project Structure

```
├── main.py                    # Main pipeline for building trees and querying
├── experiments.py             # Experiment runner for testing different configurations
├── brute_force.py             # Brute-force NN baseline for comparison
├── feature_extraction.py      # Deep feature extraction using pretrained CNN
├── ptr_vocabulary_tree.py     # Vocabulary tree implementation
├── k_ary_vocab_tree_bugged.py # Experimental k-ary tree (buggy, for reference)
└── data/
    ├── embeddings/            # Cached feature embeddings
    ├── trees/                 # Saved vocabulary trees
    └── results/               # Experiment results (JSON/CSV)
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tree-k` | Branching factor for vocabulary tree | 10 |
| `--num-queries` | Number of queries for evaluation | 100 |
| `--test-ratio` | Fraction of data for testing | 0.2 |
| `--split-strategy` | Train/test split method (`stratified` or `random`) | stratified |
| `--device` | Computation device (`cuda` or `cpu`) | auto-detect |

## Datasets

The system supports loading from DeepLake hub:
- **LFW (Labeled Faces in the Wild)**: ~13,000 images (default)
- **CelebA**: 200,000+ celebrity faces
- **VGGFace2**: 3M+ images (for large-scale testing)

## Evaluation Metrics

- **Overall Accuracy**: Percentage of queries returning correct identity
- **Retrievable Accuracy**: Accuracy only for queries with matching identity in database
- **Query Time**: Average/median time per query (ms)
- **Tree Build Time**: Time to construct the vocabulary tree

Khoury College of Computer Sciences, Northeastern University

1. Nistér, D., & Stewénius, H. (2006). Scalable recognition with a vocabulary tree. CVPR.
2. Parkhi, O. M., Vedaldi, A., & Zisserman, A. (2015). Deep face recognition. BMVC.
3. Wang, M., & Deng, W. (2018). Deep face recognition: A survey. Neurocomputing.
