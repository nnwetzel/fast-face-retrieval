import numpy as np
from sklearn.cluster import KMeans


class Node:
    """
    A node in the vocabulary tree.
    
    Each node stores a center vector (the mean of all vectors in its cluster)
    and optionally a list of child nodes. Leaf nodes have no children.
    """
    def __init__(self, center, children=None):
        self.center = center
        self.children = children 


def split_tree(vectors, k):
    """
    Build a vocabulary tree from a set of feature vectors.
    
    Args:
        vectors: Array of feature vectors (e.g., image descriptors)
        k: Branching factor - number of children per node
    
    Returns:
        Root node of the constructed vocabulary tree
    """
    return build_node(vectors, k)


def build_node(cluster, k):
    """
    Recursively build a node and its subtree from a cluster of vectors.
    
    The function computes the center of the current cluster, then splits
    the cluster into k sub-clusters using k-means, and recursively builds
    child nodes for each sub-cluster.
    
    Args:
        cluster: Array of feature vectors belonging to this node's cluster
        k: Branching factor for splitting
    
    Returns:
        A Node representing this cluster, or None if the cluster is empty
    """
    if len(cluster) == 0:
        return None
    
    # Base case: single vector becomes a leaf node
    if len(cluster) == 1:
        return Node(center=cluster[0])
    
    # Compute the centroid of all vectors in this cluster
    center = np.mean(cluster, axis=0)
    
    # Split into k sub-clusters and recursively build child nodes
    children_clusters = split(cluster, k)
    children = [build_node(c, k) for c in children_clusters]
    
    # Filter out any None children (from empty clusters)
    children = [c for c in children if c is not None]
    
    return Node(center=center, children=children)


def split(cluster, k):
    """
    Split a cluster of vectors into k sub-clusters using k-means.
    
    If the cluster has fewer than k vectors, each vector becomes its own
    sub-cluster (no k-means needed).
    
    Args:
        cluster: Array of feature vectors to split
        k: Number of sub-clusters to create
    
    Returns:
        List of k sub-clusters (each is an array of vectors)
    """
    # If we have fewer vectors than k, just put each in its own cluster
    if len(cluster) <= k:
        return [cluster[i:i+1] for i in range(len(cluster))]
    
    # Use k-means to partition vectors into k clusters
    kmeans = KMeans(n_clusters=k, n_init=10)
    assignments = kmeans.fit_predict(cluster)
    
    # Group vectors by their assigned cluster
    return [cluster[assignments == i] for i in range(k)]


def traverse_tree(node, query):
    """
    Traverse the tree from root to leaf, following the nearest child at each level.
    
    This performs approximate nearest neighbor search by greedily descending
    the tree, always choosing the child whose center is closest to the query.
    This is much faster than exhaustive search but may not find the true
    nearest neighbor.
    
    Args:
        node: Root node of the tree (or subtree)
        query: Query vector to search for
    
    Returns:
        The center vector of the leaf node reached (approximate nearest neighbor)
    """
    # Descend until we reach a leaf node (no children)
    while node.children:
        best_dist = np.inf
        best_child = None
        
        # Find the child with the closest center to the query
        for child in node.children:
            dist = np.linalg.norm(child.center - query)
            if dist < best_dist:
                best_dist = dist
                best_child = child
        
        node = best_child
    
    return node.center


def query(vec_to_img, tree, query_vec):
    """
    Find the image associated with the approximate nearest neighbor of a query vector.
    
    Args:
        vec_to_img: Dictionary mapping vector bytes to image identifiers
        tree: Root node of the vocabulary tree
        query_vec: Query feature vector
    
    Returns:
        The image identifier associated with the nearest matching vector
    """
    # Find the approximate nearest neighbor in the tree
    found_vec = traverse_tree(tree, query_vec)
    
    # Look up and return the corresponding image
    return vec_to_img[found_vec.tobytes()]
