import numpy as np
from sklearn.cluster import KMeans

"""
BUGGED because KMeans can produce big clusters which can cause the tree to be unbalanced
"""

def split_tree(vectors, k) -> np.ndarray:
    """
    Constructs a vocabulary tree (in k-ary structure) using k-means clustering. 
    Args:
        vectors (np.ndarray): Array of shape (N, D) where N is the number of vectors and D is their dimensionality.
        k (int): Branching factor for the tree.
    Returns:
        np.ndarray: Array representing the vocabulary tree.
    """
    queue = [vectors]
    result = [] # Just temporary, gets converted to numpy array since it does all the optimizations internally

    while queue:
        cluster = queue.pop(0) # The array gets made in BFS order and no pointers for better sequential reads
        if len(cluster) == 0:
            result.append(np.full(vectors.shape[1], np.nan)) # If the tree doesnt split properly put NaN value to mark it
            continue
        if len(cluster) == 1:
            result.append(cluster[0]) # Store the original vector instead of the recomputed mean so that there is no precision loss when hashing it
            continue
        result.append(np.mean(cluster, axis=0)) # Store the centroid if there are more than 1 point in the cluster
        queue.extend(split(cluster, k))

    return np.array(result, dtype=np.float32)

def split(cluster, k):
    if len(cluster) <= k:
        children = [cluster[i:i+1] for i in range(len(cluster))]
        while len(children) < k:
            children.append(np.empty((0, cluster.shape[1]))) # Add empty clusters to maintain k-ary str
    else:
        kmeans = KMeans(n_clusters=k, n_init=10)
        assignments = kmeans.fit_predict(cluster)
        children = [cluster[assignments == i] for i in range(k)]

    return children

def traverse_tree(tree, query, k) -> np.ndarray:
    node_idx = 0
    
    while True:
        first_child = node_idx * k + 1

        if first_child >= len(tree):
            return tree[node_idx]
        
        children = tree[first_child:first_child + k]
        distances = np.linalg.norm(children - query, axis=1)
        distances[np.isnan(distances)] = np.inf
        
        best_child = np.argmin(distances)
        node_idx = first_child + best_child

def query(vec_to_img, tree, query, k):
    found_vec = traverse_tree(tree, query, k)
    return vec_to_img[found_vec.tobytes()]