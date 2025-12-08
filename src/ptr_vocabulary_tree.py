import numpy as np
from sklearn.cluster import KMeans


class Node:
    def __init__(self, center, children=None):
        self.center = center
        self.children = children 

def split_tree(vectors, k):
    return build_node(vectors, k)

def build_node(cluster, k):
    if len(cluster) == 0:
        return None
    
    if len(cluster) == 1:
        return Node(center=cluster[0])
    
    center = np.mean(cluster, axis=0)
    children_clusters = split(cluster, k)
    children = [build_node(c, k) for c in children_clusters]
    children = [c for c in children if c is not None]
    
    return Node(center=center, children=children)

def split(cluster, k):
    if len(cluster) <= k:
        return [cluster[i:i+1] for i in range(len(cluster))]
    
    kmeans = KMeans(n_clusters=k, n_init=10)
    assignments = kmeans.fit_predict(cluster)
    return [cluster[assignments == i] for i in range(k)]

def traverse_tree(node, query):
    while node.children:
        best_dist = np.inf
        best_child = None
        
        for child in node.children:
            dist = np.linalg.norm(child.center - query)
            if dist < best_dist:
                best_dist = dist
                best_child = child
        
        node = best_child
    
    return node.center

def query(vec_to_img, tree, query_vec):
    found_vec = traverse_tree(tree, query_vec)
    return vec_to_img[found_vec.tobytes()]