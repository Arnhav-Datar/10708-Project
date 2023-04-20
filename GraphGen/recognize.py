import numpy as np
from graph_utils import *

def get_node_num(adj: np.ndarray):
    return adj.shape[0] - (adj.sum(axis=1) == 0).sum()

def get_edge_num(adj: np.ndarray):
    n = get_node_num(adj)
    for i in range(n):
        for j in range(n):
            if adj[i, j] != adj[j, i]:
                raise ValueError('adj is not symmetric')
    return np.sum(adj) // 2

def get_max_diameter(adj: np.ndarray):
    n = get_node_num(adj)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = 1 if adj[i, j] else np.inf

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

    if np.all(dist == np.inf):
        return 0

    return int(dist[dist < np.inf].max())

def get_connected_component_num(adj: np.ndarray):
    dsu = DSU()

    n = get_node_num(adj)
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                dsu.merge(i, j)

    cc = set()
    for i in range(n):
        cc.add(dsu.size(dsu.query(i)))

    return len(list(cc))

def get_degree_seq(adj: np.ndarray):
    return sorted(np.sum(adj, axis=1), reverse=True)