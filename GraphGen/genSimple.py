import random
import networkx as nx

import utils

def gen_undirected_simple_graph_nm(n, m, seed=None):
    assert m <= n * (n - 1) / 2
    name = f'A simple undirected graph with {n} nodes and {m} edges.'
    g = nx.gnm_random_graph(n, m, seed=seed, directed=False)
    return name, g

def gen_directed_simple_graph_nm(n, m, seed=None):
    assert m <= n * (n - 1)
    name = f'A simple directed graph with {n} nodes and {m} edges.'
    g = nx.gnm_random_graph(n, m, seed=seed, directed=True)
    return name, g

def gen_tree_n(n, seed=None):
    name = f'A tree with {n} nodes.'
    g = nx.random_tree(n, seed=seed)
    return name, g

def gen_connect_nm(n, m, seed=None):
    name = f'A connected simple graph with {n} nodes and {m} edges.'
    assert m >= n - 1
    cnt = 0
    while True:
        g = nx.gnm_random_graph(n, m)
        if utils.connected(g):
            return name, g
        cnt += 1
        if cnt > 100:
            raise Exception("Failed")

def gen_c_connect_graph(ns, ms, seed=None):
    assert len(ns) == len(ms)
    c = len(ns)
    n = sum(ns)
    m = sum(ms)
    assert all([y >= x - 1 for x, y in zip(ns, ms)])
    name = f'A graph with {n} nodes, {m} edges, and {c} connected components.'
    _, g = gen_connect_nm(ns[0], ms[0], seed)
    for i, (x, y) in enumerate(zip(ns, ms)):
        if i == 0:
            continue
        utils.join(g, gen_connect_nm(x, y)[1])

    return name, g

def gen_tree_nd(n, d, seed=None):
    name = f'A tree with {n} nodes and depth {d}.'

    def _sample():
        g = nx.graph.Graph()
        depth = {}
        candidate = set()
        g.add_node(0)
        depth[0] = 0
        candidate.add(0)
        next_idx = 1
        while len(g.nodes) < n:
            u = random.choice(list(candidate))
            g.add_node(next_idx)
            g.add_edge(u, next_idx)
            depth[next_idx] = depth[u] + 1
            if depth[next_idx] < d:
                candidate.add(next_idx)

            next_idx += 1

        if max(depth.values()) == d:
            return g
        else:
            return None

    cnt = 0
    while True:
        g = _sample()
        if g is not None:
            return name, g
        cnt += 1
        if cnt > 100:
            raise Exception("Failed")
        
if __name__ == '__main__':
    # und simple n m
    name, g = gen_undirected_simple_graph_nm(5, 8)
    print(name, g.nodes, g.edges)

    # und directed n m
    name, g = gen_directed_simple_graph_nm(5, 8)
    print(name, g.nodes, g.edges)

    # gen tree n
    name, g = gen_tree_n(10)
    print(name, g.nodes, g.edges)

    # conn nm
    name, g = gen_connect_nm(10, 10)
    print(name, g.nodes, g.edges, utils.connected(g))

    # c conn
    name, g = gen_c_connect_graph([5, 3], [4, 2])
    print(name, g.nodes, g.edges, utils.connected(g))

    # gen tree with depth d
    name, g = gen_tree_nd(8, 4)
    print(name, g.nodes, g.edges)
