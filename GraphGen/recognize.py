import numpy as np

##### from utils.py
class DSU:
    def __init__(self):
        self.pa = {}
        self.sz = {}

    def size(self, x):
        if x not in self.sz:
            return 1
        else:
            return self.sz[x]

    def query(self, x):
        if x not in self.pa or x == self.pa[x]:
            return x
        else:
            self.pa[x] = self.query(self.pa[x])
            return self.pa[x]

    def merge(self, x, y):
        if x not in self.pa:
            self.pa[x] = x
            self.sz[x] = 1
        if y not in self.pa:
            self.pa[y] = y
            self.sz[y] = 1
        x = self.query(x)
        y = self.query(y)
        if x == y:
            return

        if self.sz[x] < self.sz[y]:
            x, y = y, x
        self.pa[y] = x
        self.sz[x] += self.sz[y]
        self.sz[y] = 0

def connected(graph):
    dsu = DSU()
    for i, j in graph.edges:
        dsu.merge(i, j)

    # assuming node 0 is always in graph
    if dsu.size(0) < len(graph.nodes):
        return False
    else:
        return True

def join(g1, g2):
    idx = {}
    start_idx = max(g1.nodes)+1

    for u in g2.nodes:
        idx[u] = start_idx
        g1.add_node(idx[u])
        start_idx += 1

    for u, v in g2.edges:
        g1.add_edge(idx[u], idx[v])
##### from utils.py

def get_node_num(adj: np.ndarray):
    return adj.shape[0]

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