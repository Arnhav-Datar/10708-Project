import networkx as nx

import utils

def get_node_num(g: nx.graph.Graph):
    return len(g.nodes)

def get_edge_num(g: nx.graph.Graph):
    return len(g.edges)

def get_connected_component_sizes(g: nx.graph.Graph):
    assert not g.is_directed()
    dsu = utils.DSU()

    for (u, v) in g.edges:
        dsu.merge(u, v)

    cc = set()
    for u in g.nodes:
        cc.add(dsu.size(dsu.query(u)))

    return list(cc)

def get_degree(g: nx.graph.Graph):
    assert not g.is_directed()
    return [len(x[1]) for x in g.adjacency()]

def get_is_tree(g: nx.graph.Graph):
    assert not g.is_directed()

    if len(get_connected_component_sizes(g)) == 1 and \
            get_edge_num(g) == get_node_num(g) - 1:
        return True
    else:
        return False

def get_tree_depth(g: nx.graph.Graph):
    assert not g.is_directed()
    assert get_is_tree(g)

    def dfs(u, dep, maxdep, vis):
        vis[u] = 1
        maxdep = max(maxdep, dep)

        for v in g[u]:
            if v not in vis:
                maxdep = max(maxdep, dfs(v, dep + 1, maxdep, vis))

        return maxdep

    # assert 0 is root
    return dfs(0, 0, 0, {})

if __name__ == '__main__':
    g = nx.graph.Graph()
    g.add_nodes_from([0,1,2,3,4])
    g.add_edges_from([(0,1),(1,2),(3,4)])
    assert get_node_num(g) == 5
    assert get_edge_num(g) == 3
    assert len(get_connected_component_sizes(g)) == 2
    g.add_edge(2,3)
    assert len(get_connected_component_sizes(g)) == 1
    assert get_degree(g) == [1,2,2,2,1]
    assert get_is_tree(g) == True
    assert get_tree_depth(g) == 4
