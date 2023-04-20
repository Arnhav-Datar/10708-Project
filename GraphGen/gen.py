import pickle
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from tqdm import trange, tqdm
from recognize import *

tot = 100000
num_features = [2,3,4]
params = {
    'scale_free_graph': {
        'num': int(tot * 0.3),
        'n': [5, 10, 25, 50]
    },
    'erdos_renyi_graph': {
        'num': int(tot * 0.3),
        'np': [(5, 0.3), (10, 0.2), (25, 0.1), (50, 0.05)]
    },
    'random_geometric_graph': {
        'num': int(tot * 0.3),
        'nr': [(5, 0.5), (10, 0.4), (25, 0.2), (50, 0.2)]
    },
    'random_tree': {
        'num': int(tot * 0.1),
        'n': [5, 10, 25, 50]
    }
}

def nxgraph_to_adj_matrix(g: nx.graph.Graph) -> np.ndarray:
    adj = nx.adjacency_matrix(g).todense()
    # remove multi-edges
    adj[adj > 0] = 1

    # remove self-loop
    for i in range(adj.shape[0]):
        adj[i, i] = 0

    # make sure the graph is undirected
    adj = np.maximum(adj, adj.T)

    return adj

def gen_scale_free_graph(count: int) -> [np.ndarray]:
    def _gen():
        n = np.random.choice(params['scale_free_graph']['n'])
        g = nx.scale_free_graph(n)
    
        return nxgraph_to_adj_matrix(g)

    adjs = Parallel(n_jobs=8)(delayed(_gen)() for _ in tqdm(range(count)))
    
    return adjs

def gen_erdos_renyi_graph(count: int) -> [np.ndarray]:
    def _gen():
        idx = np.random.choice(len(params['erdos_renyi_graph']['np']))
        n, p = params['erdos_renyi_graph']['np'][idx]
        g = nx.erdos_renyi_graph(n, p)
        
        return nxgraph_to_adj_matrix(g)

    adjs = Parallel(n_jobs=8)(delayed(_gen)() for _ in tqdm(range(count)))
    
    return adjs

def gen_random_geometric_graph(count: int) -> [np.ndarray]:
    def _gen():
        idx = np.random.choice(len(params['random_geometric_graph']['nr']))
        n, r = params['random_geometric_graph']['nr'][idx]
        g = nx.random_geometric_graph(n, r)
        
        return nxgraph_to_adj_matrix(g)

    adjs = Parallel(n_jobs=8)(delayed(_gen)() for _ in tqdm(range(count)))
    
    return adjs

def gen_random_tree(count: int) -> [np.ndarray]:
    def _gen():
        n = np.random.choice(params['random_tree']['n'])
        g = nx.random_tree(n)

        return nxgraph_to_adj_matrix(g)

    adjs = Parallel(n_jobs=8)(delayed(_gen)() for _ in tqdm(range(count)))
    
    return adjs

def gen_properties(adjs: [np.ndarray]) -> [dict[str, int]]:
    def _gen(adj):
        n = get_node_num(adj)
        m = get_edge_num(adj)
        max_diameter = get_max_diameter(adj)
        cc_num = get_connected_component_num(adj)
        degree_seq = get_degree_seq(adj)

        have_cycle = m > n - cc_num
        max_deg = np.max(degree_seq)
        min_deg = np.min(degree_seq)

        properties = {
            'n': n,
            'm': m,
            'max_diameter': max_diameter,
            'cc_num': cc_num,
            'max_deg': max_deg,
            'min_deg': min_deg,
            'cycle': have_cycle
        }

        return properties
    
    props = Parallel(n_jobs=8)(delayed(_gen)(adj) for adj in tqdm(adjs))
    
    return props

def main():
    adjs = []
    adjs.extend(gen_scale_free_graph(params['scale_free_graph']['num']))
    adjs.extend(gen_erdos_renyi_graph(params['erdos_renyi_graph']['num']))
    adjs.extend(gen_random_geometric_graph(params['random_geometric_graph']['num']))
    adjs.extend(gen_random_tree(params['random_tree']['num']))

    np.random.shuffle(adjs)
    properties = gen_properties(adjs)

    with open('graphs.pkl', 'wb') as f:
        pickle.dump(adjs, f)
    with open('properties.pkl', 'wb') as f:
        pickle.dump(properties, f)
    # with open('descs.pkl', 'wb') as f:
    #     pickle.dump(descs, f)

if __name__ == '__main__':
    np.random.seed(1)
    main()