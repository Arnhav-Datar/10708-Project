import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), 'GraphGen'))

from recognize import *
import re
import numpy as np

def process_scores(scores):
    results = []
    # convert dictionary of lists to numpy 2D array
    for key in scores:
        results.append(np.array(scores[key]))
    results = np.array(results).T
    proc_res = np.clip(results, 0, 1)
    return {
        'property_match': np.sum(proc_res, axis=1) / np.sum(results != -1, axis=1)
    }
    
def score(descs, adj_mat_hat):
    scores = {
        'nodes': [],
        'edges': [],
        'max_diameter': [],
        'cc_num': [],
        'max_deg': [],
        'min_deg': [],
        'have_cycle': [],
    }
    for desc, adj in zip(descs, adj_mat_hat):
        m = re.search(r'\b([0-9]+)\s+nodes', desc, re.IGNORECASE)
        nodes = get_node_num(adj)
        if m is not None:
            n = int(m.group(1))
            scores['nodes'].append(int(n == nodes))
        else:
            scores['nodes'].append(-1)
        
        edg = get_edge_num(adj)
        m = re.search(r'\b([0-9]+)\s+edges', desc, re.IGNORECASE)
        if m is not None:
            n = int(m.group(1))
            scores['edges'].append(int(n == edg))
        else:
            scores['edges'].append(-1)
        
        m = re.search(r'max diameter\s+([0-9]+)\b', desc, re.IGNORECASE)
        if m is not None:
            n = int(m.group(1))
            scores['max_diameter'].append(int(n == get_max_diameter(adj)))
        else:
            scores['max_diameter'].append(-1)
        
        cc_num = get_connected_component_num(adj)
        degree_seq = get_degree_seq(adj)

        have_cycle = edg > nodes - cc_num
        max_deg = np.max(degree_seq)
        min_deg = np.min(degree_seq)
        
        m = re.search(r'\b([0-9]+)\s+connected components', desc, re.IGNORECASE)
        if m is not None:
            n = int(m.group(1))
            scores['cc_num'].append(int(n == cc_num))
        else:
            scores['cc_num'].append(-1)
        
        m = re.search(r'min degree\s+([0-9]+)\b', desc, re.IGNORECASE)
        if m is not None:
            n = int(m.group(1))
            scores['min_deg'].append(int(n == min_deg))
        else:
            scores['min_deg'].append(-1)
        
        m = re.search(r'max degree\s+([0-9]+)\b', desc, re.IGNORECASE)
        if m is not None:
            n = int(m.group(1))
            scores['max_deg'].append(int(n == max_deg))
        else:
            scores['max_deg'].append(-1)
        
        m = re.search(r'\b([0-9]+)\s+cycle', desc, re.IGNORECASE)
        if m is not None:
            n = (str(m.group(1)) == 'has')
            scores['have_cycle'].append(int(n == have_cycle))
        else:
            scores['have_cycle'].append(-1)
    
    return process_scores(scores)
                