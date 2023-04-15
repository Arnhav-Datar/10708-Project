import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), 'GraphGen'))

from recognize import *
import re
import numpy as np
from joblib import Parallel, delayed

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
    
def score(props, adj_mat_hat):
    def _get_score(prop, adj):
        closeness = 0
        score = 0
        n_prop = 0
        for key, val in prop.items():
            n = get_node_num(adj)
            m = get_edge_num(adj)
            if key == 'n' and val is not None:
                n_prop += 1
                score += int(n == val)
                closeness += np.exp(-(n - val)**2)
            elif key == 'm' and val is not None:
                n_prop += 1
                score += int(m == val)
                closeness += np.exp(-(m - val)**2)
            elif key == 'max_diameter' and val is not None:
                n_prop += 1
                md = get_max_diameter(adj)
                score += int(md == val)
                closeness += np.exp(-(md - val)**2)
            else:
                cc_num = get_connected_component_num(adj)
                degree_seq = get_degree_seq(adj)

                have_cycle = edg > nodes - cc_num
                max_deg = np.max(degree_seq)
                min_deg = np.min(degree_seq)
                if key == 'cc_num' and val is not None:
                    n_prop += 1
                    score += int(cc_num == val)
                    closeness += np.exp(-(cc_num - val)**2)
                elif key == 'max_deg' and val is not None:
                    n_prop += 1
                    score += int(max_deg == val)
                    closeness += np.exp(-(max_deg - val)**2)
                elif key == 'min_deg' and val is not None:
                    n_prop += 1
                    score += int(min_deg == val)
                    closeness += np.exp(-(min_deg - val)**2)
                elif key == 'have_cycle' and val is not None:
                    n_prop += 1
                    score += int(have_cycle == val)
                    closeness += int(have_cycle == val)
                
        return score / n_prop, closeness / n_prop

    score_close = Parallel(n_jobs=8)(delayed(_get_score)(prop, adj) for prop, adj in zip(props, adj_mat_hat))
    scores, closeness = zip(*score_close)
    return {
        'property_match': scores,
        'closeness': closeness
    }
