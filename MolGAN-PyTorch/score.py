import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), 'GraphGen'))

from recognize import *
import re
import numpy as np
from joblib import Parallel, delayed
from graph_data import SyntheticGraphDataset
    
def score(props, adj_mat_hat):
    def _get_score(prop, adj):
        closeness = 0
        score = 0
        n_prop = 0
        score_n = 0
        score_e = 0
        for i, true_prop in enumerate(prop):
            eval_fn = SyntheticGraphDataset._get_eval_str_fn()[i]
            if true_prop < 0:
                continue
            
            n_prop += 1
            pred_prop = eval_fn(adj)
            score += int(int(pred_prop) == int(true_prop))
            closeness += np.exp(-(int(pred_prop) - int(true_prop))**2)
            if i == 0:
                score_n += int(int(pred_prop) == int(true_prop))
            elif i == 1:
                score_e += int(int(pred_prop) == int(true_prop))
                
                
        return score / float(n_prop), closeness / n_prop, score_n, score_e

    score_close_n_e = Parallel(n_jobs=8)(delayed(_get_score)(prop, adj) for prop, adj in zip(props, adj_mat_hat))
    scores, closeness, scores_n, scores_e = zip(*score_close_n_e)
    return {
        'property_match': scores,
        'closeness': closeness,
        'n_match': scores_n,
        'm_match': scores_e
    }
