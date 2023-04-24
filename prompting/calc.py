import argparse
import numpy as np

from collections import defaultdict

def analyze(score, properties, pred, error, idxs):
    keys = ['node', 'edge', 'min deg', 'max deg', 'max diameter', 'cc num', 'cycle']
    score_, matches = [], defaultdict(lambda: [])
    for idx in idxs:
        if idx in error:
            score_.append(0)
            for i, key in enumerate(keys):
                if properties[idx][i] is not None:
                    matches[key].append(0)
        else:
            score_.append(score[idx])
            for i, key in enumerate(keys):
                if properties[idx][i] is not None:
                    matches[key].append(1 if properties[idx][i] == pred[idx][i] else 0)

    print(f'mean score of {len(score_)} graphs = {np.mean(score_)}')
    for key in keys:
        if len(matches[key]) > 0:
            print(f'{key} match = {np.mean(matches[key])}')
    print()

def main(opt):
    score = {}
    properties = {}
    pred = {}
    error = {}
    with open(opt.log_file, 'r') as f:
        idx = None
        for line in f:
            if 'idx = ' in line:
                idx = int(line.split(' = ')[1])
            elif 'properties = ' in line:
                properties[idx] = eval(line.split(' = ')[1])
            elif 'pred = ' in line:
                pred[idx] = eval(line.split(' = ')[1])
            elif 'error = ' in line:
                error[idx] = line.split(' = ')[1]
            elif 'score = ' in line:
                score[idx] = float(line.split(' = ')[1])

    print('missing idx')
    for idx in range(opt.max_idx):
        if idx not in score:
            print(idx, end=',')
    print()
    analyze(score, properties, pred, error, range(len(score)))

    bins = [0, 5, 10, 25, 50]
    for i in range(len(bins)-1):
        l, r = bins[i]+1, bins[i+1]
        idx_subset = [idx for idx in score if l <= properties[idx][0] <= r]
        print(f'graphs with {l} <= n <= {r}')
        analyze(score, properties, pred, error, idx_subset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_idx', type=int, default=500)
    parser.add_argument('--log_file', type=str, required=True)
    opt = parser.parse_args()
    main(opt)