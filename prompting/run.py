import argparse 
import pickle
from dotenv import load_dotenv
load_dotenv('./.env')
import os
import time
import random
import numpy as np
from retry import retry
import threading
from joblib import Parallel, delayed, parallel_backend
from tqdm import trange

import sys
sys.path.insert(0, '../MolGAN-PyTorch')
import graph_data
from prompt import graph_prompt

@retry(tries=5, delay=30)
def chatgpt(idx, intro, examples, prompt, max_tokens, stop=["```end"]):
    messages = [
        {"role": "system", "content": intro},
    ]
    i = 0
    for (ob, act) in examples:
        if i == 1:
            continue
        messages += [
            {"role": "system", "name": "example_user", "content": ob},
            {"role": "system", "name": "example_assistant", "content": act},
        ]
        i += 1
    messages += [
        {"role": "user", "content": prompt}
    ]
    
    import openai
    openai.api_key = eval(os.getenv('OPENAI_API_KEYS'))[idx]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        n=1,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
    )
    
    return response["choices"][0]["message"]["content"]

def load_data(opt):
    with open(os.path.join(opt.data_dir, 'graphs.pkl'), 'rb') as f:
        adj_matrix = pickle.load(f)
    with open(os.path.join(opt.data_dir, 'properties.pkl'), 'rb') as f:
        properties = pickle.load(f)

    tmp = list(zip(adj_matrix, properties))
    np.random.shuffle(tmp)
    adj_matrix, properties = zip(*tmp)

    return adj_matrix, properties

def run(api_idx, idx, data, lock):
    g, _, _, text_desc, properties = data

    n = properties[0]
    msg = ""
    msg += f'idx = {idx}\n'
    msg += f'text_desc = {text_desc}\n'

    try:
        graph_str = chatgpt(
                idx=api_idx,
                intro=graph_prompt['intro'],
                examples=graph_prompt['examples'],
                prompt=graph_prompt['prompt'].format(description=text_desc),
                max_tokens=min(3500, int((n**2 + n + 1)*2))
                # n^2+n+1 + extra should be enough according to https://platform.openai.com/tokenizer
            )
        msg += f'raw prompt result = {graph_str}\n'
        graph_str = graph_str.split('```')[1].strip()
        msg += f'graph_str = {graph_str}\n'

        adj_matrix = list(map(lambda x: list(map(int, x.split())), graph_str.split('\n')))
        n = max(len(adj_matrix), max(map(len, adj_matrix)))
        adj_mat = np.zeros((n, n))
        for i, row in enumerate(adj_matrix):
            for j, val in enumerate(row):
                adj_mat[i][j] = val
        adj_mat = np.maximum(adj_mat, adj_mat.T)
        nodes = (adj_mat.sum(axis=1)!=0).astype(int)
        msg += f'adj_mat = {adj_mat}\n'
        
        pred = graph_data.SyntheticGraphDataset.get_prop(adj_mat, nodes, properties)
        msg += f'pred = {pred}\n'

        correct = [pred[i] == properties[i] for i in range(len(pred)) if properties[i] is not None]

        msg += f'properties = {properties}\n'
        msg += f'correct = {correct}\n'
        msg += f'score = {np.mean(correct)}\n'
        msg += '====================================='
    except Exception as e:
        msg += f'error = {e}\n'
        msg += f'score = 0\n'
        msg += '====================================='

    lock.acquire()
    print(msg, flush=True)
    lock.release()

def main(opt, skip=0):
    lock = threading.Lock()

    dataset = graph_data.SyntheticGraphDataset(
        data_dir=os.path.join(opt.data_dir, 'test'),
        max_node=50, 
        max_len=0, # we don't need text here
        model_name='bert-base-uncased' # dummy
    )

    n_jobs = len(eval(os.getenv('OPENAI_API_KEYS')))
    with parallel_backend('threading', n_jobs=n_jobs):
        Parallel()(delayed(run)(idx % n_jobs, idx, dataset[idx], lock) \
            for idx in trange(skip, len(dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--skip', type=int, default=0)
    opt = parser.parse_args()

    np.random.seed(0)
    random.seed(0)

    main(opt, skip=opt.skip)
