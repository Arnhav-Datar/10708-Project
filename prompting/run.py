import argparse 
import pickle
from dotenv import load_dotenv
load_dotenv('./.env')
import os
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')
import time
import numpy as np
from retry import retry

import sys
sys.path.insert(0, '../MolGAN-PyTorch')
import graph_data

@retry(tries=3, delay=30)
def chatgpt(intro, examples, prompt, max_tokens, stop=["\n\n"]):
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

def main(opt, idxs):
    from prompt import graph_prompt
    from collections import defaultdict

    perf_cnt = defaultdict(lambda: [])

    dataset = graph_data.SyntheticGraphDataset(
        data_dir=opt.data_dir, 
        max_node=50, 
        max_len=0, # we don't need text here
        model_name='bert-base-uncased' # dummy
    )

    for idx in idxs:
        adj_matrix, _, text_desc, properties = dataset[idx]
        n = properties[0]
        adj_matrix = adj_matrix[:n, :n]
        print(f'idx = {idx}')
        print(f'text_desc = {text_desc}')
        print(f'properties = {properties}')

        try:
            graph_str = chatgpt(
                    intro=graph_prompt['intro'],
                    examples=graph_prompt['examples'],
                    prompt=graph_prompt['prompt'].format(description=text_desc),
                    max_tokens=min(3900, int((n**2 + n + 1)*2))
                    # n^2+n+1 + extra should be enough according to https://platform.openai.com/tokenizer
                )
            graph_str = graph_str.split('```')[1].strip()
            print(f'graph_str = {graph_str}')
    
            adj_matrix = list(map(lambda x: list(map(int, x.split())), graph_str.split('\n')))
            adj_mat = np.zeros((n, n))
            for i, row in enumerate(adj_matrix):
                for j, val in enumerate(row):
                    adj_mat[i][j] = val
            adj_mat = np.maximum(adj_mat, adj_mat.T)
            print(f'adj_mat = {adj_mat}')
            
            pred = graph_data.SyntheticGraphDataset.get_prop(adj_mat, properties)
            print(f'pred = {pred}')

            correct = [pred[i] == properties[i] for i in range(len(pred)) if properties[i] is not None]

            print(f'correct = {correct}')
            print(f'score = {np.mean(correct)}')
            print('=====================================', flush=True)

            # group by node count
            perf_cnt[n].append((text_desc, properties, graph_str, adj_matrix, pred, correct, np.mean(correct)))
        except Exception as e:
            print(f'error = {e}')
            print(f'score = 0')
            print('=====================================', flush=True)
            perf_cnt[n].append((text_desc, properties, None, None, None, None, None))

        # input('Press Enter to continue...')

def select_index(opt):
    dataset = graph_data.SyntheticGraphDataset(
        data_dir=opt.data_dir,
        max_node=50,
        max_len=0, # we don't need text here
        model_name='bert-base-uncased' # dummy
    )

    from collections import defaultdict
    idxs = defaultdict(lambda: [])
    for i in range(len(dataset)):
        _, _, _, properties = dataset[i]
        n = properties[0]
        idxs[n].append(i)
        
    idxs = {n: np.random.choice(idxs[n], size=25, replace=False) for n in idxs}

    return idxs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--node_num', type=int, required=True)
    opt = parser.parse_args()

    np.random.seed(0)

    # idxs = select_index(opt)
    # with open('idxs.pkl', 'wb') as f:
    #     pickle.dump(idxs, f)
    with open('idxs.pkl', 'rb') as f:
        idxs = pickle.load(f)
    main(opt, idxs[opt.node_num])
