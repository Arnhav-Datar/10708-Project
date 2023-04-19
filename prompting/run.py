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

def main(opt):
    from prompt import graph_prompt
    from collections import defaultdict

    perf_cnt = defaultdict(lambda: [])

    adj_matrixs, properties = load_data(opt)
    for _, propertie in zip(adj_matrixs, properties):
        if len(perf_cnt[propertie['n']]) >= 10:
            continue

        prop = [propertie['n'], propertie['m'], propertie['max_deg'], propertie['min_deg']]
        # prop += [propertie['cc_num'], propertie['max_diameter'], propertie['cycle']]
        template = ['{} nodes', '{} edges', 'max node degree {}', 'min node degree {}']
        # template += '{} connected components', 'maximum diameter {}', 'have cycle' if have_cycle else 'no cycle']
        eval_func = [
            lambda g: max(g.shape[0], g.shape[1]),
            lambda g: g.sum() // 2,
            lambda g: g.sum(axis=1).max(),
            lambda g: g.sum(axis=1).min(),
        ]
        # eval_func: TODO

        count = np.random.randint(4, 4+1)
        choice_idx = np.random.choice(np.arange(len(prop)), count, replace=False)
        description = 'Graph with ' + ', '.join([template[i].format(prop[i]) for i in choice_idx])
        print(description)

        try:
            graph_str = chatgpt(
                    intro=graph_prompt['intro'],
                    examples=graph_prompt['examples'],
                    prompt=graph_prompt['prompt'].format(description=description),
                    max_tokens=int((propertie['n']**2 + propertie['n'] + 5)*1.5)
                    # n^2+n+5 + extra should be enough according to https://platform.openai.com/tokenizer
                )
            graph_str = graph_str.split('```')[1]
            print('='*20)
            print(graph_str)
            print('='*20)
       
            adj_matrix = np.array(list(map(lambda x: list(map(int, x.split())), graph_str.split('\n'))))
            print(adj_matrix)
            print('='*20)
            
            pred = [eval_func[i](adj_matrix) for i in choice_idx]
            print(pred)
            print('='*20)

            correct = [eval_func[i](adj_matrix) == prop[i] for i in choice_idx]

            print(correct)
            print(np.mean(correct))
            print('='*20)

            # group by node count
            pref_cnt[propertie['n']].append((prop, choice_idx, description, graph_str, adj_matrix, pred, correct, np.mean(correct)))
        except Exception as e:
            print(e)
            perf_cnt[propertie['n']].append((prop, choice_idx, description, None, None, None, None, None))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    opt = parser.parse_args()

    np.random.seed(0)

    main(opt)
