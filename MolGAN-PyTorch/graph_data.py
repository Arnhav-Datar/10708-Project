import torch
from transformers import AutoTokenizer
from torch.utils import data
import pickle
import numpy as np
import os
import pickle

class SimpleSyntheticGraphDataset(data.Dataset):
    """Dataset Class for synthetic graph dataset."""

    def __init__(self, data_dir, max_node, max_len, model_name='bert-base-uncased'):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, 'graphs.pkl'), 'rb') as f:
            self.adj_matrix = pickle.load(f)
        with open(os.path.join(data_dir, 'properties.pkl'), 'rb') as f:
            self.properties = pickle.load(f)
        assert len(self.adj_matrix) == len(self.properties)

        for i in range(len(self.adj_matrix)):
            node_size = self.adj_matrix[i].shape[0]
            if node_size > max_node:
                raise Exception('Node size is larger than max_node')
            self.adj_matrix[i] = np.pad(self.adj_matrix[i], (0, max_node - node_size), 'constant', constant_values=0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_len = max_len
        self.max_node = max_node

    def _gen_text(self, property):
        n = property['n']
        m = property['m']
        text = f'Undirected graph with {n} nodes and {m} edges.'
        return text

    def _encode_text(self, text):
        return self.tokenizer(text, add_special_tokens=True, truncation=False, max_length=self.max_len, padding='max_length')

    def __getitem__(self, index):
        adj_matrix = self.adj_matrix[index]
        test_desc = self._gen_text(self.properties[index])
        encoded_text = self._encode_text(test_desc)

        return adj_matrix, encoded_text, test_desc
    
    def __len__(self):
        return len(self.adj_matrix)

    @staticmethod
    def collate_fn(batch):
        adj_matrix = torch.from_numpy(np.stack([item[0] for item in batch])).type(torch.FloatTensor)
        ids = torch.from_numpy(np.stack([item[1].input_ids for item in batch]))
        attention_mask = torch.from_numpy(np.stack([item[1].attention_mask for item in batch]))
        desc = [item[2] for item in batch]
        # tokens = [item[1].tokens for item in batch]
        return adj_matrix, ids, attention_mask, desc

def get_loaders(data_dir, max_node, max_len, model_name, batch_size, num_workers=1):
    """Build and return a data loader."""

    dataset = SimpleSyntheticGraphDataset(data_dir, max_node, max_len, model_name)
    train, val, test = torch.utils.data.random_split(dataset, [0.65, 0.15, 0.2])
    train_loader = data.DataLoader(dataset=train,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   collate_fn=SimpleSyntheticGraphDataset.collate_fn)
    val_loader = data.DataLoader(dataset=val,
                                 batch_size=batch_size*2,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=SimpleSyntheticGraphDataset.collate_fn)
    test_loader = data.DataLoader(dataset=test,
                                  batch_size=batch_size*2,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=SimpleSyntheticGraphDataset.collate_fn)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    ds = SimpleSyntheticGraphDataset('./data/graphgen', 50, 128)
    print('-'*10, 'test dataset', '-'*10)
    print(ds[0])
    print('-'*10, 'test dataloader', '-'*10)
    print('len', len(ds))
    dl = data.DataLoader(dataset=ds, batch_size=4, shuffle=True, num_workers=1
                         , collate_fn=SimpleSyntheticGraphDataset.collate_fn)
    print('len', len(dl)) 
    batch = next(iter(dl))
    print('adj_matrix', batch[0].shape)
    print('ids', batch[1].shape)
    print('attention_mask', batch[2].shape)
    print('tokens', batch[3])