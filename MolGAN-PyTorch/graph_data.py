import torch
from transformers import AutoTokenizer
from torch.utils import data
import pickle
import numpy as np
import os
import pickle

class SyntheticGraphDataset(data.Dataset):
    """Dataset Class for synthetic graph dataset."""

    def __init__(self, data_dir, max_node, max_len, model_name='bert-base-uncased'):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, 'graphs.pkl'), 'rb') as f:
            self.adj_matrix = pickle.load(f)
        with open(os.path.join(data_dir, 'descs.pkl'), 'rb') as f:
            self.desc = pickle.load(f)
        assert len(self.adj_matrix) == len(self.desc)

        for i in range(len(self.adj_matrix)):
            node_size = self.adj_matrix[i].shape[0]
            if node_size > max_node:
                raise Exception('Node size is larger than max_node')
            self.adj_matrix[i] = np.pad(self.adj_matrix[i], (0, max_node - node_size), 'constant', constant_values=0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoded_text = self.tokenizer(self.desc, add_special_tokens=True, truncation=False, max_length=max_len, padding='max_length')

    def __getitem__(self, index):
        return self.adj_matrix[index], self.encoded_text[index]
    
    def __len__(self):
        return len(self.adj_matrix)

    @staticmethod
    def collate_fn(batch):
        adj_matrix = torch.from_numpy(np.stack([item[0] for item in batch]))
        ids = torch.from_numpy(np.stack([item[1].ids for item in batch]))
        attention_mask = torch.from_numpy(np.stack([item[1].attention_mask for item in batch]))
        tokens = [item[1].tokens for item in batch]
        return adj_matrix, ids, attention_mask, tokens

def get_loaders(data_dir, max_node, max_len, model_name, batch_size, num_workers=1):
    """Build and return a data loader."""

    dataset = SyntheticGraphDataset(data_dir, max_node, max_len, model_name)
    train, val, test = torch.utils.data.random_split(dataset, [0.65, 0.15, 0.2])
    train_loader = data.DataLoader(dataset=train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=SyntheticGraphDataset.collate_fn)
    val_loader = data.DataLoader(dataset=val,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=SyntheticGraphDataset.collate_fn)
    test_loader = data.DataLoader(dataset=test,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=SyntheticGraphDataset.collate_fn)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    ds = SyntheticGraphDataset('./data', 50, 128)
    print('-'*10, 'test dataset', '-'*10)
    print(ds[0])
    print('-'*10, 'test dataloader', '-'*10)
    dl = data.DataLoader(dataset=ds, batch_size=4, shuffle=True, num_workers=1
                         , collate_fn=SyntheticGraphDataset.collate_fn)
    batch = next(iter(dl))
    print('adj_matrix', batch[0].shape)
    print('ids', batch[1].shape)
    print('attention_mask', batch[2].shape)
    print('tokens', batch[3])