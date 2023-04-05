from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from transformers import BertTokenizer
from PIL import Image
import torch
import os
import random

from data.sparse_molecular_dataset import SparseMolecularDataset

class SparseMoleCular(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, data_dir):
        """Initialize and preprocess the CelebA dataset."""
        self.data = SparseMolecularDataset()
        self.data.load(data_dir)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        return index, self.data.data[index], self.data.smiles[index],\
               self.data.data_S[index], self.data.data_A[index],\
               self.data.data_X[index], self.data.data_D[index],\
               self.data.data_F[index], self.data.data_Le[index],\
               self.data.data_Lv[index]

    def __len__(self):
        """Return the number of images."""
        return len(self.data)


def get_loader(image_dir, batch_size, mode, num_workers=1):
    """Build and return a data loader."""

    dataset = SparseMoleCular(image_dir)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir):
        self.load(data_dir)
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 128, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def load(self, data_dir):
        with open(os.path.join(data_dir, 'graphs.pkl', 'rb') as f:
            self.adjMats = pickle.load(f)
        with open(os.path.join(data_dir, 'descs.pkl', 'rb') as f:
            self.descs = pickle.load(f)
        
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y