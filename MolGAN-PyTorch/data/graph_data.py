import pickle
import numpy as np
import os

class RandomGraphDataset:
    def __init__(self):
        pass
    
    def load(self, data_dir):
        with open(os.path.join(data_dir, 'graphs.pkl', 'rb') as f:
            self.data = pickle.load(f)