"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, opt, datasets):
        self.dataset_list = opt["modes"]
        self.datasets = datasets
        self.total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[0:-1]])
        self.length = max([len(ds) for ds in self.datasets])
        
        self.partition = opt['ratio']
        self.partition = np.array(self.partition).cumsum()
        # print(self.partition)

    def __getitem__(self, index):
        while True:
            p = np.random.rand()
            for i in range(len(self.dataset_list)):    
                if p <= self.partition[i]:
                    a = self.datasets[i][index % len(self.datasets[i])]
                    if a == None:
                        continue
                    return a

    def __len__(self):
        return self.total_length