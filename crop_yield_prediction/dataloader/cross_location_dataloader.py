#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class CrossLocationDataset(Dataset):
    """
    Case 0 n_triplets_per_file == (max_index + 1): load numpy file in __init__, retrieve idx in __getitem__
    Case 1 n_triplets_per_file == 1: load numpy file for idx in __getitem__
    Case 2 n_triplets_per_file > 1: load numpy file that stores idx (and others) in __getitem__
    idx is the index in "current" train/validation/test set. global idx is the index in the whole data set.
    Indices in train/validation/test set need to be sequential.
    """
    def __init__(self, data_dir, global_index_dic, y, n_tsteps, max_index, n_triplets_per_file):
        self.data_dir = data_dir
        self.global_index_dic = global_index_dic
        self.n_triplets = len(global_index_dic)
        self.y = y
        self.n_tsteps = n_tsteps
        self.max_index = max_index
        if n_triplets_per_file == (max_index + 1):
            self.X_data = np.load('{}/0_{}.npy'.format(data_dir, max_index))
        assert n_triplets_per_file == 1

    def __len__(self):
        return self.n_triplets

    def __getitem__(self, idx):
        global_idx = self.global_index_dic[idx]
        X_idx = np.load('{}/{}.npy'.format(self.data_dir, global_idx))[0][:self.n_tsteps]
        y_idx = np.array(self.y[idx])

        return torch.from_numpy(X_idx).float(), torch.from_numpy(y_idx).float()


def cross_location_dataloader(data_dir, global_index_dic, y, n_tsteps, max_index, n_triplets_per_file,
                              batch_size=50, shuffle=True, num_workers=4):
    """
    img_type: 'landsat', 'rgb', or 'naip'
    augment: random flip and rotate for data augmentation
    shuffle: turn shuffle to False for producing embeddings that correspond to original tiles.
    Returns a DataLoader with either NAIP (RGB/IR), RGB, or Landsat tiles.
    """

    dataset = CrossLocationDataset(data_dir, global_index_dic, y, n_tsteps, max_index,
                                   n_triplets_per_file=n_triplets_per_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
