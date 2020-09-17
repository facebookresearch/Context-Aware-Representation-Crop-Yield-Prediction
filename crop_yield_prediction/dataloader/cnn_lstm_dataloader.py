#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class CnnLSTMDataset(Dataset):
    """
    Case 0 n_triplets_per_file == (max_index + 1): load numpy file in __init__, retrieve idx in __getitem__
    Case 1 n_triplets_per_file == 1: load numpy file for idx in __getitem__
    Case 2 n_triplets_per_file > 1: load numpy file that stores idx (and others) in __getitem__
    idx is the index in "current" train/validation/test set. global idx is the index in the whole data set.
    Indices in train/validation/test set need to be sequential.
    """
    def __init__(self, data_dir, start_index, end_index, y, n_tsteps, max_index, n_triplets_per_file):
        self.data_dir = data_dir
        self.start_index = start_index
        self.end_index = end_index
        self.n_triplets = end_index - start_index + 1
        self.n_triplets_per_file = n_triplets_per_file
        self.y = y
        self.n_tsteps = n_tsteps
        self.max_index = max_index
        if n_triplets_per_file == (max_index + 1):
            self.X_data = np.load('{}/0_{}.npy'.format(data_dir, max_index))

    def __len__(self):
        return self.n_triplets

    def __getitem__(self, idx):
        global_idx = idx + self.start_index

        if self.n_triplets_per_file == (self.max_index + 1):
            X_idx = self.X_data[global_idx][:self.n_tsteps]
        else:
            if self.n_triplets_per_file > 1:
                file_idx = global_idx // self.n_triplets_per_file
                local_idx = global_idx % self.n_triplets_per_file

                end_idx = min((file_idx+1)*self.n_triplets_per_file-1, self.max_index)
                X_idx = np.load('{}/{}_{}.npy'.format(self.data_dir,
                                                      file_idx * self.n_triplets_per_file,
                                                      end_idx))[local_idx][:self.n_tsteps]
            else:
                X_idx = np.load('{}/{}.npy'.format(self.data_dir, global_idx))[0][:self.n_tsteps]
        y_idx = np.array(self.y[idx])

        X_idx = X_idx[:, 0, :, :, :]

        return torch.from_numpy(X_idx).float(), torch.from_numpy(y_idx).float()


def cnn_lstm_dataloader(data_dir, start_index, end_index, y, n_tsteps, max_index, n_triplets_per_file,
                        batch_size=50, shuffle=True, num_workers=4):
    """
    img_type: 'landsat', 'rgb', or 'naip'
    augment: random flip and rotate for data augmentation
    shuffle: turn shuffle to False for producing embeddings that correspond to original tiles.
    Returns a DataLoader with either NAIP (RGB/IR), RGB, or Landsat tiles.
    """

    dataset = CnnLSTMDataset(data_dir, start_index, end_index, y, n_tsteps, max_index,
                         n_triplets_per_file=n_triplets_per_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
