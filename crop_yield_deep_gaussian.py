#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from crop_yield_prediction.models.deep_gaussian_process import *

from pathlib import Path
import torch
import argparse


def train_cnn_gp(times, train_years, dropout=0.5, dense_features=None,
                 pred_years=range(2014, 2019), num_runs=2, train_steps=25000,
                 batch_size=32, starter_learning_rate=1e-3, weight_decay=0, l1_weight=0,
                 patience=None, use_gp=True, sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    histogram_path = Path('data/deep_gaussian/data.npz')
    savedir = Path('results/deep_gaussian/nt{}_tyear{}_cnn'.format(times[0], train_years))

    model = ConvModel(in_channels=9, dropout=dropout, dense_features=dense_features,
                      savedir=savedir, use_gp=use_gp, sigma=sigma, r_loc=r_loc,
                      r_year=r_year, sigma_e=sigma_e, sigma_b=sigma_b, device=device)
    model.run(times, train_years, histogram_path, pred_years, num_runs, train_steps, batch_size,
              starter_learning_rate, weight_decay, l1_weight, patience)


def train_rnn_gp(times, train_years, num_bins=32, hidden_size=128,
                 rnn_dropout=0.75, dense_features=None, pred_years=range(2014, 2019),
                 num_runs=2, train_steps=10000, batch_size=32, starter_learning_rate=1e-3, weight_decay=0,
                 l1_weight=0, patience=None, use_gp=True, sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    histogram_path = Path('data/deep_gaussian/data.npz')
    savedir = Path('results/deep_gaussian/nt{}_tyear{}_rnn'.format(times[0], train_years))

    model = RNNModel(in_channels=9, num_bins=num_bins, hidden_size=hidden_size,
                     rnn_dropout=rnn_dropout, dense_features=dense_features,
                     savedir=savedir, use_gp=use_gp, sigma=sigma, r_loc=r_loc, r_year=r_year,
                     sigma_e=sigma_e, sigma_b=sigma_b, device=device)
    model.run(times, train_years, histogram_path, pred_years, num_runs, train_steps, batch_size,
              starter_learning_rate, weight_decay, l1_weight, patience)


if __name__ == '__main__':
    get_features_for_deep_gaussian()

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str)
    parser.add_argument('--time', type=int, default=None, metavar='TIME', required=True)
    parser.add_argument('--train-years', type=int, default=None, metavar='TRAINYEAR', required=True)

    args = parser.parse_args()
    model_type = args.type
    times = [args.time]
    train_years = args.train_years

    if model_type == 'cnn':
        train_cnn_gp(times, train_years)
    elif model_type == 'rnn':
        train_rnn_gp(times, train_years)
