#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import pandas as pd
import argparse
import torch.optim as optim
from pathlib import Path
import sys

sys.path.append("..")

from crop_yield_prediction.models.cnn_lstm import CnnLstm
from crop_yield_prediction.train_cnn_lstm import train_cnn_lstm
from crop_yield_prediction.utils import plot_predict
from crop_yield_prediction.utils import plot_predict_error
from crop_yield_prediction.utils import output_to_csv_simple
from crop_yield_prediction.train_cnn_lstm import eval_test


def crop_yield_train_cnn_lstm(args, data_dir, model_out_dir, result_out_dir, log_out_dir, start_year, end_year,
                         n_tsteps, train_years=None):
    batch_size = 64
    test_batch_size = 128
    n_triplets_per_file = 1
    epochs = 50
    n_experiment = 2

    patience = args.patience if args.patience != 9999 else None
    feature = args.feature
    feature_len = args.feature_len
    tilenet_zdim = args.tilenet_zdim
    lstm_inner = args.lstm_inner

    params = '{}_nt{}_es{}_{}_tyear{}_zdim{}_din{}'.format(start_year, n_tsteps, patience, feature, train_years, tilenet_zdim, lstm_inner)

    os.makedirs(log_out_dir, exist_ok=True)
    param_model_out_dir = '{}/{}'.format(model_out_dir, params)
    os.makedirs(param_model_out_dir, exist_ok=True)
    param_result_out_dir = '{}/{}'.format(result_out_dir, params)
    os.makedirs(param_result_out_dir, exist_ok=True)

    if feature == 'all':
        X_dir = '{}/nr_25_dr100'.format(data_dir)
    else:
        X_dir = '{}/nr_25_dr100_{}'.format(data_dir, feature)

    dim_y = pd.read_csv('{}/dim_y.csv'.format(data_dir))
    dim_y = dim_y.astype({'state': int, 'county': int, 'year': int, 'value': float, 'lat': float, 'lon': float})
    max_index = len(dim_y) - 1

    results = dict()
    for year in range(start_year, end_year + 1):
        print('Predict year {}......'.format(year))

        test_idx = (dim_y['year'] == year)
        valid_idx = (dim_y['year'] == (year - 1))
        if train_years is None:
            train_idx = (dim_y['year'] < (year - 1))
        else:
            train_idx = (dim_y['year'] < (year - 1)) & (dim_y['year'] >= (year - 1 - train_years))

        y_valid, y_train = np.array(dim_y.loc[valid_idx]['value']), np.array(dim_y.loc[train_idx]['value'])
        y_test, dim_test = np.array(dim_y.loc[test_idx]['value']), np.array(dim_y.loc[test_idx][['state', 'county']])

        test_indices = [i for i, x in enumerate(test_idx) if x]
        valid_indices = [i for i, x in enumerate(valid_idx) if x]
        train_indices = [i for i, x in enumerate(train_idx) if x]

        # check if the indices are sequential
        assert all(elem == 1 for elem in [y - x for x, y in zip(test_indices[:-1], test_indices[1:])])
        assert all(elem == 1 for elem in [y - x for x, y in zip(valid_indices[:-1], valid_indices[1:])])
        assert all(elem == 1 for elem in [y - x for x, y in zip(train_indices[:-1], train_indices[1:])])
        print('Train size {}, valid size {}, test size {}'.format(y_train.shape[0], y_valid.shape[0], y_test.shape[0]))

        test_corr_lis, test_r2_lis, test_rmse_lis = [], [], []
        test_prediction_lis = []
        for i in range(n_experiment):
            print('Experiment {}'.format(i))

            cnn_lstm = CnnLstm(tn_in_channels=feature_len,
                               tn_z_dim=tilenet_zdim,
                               d_model=tilenet_zdim,
                               d_inner=lstm_inner)

            optimizer = optim.Adam(cnn_lstm.parameters(), lr=0.001)

            trained_epochs = train_cnn_lstm(model=cnn_lstm,
                                            X_dir=X_dir,
                                            X_train_indices=(train_indices[0], train_indices[-1]),
                                            y_train=y_train,
                                            X_valid_indices=(valid_indices[0], valid_indices[-1]),
                                            y_valid=y_valid,
                                            X_test_indices=(test_indices[0], test_indices[-1]),
                                            y_test=y_test,
                                            n_tsteps=n_tsteps,
                                            max_index=max_index,
                                            n_triplets_per_file=n_triplets_per_file,
                                            patience=patience,
                                            optimizer=optimizer,
                                            batch_size=batch_size,
                                            test_batch_size=test_batch_size,
                                            n_epochs=epochs,
                                            out_dir=param_model_out_dir,
                                            year=year,
                                            exp_idx=i,
                                            log_file='{}/{}.txt'.format(log_out_dir, params))

            test_prediction, rmse, r2, corr = eval_test(X_dir,
                                                        X_test_indices=(test_indices[0], test_indices[-1]),
                                                        y_test=y_test,
                                                        n_tsteps=n_tsteps,
                                                        max_index=max_index,
                                                        n_triplets_per_file=n_triplets_per_file,
                                                        batch_size=test_batch_size,
                                                        model_dir=param_model_out_dir,
                                                        model=cnn_lstm,
                                                        epochs=trained_epochs,
                                                        year=year,
                                                        exp_idx=i,
                                                        log_file='{}/{}.txt'.format(log_out_dir, params))
            test_corr_lis.append(corr)
            test_r2_lis.append(r2)
            test_rmse_lis.append(rmse)

            test_prediction_lis.append(test_prediction)

        test_prediction = np.mean(np.asarray(test_prediction_lis), axis=0)
        np.save('{}/{}.npy'.format(param_result_out_dir, year), test_prediction)
        plot_predict(test_prediction, dim_test, Path('{}/pred_{}.html'.format(param_result_out_dir, year)))
        plot_predict_error(test_prediction, y_test, dim_test, Path('{}/err_{}.html'.format(param_result_out_dir, year)))

        results[year] = {'test_rmse': np.around(np.mean(test_rmse_lis), 3),
                         'test_r2': np.around(np.mean(test_r2_lis), 3),
                         'test_corr': np.around(np.mean(test_corr_lis), 3)}

        output_to_csv_simple(results, param_result_out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop Yield Train CNN_LSTM')
    parser.add_argument('--patience', type=int, default=9999, metavar='PATIENCE')
    parser.add_argument('--feature', type=str, default='all', metavar='FEATURE')
    parser.add_argument('--feature-len', type=int, default=9, metavar='FEATURE_LEN')
    parser.add_argument('--year', type=int, default=2014, metavar='YEAR')
    parser.add_argument('--ntsteps', type=int, default=7, metavar='NTSTEPS', required=True)
    parser.add_argument('--train-years', type=int, default=None, metavar='TRAINYEAR', required=True)
    parser.add_argument('--tilenet-zdim', type=int, default=256, metavar='ZDIM')
    parser.add_argument('--lstm-inner', type=int, default=512, metavar='LSTM_INNER')

    args = parser.parse_args()

    crop_yield_train_cnn_lstm(args,
                              data_dir='data/spatial_temporal/counties',
                              model_out_dir='results/cnn_lstm/models',
                              result_out_dir='results/cnn_lstm/results',
                              log_out_dir='results/cnn_lstm/prediction_logs',
                              start_year=args.year,
                              end_year=args.year,
                              n_tsteps=args.ntsteps,
                              train_years=args.train_years)
