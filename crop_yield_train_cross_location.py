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

from crop_yield_prediction import CLIMATE_VARS
from crop_yield_prediction.models.semi_transformer import SemiTransformer
from crop_yield_prediction.train_cross_location import train_attention
from crop_yield_prediction.utils import plot_predict
from crop_yield_prediction.utils import plot_predict_error
from crop_yield_prediction.utils import output_to_csv_simple
from crop_yield_prediction.train_cross_location import eval_test


def crop_yield_train_cross_location(args, data_dir, model_out_dir, result_out_dir, log_out_dir, start_year, end_year,
                                    n_tsteps, train_years=None):
    batch_size = 64
    test_batch_size = 128
    n_triplets_per_file = 1
    epochs = 50
    attention_nhead = 8
    adam_lr = 0.001
    adam_betas = (0.9, 0.999)
    n_experiment = 2

    neighborhood_radius = args.neighborhood_radius
    distant_radius = args.distant_radius
    weight_decay = args.weight_decay
    tilenet_margin = args.tilenet_margin
    tilenet_l2 = args.tilenet_l2
    tilenet_ltn = args.tilenet_ltn
    tilenet_zdim = args.tilenet_zdim
    attention_layer = args.attention_layer
    attention_dff = args.attention_dff
    sentence_embedding = args.sentence_embedding
    dropout = args.dropout
    unsup_weight = args.unsup_weight
    patience = args.patience if args.patience != 9999 else None
    feature = args.feature
    feature_len = args.feature_len
    query_type = args.query_type

    assert tilenet_zdim % attention_nhead == 0

    if feature == 'all':
        X_dir = '{}/nr_{}'.format(data_dir, neighborhood_radius) if distant_radius is None else \
            '{}/nr_{}_dr{}'.format(data_dir, neighborhood_radius, distant_radius)
    else:
        X_dir = '{}/nr_{}_{}'.format(data_dir, neighborhood_radius, feature) if distant_radius is None else \
            '{}/nr_{}_dr{}_{}'.format(data_dir, neighborhood_radius, distant_radius, feature)

    dim_y = pd.read_csv('{}/dim_y.csv'.format(data_dir))
    dim_y = dim_y.astype({'state': int, 'county': int, 'year': int, 'value': float, 'lat': float, 'lon': float})
    max_index = len(dim_y) - 1

    results = dict()

    group_indices = {'1': (27, 38, 46, 55), '2': (31, 20, 19, 29), '3': (17, 18, 39, 21)}
    # group_indices = {'1': (38, 46, 31, 20, 19), '2': (55, 17, 18, 39, 26)}
    for year in range(start_year, end_year + 1):
        for test_group in group_indices.keys():
            print('Predict year {}......'.format(year))

            test_idx = (dim_y['year'] == year) & (dim_y['state'].isin(group_indices[test_group]))
            valid_idx = (dim_y['year'] == (year - 1)) & (dim_y['state'].isin(group_indices[test_group]))
            for train_group in group_indices.keys():
                params = 'train{}_test{}_{}_nt{}_nr{}_dr{}_wd{}_mar{}_l2{}_ltn{}_zd{}_al{}_adff{}_se{}_dr{}_uw{}_es{}_{}_tyear{}_qt{}'.format(
                    train_group,
                    test_group,
                    start_year,
                    n_tsteps,
                    neighborhood_radius,
                    distant_radius,
                    weight_decay,
                    tilenet_margin, tilenet_l2,
                    tilenet_ltn, tilenet_zdim,
                    attention_layer, attention_dff,
                    sentence_embedding, dropout,
                    unsup_weight, patience, feature,
                    train_years, query_type)

                os.makedirs(log_out_dir, exist_ok=True)
                param_model_out_dir = '{}/{}'.format(model_out_dir, params)
                os.makedirs(param_model_out_dir, exist_ok=True)
                param_result_out_dir = '{}/{}'.format(result_out_dir, params)
                os.makedirs(param_result_out_dir, exist_ok=True)

                if train_years is None:
                    train_idx = (dim_y['year'] < (year - 1)) & (dim_y['state'].isin(group_indices[train_group]))
                else:
                    train_idx = (dim_y['year'] < (year - 1)) & (dim_y['year'] >= (year - 1 - train_years)) \
                                & (dim_y['state'].isin(group_indices[train_group]))

                y_valid, y_train = np.array(dim_y.loc[valid_idx]['value']), np.array(dim_y.loc[train_idx]['value'])
                y_test, dim_test = np.array(dim_y.loc[test_idx]['value']), np.array(
                    dim_y.loc[test_idx][['state', 'county']])

                test_indices = [i for i, x in enumerate(test_idx) if x]
                valid_indices = [i for i, x in enumerate(valid_idx) if x]
                train_indices = [i for i, x in enumerate(train_idx) if x]

                train_size, valid_size, test_size = y_train.shape[0], y_valid.shape[0], y_test.shape[0]
                train_indices_dic = {local_index: global_index for local_index, global_index in
                                     zip(range(train_size), train_indices)}
                valid_indices_dic = {local_index: global_index for local_index, global_index in
                                     zip(range(valid_size), valid_indices)}
                test_indices_dic = {local_index: global_index for local_index, global_index in
                                    zip(range(test_size), test_indices)}
                print('Train size {}, valid size {}, test size {}'.format(train_size, valid_size, test_size))

                test_corr_lis, test_r2_lis, test_rmse_lis = [], [], []
                test_prediction_lis = []
                for i in range(n_experiment):
                    print('Experiment {}'.format(i))

                    semi_transformer = SemiTransformer(
                        tn_in_channels=feature_len,
                        tn_z_dim=tilenet_zdim,
                        tn_warm_start_model=None,
                        sentence_embedding=sentence_embedding,
                        output_pred=True,
                        query_type=query_type,
                        attn_n_tsteps=n_tsteps,
                        d_word_vec=tilenet_zdim,
                        d_model=tilenet_zdim,
                        d_inner=attention_dff,
                        n_layers=attention_layer,
                        n_head=attention_nhead,
                        d_k=tilenet_zdim // attention_nhead,
                        d_v=tilenet_zdim // attention_nhead,
                        dropout=dropout,
                        apply_position_enc=True)

                    optimizer = optim.Adam(semi_transformer.parameters(), lr=adam_lr, betas=adam_betas,
                                           weight_decay=weight_decay)

                    trained_epochs = train_attention(model=semi_transformer,
                                                     X_dir=X_dir,
                                                     X_train_indices_dic=train_indices_dic,
                                                     y_train=y_train,
                                                     X_valid_indices_dic=valid_indices_dic,
                                                     y_valid=y_valid,
                                                     X_test_indices_dic=test_indices_dic,
                                                     y_test=y_test,
                                                     n_tsteps=n_tsteps,
                                                     max_index=max_index,
                                                     n_triplets_per_file=n_triplets_per_file,
                                                     tilenet_margin=tilenet_margin,
                                                     tilenet_l2=tilenet_l2,
                                                     tilenet_ltn=tilenet_ltn,
                                                     unsup_weight=unsup_weight,
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
                                                                X_test_indices_dic=test_indices_dic,
                                                                y_test=y_test,
                                                                n_tsteps=n_tsteps,
                                                                max_index=max_index,
                                                                n_triplets_per_file=n_triplets_per_file,
                                                                batch_size=test_batch_size,
                                                                model_dir=param_model_out_dir,
                                                                model=semi_transformer,
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
                plot_predict_error(test_prediction, y_test, dim_test,
                                   Path('{}/err_{}.html'.format(param_result_out_dir, year)))

                results[year] = {'test_rmse': np.around(np.mean(test_rmse_lis), 3),
                                 'test_r2': np.around(np.mean(test_r2_lis), 3),
                                 'test_corr': np.around(np.mean(test_corr_lis), 3)}

                output_to_csv_simple(results, param_result_out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop Yield Train Semi Transformer')
    parser.add_argument('--neighborhood-radius', type=int, default=None, metavar='NR')
    parser.add_argument('--distant-radius', type=int, default=None, metavar='DR')
    parser.add_argument('--weight-decay', type=float, metavar='WDECAY')
    parser.add_argument('--tilenet-margin', type=float, default=50.0, metavar='MARGIN')
    parser.add_argument('--tilenet-l2', type=float, default=0.01, metavar='L2')
    parser.add_argument('--tilenet-ltn', type=float, default=0.1, metavar='LTN')
    parser.add_argument('--tilenet-zdim', type=int, default=512, metavar='ZDIM')
    parser.add_argument('--attention-layer', type=int, default=2, metavar='ALAYER')
    parser.add_argument('--attention-dff', type=int, default=2048, metavar='ADFF')
    parser.add_argument('--sentence-embedding', type=str, default='simple_average', metavar='SEMD')
    parser.add_argument('--query-type', type=str, default='fixed', metavar='QTYPE')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='DROPOUT')
    parser.add_argument('--unsup-weight', type=float, default=0.2, metavar='UWEIGHT')
    parser.add_argument('--patience', type=int, default=9999, metavar='PATIENCE')
    parser.add_argument('--feature', type=str, default='all', metavar='FEATURE')
    parser.add_argument('--feature-len', type=int, default=9, metavar='FEATURE_LEN')
    parser.add_argument('--year', type=int, default=2014, metavar='YEAR')
    parser.add_argument('--ntsteps', type=int, default=7, metavar='NTSTEPS', required=True)
    parser.add_argument('--train-years', type=int, default=None, metavar='TRAINYEAR', required=True)

    args = parser.parse_args()

    crop_yield_train_cross_location(args,
                                    data_dir='data/spatial_temporal/counties',
                                    model_out_dir='results/cross_location/models',
                                    result_out_dir='results/cross_location/results',
                                    log_out_dir='results/cross_location/prediction_logs',
                                    start_year=args.year,
                                    end_year=args.year,
                                    n_tsteps=args.ntsteps,
                                    train_years=args.train_years)
