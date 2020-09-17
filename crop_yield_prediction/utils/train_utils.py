#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd

from crop_yield_prediction.plot import crop_yield_plot
from crop_yield_prediction.plot import crop_yield_prediction_error_plot


def get_statistics(y, prediction, valid):
    corr = tuple(map(lambda x: np.around(x, 3), pearsonr(y, prediction)))
    r2 = np.around(r2_score(y, prediction), 3)
    rmse = np.around(sqrt(mean_squared_error(y, prediction)), 3)

    if valid:
        print('Validation - Pearson correlation: {}, R2: {}, RMSE: {}'.format(corr, r2, rmse))
    else:
        print('Test - Pearson correlation: {}, R2: {}, RMSE: {}'.format(corr, r2, rmse))

    return corr, r2, rmse


def get_latest_model_dir(model_dir):
    latest_folder = sorted([x for x in os.listdir(model_dir) if x.startswith('log')], key=lambda x: int(x[3:]))[-1]

    return os.path.join(model_dir, latest_folder)


def get_latest_model(model_dir, cv=None):
    log_folders = sorted([x for x in os.listdir(model_dir) if x.startswith('log')], key=lambda x: int(x[3:]))[-1]
    check_dir = os.path.join(model_dir, log_folders) if cv is None else os.path.join(model_dir, log_folders, cv)
    latest_model = sorted([x for x in os.listdir(check_dir) if x.endswith('.tar')],
                          key=lambda x: int(x.split('.')[0][13:]))[-1]
    return os.path.join(check_dir, latest_model)


def get_latest_models_cvs(model_dir, cvs):
    log_folders = sorted([x for x in os.listdir(model_dir) if x.startswith('log')], key=lambda x: int(x[3:]))[-1]
    latest_models = []
    for cv in cvs:
        check_dir = os.path.join(model_dir, log_folders, cv)
        latest_model = sorted([x for x in os.listdir(check_dir) if x.endswith('.tar')],
                              key=lambda x: int(x.split('.')[0][13:]))[-1]
        latest_models.append(os.path.join(check_dir, latest_model))
    return latest_models


def plot_predict(prediction, dim, savepath):
    pred_dict = {}
    for idx, pred in zip(dim, prediction):
        state, county = idx
        state = str(int(state)).zfill(2)
        county = str(int(county)).zfill(3)

        pred_dict[state + county] = pred

    crop_yield_plot(pred_dict, savepath)


def plot_predict_error(prediction, real_values, dim, savepath):
    test_pred_error = prediction - real_values
    pred_dict = {}
    for idx, err in zip(dim, test_pred_error):
        state, county = idx
        state = str(int(state)).zfill(2)
        county = str(int(county)).zfill(3)

        pred_dict[state + county] = err

    crop_yield_prediction_error_plot(pred_dict, savepath)


def output_to_csv_no_spatial(results_dic, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    years = sorted(results_dic.keys())
    model_types = sorted(results_dic[years[0]].keys())

    for dt in ['valid', 'test']:
        data = []
        for year in years:
            year_data, columns = [], []
            for st in ['corr', 'r2', 'rmse']:
                for mt in model_types:
                    year_data.append(results_dic[year][mt]['{}_{}'.format(dt, st)])
                    columns.append('{}_{}'.format(mt, '{}_{}'.format(dt, st)))
            data.append(year_data)

        data = pd.DataFrame(data, columns=columns, index=years)
        data.to_csv('{}/{}.csv'.format(out_dir, dt))


def output_to_csv_complex(results_dic, out_dir):
    years = sorted(results_dic.keys())
    model_types = sorted(results_dic[years[0]].keys())

    for dt in ['train', 'test']:
        data = []
        for year in years:
            year_data, columns = [], []
            for st in ['corr', 'r2', 'rmse']:
                for mt in model_types:
                    year_data.append(results_dic[year][mt]['{}_{}'.format(dt, st)])
                    columns.append('{}_{}'.format(mt, '{}_{}'.format(dt, st)))
            data.append(year_data)

        data = pd.DataFrame(data, columns=columns, index=years)
        data.to_csv('{}/{}.csv'.format(out_dir, dt))


def output_to_csv_simple(results_dic, out_dir):
    years = sorted(results_dic.keys())

    data = []
    for year in years:
        year_data, columns = [], []
        for st in ['corr', 'r2', 'rmse']:
            year_data.append(results_dic[year]['test_'+st])
            columns.append(st)
        data.append(year_data)

    data = pd.DataFrame(data, columns=columns, index=years)
    data.to_csv('{}/test.csv'.format(out_dir))
