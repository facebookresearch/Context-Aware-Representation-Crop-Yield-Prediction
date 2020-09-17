#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit

from scipy.stats import randint
from scipy.stats import uniform

sys.path.append("..")

from crop_yield_prediction.utils import Logger
from crop_yield_prediction.utils import plot_predict
from crop_yield_prediction.utils import plot_predict_error
from crop_yield_prediction.utils import get_statistics
from crop_yield_prediction.utils import output_to_csv_no_spatial


def _train_tuned_ridge_regression(train_valid, train, valid, test, predefined_split):
    print('Tuned Ridge Regression')
    X_scaler = StandardScaler()
    X_scaler.fit(train['X'])
    X_train_valid = X_scaler.transform(train_valid['X'])
    X_valid = X_scaler.transform(valid['X'])
    X_test = X_scaler.transform(test['X'])

    regr = Ridge()
    params = {'alpha': [10000, 5000, 1000, 500, 100, 75, 50, 25, 10, 1.0, 0.0001, 0]}
    regr_search = GridSearchCV(regr, params, cv=predefined_split)
    regr_search.fit(X_train_valid, train_valid['y'])
    # print(regr_search.best_params_)
    valid_prediction = regr_search.predict(X_valid)
    test_prediction = regr_search.predict(X_test)

    valid_corr, valid_r2, valid_rmse = get_statistics(valid['y'], valid_prediction, valid=True)
    test_corr, test_r2, test_rmse = get_statistics(test['y'], test_prediction, valid=False)

    return {'valid_corr': valid_corr[0], 'valid_r2': valid_r2, 'valid_rmse': valid_rmse,
            'test_corr': test_corr[0], 'test_r2': test_r2, 'test_rmse': test_rmse}, test_prediction


def _train_tuned_random_forest(train_valid, train, valid, test, predefined_split):
    print('Tuned RandomForest')

    valid_corr_lis, valid_r2_lis, valid_rmse_lis = [], [], []
    test_corr_lis, test_r2_lis, test_rmse_lis = [], [], []
    test_predictions = []
    for i in range(2):
        n_features = train_valid['X'].shape[1]

        max_features = randint(1, n_features + 1)
        min_samples_split = randint(2, 51)
        min_samples_leaf = randint(1, 51)
        min_weight_fraction_leaf = uniform(0.0, 0.5)
        max_leaf_nodes = randint(10, 1001)

        params = {'max_features': max_features,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'min_weight_fraction_leaf': min_weight_fraction_leaf,
                  'max_leaf_nodes': max_leaf_nodes}

        rf_search = RandomizedSearchCV(estimator=RandomForestRegressor(n_estimators=100, n_jobs=-1),
                                       param_distributions=params,
                                       n_iter=100,
                                       cv=predefined_split,
                                       n_jobs=-1)
        rf_search.fit(train_valid['X'], train_valid['y'])
        # print(rf_search.best_params_)
        valid_prediction = rf_search.predict(valid['X'])
        test_prediction = rf_search.predict(test['X'])

        valid_corr, valid_r2, valid_rmse = get_statistics(valid['y'], valid_prediction, valid=True)
        test_corr, test_r2, test_rmse = get_statistics(test['y'], test_prediction, valid=False)

        test_predictions.append(np.asarray(test_prediction))

        valid_corr_lis.append(valid_corr)
        valid_r2_lis.append(valid_r2)
        valid_rmse_lis.append(valid_rmse)
        test_corr_lis.append(test_corr)
        test_r2_lis.append(test_r2)
        test_rmse_lis.append(test_rmse)

    return {'valid_corr': np.around(np.mean([x[0] for x in valid_corr_lis]), 3),
            'valid_r2': np.around(np.mean(valid_r2_lis), 3),
            'valid_rmse': np.around(np.mean(valid_rmse_lis), 3),
            'test_corr': np.around(np.mean([x[0] for x in test_corr_lis]), 3),
            'test_r2': np.around(np.mean(test_r2_lis), 3),
            'test_rmse': np.around(np.mean(test_rmse_lis), 3)}, np.mean(np.asarray(test_predictions), axis=0)


def _train_tuned_neural_network(train_valid, train, valid, test, predefined_split):
    print('Tuned neural network')
    X_scaler = StandardScaler()
    X_scaler.fit(train['X'])
    X_train_valid = X_scaler.transform(train_valid['X'])
    X_valid = X_scaler.transform(valid['X'])
    X_test = X_scaler.transform(test['X'])

    valid_corr_lis, valid_r2_lis, valid_rmse_lis = [], [], []
    test_corr_lis, test_r2_lis, test_rmse_lis = [], [], []
    test_predictions = []
    for i in range(2):
        params = {'hidden_layer_sizes': [(256, 256), (256, )],
                  'learning_rate': ['adaptive', 'constant'],
                  'learning_rate_init': [0.01, 0.001],
                  'alpha': [0.0001, 0.001, 0.05, 0.1],
                  'activation': ['tanh', 'relu']
                  }
        mlp_search = GridSearchCV(estimator=MLPRegressor(max_iter=500),
                                  param_grid=params,
                                  cv=predefined_split,
                                  n_jobs=-1)
        mlp_search.fit(X_train_valid, train_valid['y'])
        # print(mlp_search.best_params_)
        valid_prediction = mlp_search.predict(X_valid)
        test_prediction = mlp_search.predict(X_test)

        valid_corr, valid_r2, valid_rmse = get_statistics(valid['y'], valid_prediction, valid=True)
        test_corr, test_r2, test_rmse = get_statistics(test['y'], test_prediction, valid=False)

        test_predictions.append(np.asarray(test_prediction))

        valid_corr_lis.append(valid_corr)
        valid_r2_lis.append(valid_r2)
        valid_rmse_lis.append(valid_rmse)
        test_corr_lis.append(test_corr)
        test_r2_lis.append(test_r2)
        test_rmse_lis.append(test_rmse)

    return {'valid_corr': np.around(np.mean([x[0] for x in valid_corr_lis]), 3),
            'valid_r2': np.around(np.mean(valid_r2_lis), 3),
            'valid_rmse': np.around(np.mean(valid_rmse_lis), 3),
            'test_corr': np.around(np.mean([x[0] for x in test_corr_lis]), 3),
            'test_r2': np.around(np.mean(test_r2_lis), 3),
            'test_rmse': np.around(np.mean(test_rmse_lis), 3)}, np.mean(np.asarray(test_predictions), axis=0)


def _predict_crop_yield(train_valid_df, train_df, valid_df, test_df, predefined_split, out_dir, year, end_month):

    all_features = [x for x in train_valid_df.columns.values
                if x not in ['year', 'state', 'county', 'yield'] and not x.endswith('mean')]
    features = []
    for f in all_features:
        if f[-1].isdigit():
            if int(f[-1]) <= end_month:
                features.append(f)
        else:
            features.append(f)
    print(features)

    dims = ['state', 'county']
    train_valid = {'y': np.array(train_valid_df['yield']), 'X': np.array(train_valid_df[features])}
    train = {'y': np.array(train_df['yield']), 'X': np.array(train_df[features])}
    valid = {'y': np.array(valid_df['yield']), 'X': np.array(valid_df[features])}
    test = {'y': np.array(test_df['yield']), 'X': np.array(test_df[features])}
    dim_test = np.array(test_df[dims])

    tuned_rr, tuned_rr_test_prediction = _train_tuned_ridge_regression(train_valid, train, valid, test, predefined_split)
    tuned_rf, tuned_rf_test_prediction = _train_tuned_random_forest(train_valid, train, valid, test, predefined_split)
    tuned_nn, tuned_nn_test_prediction = _train_tuned_neural_network(train_valid, train, valid, test, predefined_split)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for prediction, name in zip([tuned_rr_test_prediction,
                                 tuned_rf_test_prediction,
                                 tuned_nn_test_prediction],
                                ['tuned_rr', 'tuned_rf', 'tuned_nn']):
        np.save('{}/{}_{}.npy'.format(out_dir, name, year), prediction)
        plot_predict(prediction, dim_test, Path('{}/pred_{}_{}.html'.format(out_dir, name, year)))
        plot_predict_error(prediction, test['y'], dim_test, Path('{}/err_{}_{}.html'.format(out_dir, name, year)))

    return {'tuned_rr': tuned_rr, 'tuned_rf': tuned_rf, 'tuned_nn': tuned_nn}


def predict_no_spatial(csv_file, start_year, end_year, end_month, train_years, out_dir):
    print('Predict for {}...........'.format(csv_file))
    data = pd.read_csv(csv_file)

    results = {}
    for year in range(start_year, end_year+1):
        print('Predict year {}.....'.format(year))
        train_valid_data = data.loc[(data['year'] < year) & (data['year'] >= (year - (train_years + 1)))]
        train_data = data.loc[(data['year'] < (year - 1)) & (data['year'] >= (year - (train_years + 1)))]
        valid_data = data.loc[data['year'] == year - 1]
        test_data = data.loc[data['year'] == year]

        print('Train size {}, validate size {}, test size {}'.format(len(train_data), len(valid_data), len(test_data)))

        valid_index = train_valid_data['year'] == year - 1
        valid_fold = [0 if x else -1 for x in valid_index]
        predefined_split = PredefinedSplit(test_fold=valid_fold)

        year_results = _predict_crop_yield(train_valid_data, train_data, valid_data, test_data,
                                           predefined_split, '{}/nt{}_end_month{}'.format(out_dir, train_years, end_month), year, end_month)

        results[year] = year_results

    output_to_csv_no_spatial(results, '{}/nt{}_end_month{}'.format(out_dir, train_years, end_month))
