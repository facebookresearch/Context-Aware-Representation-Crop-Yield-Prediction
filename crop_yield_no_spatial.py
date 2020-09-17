#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from crop_yield_prediction.utils import Logger
from crop_yield_prediction.models import *

import os
import sys
import argparse


def predict_for_no_spatial(train_years):
    log_folder = 'results/no_spatial/prediction_logs'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    sys.stdout = Logger('{}/nt{}_all_results_online_learning.txt'.format(log_folder, train_years))

    predict_no_spatial('data/no_spatial/soybeans_3_9.csv', 2014, 2018, 9, train_years,
                       'crop_yield_no_spatial/results/all')
    predict_no_spatial('data/no_spatial/soybeans_3_9.csv', 2014, 2018, 8, train_years,
                       'crop_yield_no_spatial/results/all')
    predict_no_spatial('data/no_spatial/soybeans_3_9.csv', 2014, 2018, 7, train_years,
                       'crop_yield_no_spatial/results/all')
    predict_no_spatial('data/no_spatial/soybeans_3_9.csv', 2014, 2018, 6, train_years,
                       'crop_yield_no_spatial/results/all')
    predict_no_spatial('data/no_spatial/soybeans_3_9.csv', 2014, 2018, 5, train_years,
                       'crop_yield_no_spatial/results/all')

    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', required=True)
    parser.add_argument('--train-years', type=int, default=None, metavar='TRAINYEAR', required=True)

    args = parser.parse_args()

    predict = args.predict
    train_years = args.train_years

    if predict == 'no_spatial':
        predict_for_no_spatial(train_years)
