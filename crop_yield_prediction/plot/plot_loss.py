#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_loss(params):
    out_dir = '../../results/spatial_temporal/plots/{}'.format(params[:-4])
    os.makedirs(out_dir, exist_ok=True)
    prediction_log = '../../results/spatial_temporal/prediction_logs/{}'.format(params)
    train_epochs_dic = defaultdict(lambda: defaultdict(list))
    train_loss_dic, train_super_loss_dic, train_unsuper_loss_dic = (defaultdict(lambda: defaultdict(list)) for _ in range(3))
    valid_loss_dic, valid_super_loss_dic, valid_unsuper_loss_dic = (defaultdict(lambda: defaultdict(list)) for _ in range(3))
    valid_l_n_loss_dic, valid_l_d_loss_dic, valid_l_nd_loss_dic, valid_sn_loss_dic, valid_tn_loss_dic, valid_norm_loss_dic = \
        (defaultdict(lambda: defaultdict(list)) for _ in range(6))
    valid_rmse_dic, valid_r2_dic, valid_corr_dic = (defaultdict(lambda: defaultdict(list)) for _ in range(3))
    test_epochs_dic = defaultdict(lambda: defaultdict(list))
    test_rmse_dic, test_r2_dic, test_corr_dic = (defaultdict(lambda: defaultdict(list)) for _ in range(3))

    exp = 0
    year = 0
    with open(prediction_log) as f:
        content = f.readlines()
        for line in content:
            line = line.strip()
            if line.startswith('Predict'):
                year = int(line.split()[2][:4])
            if line.startswith('Experiment'):
                exp = int(line.split()[1])
            if 'Epoch' in line:
                train_epochs_dic[year][exp].append(int(line.split()[2]))
            if 'Training' in line:
                ws = line.split()
                train_loss_dic[year][exp].append(float(ws[4][:-1]))
                train_super_loss_dic[year][exp].append(float(ws[7][:-1]))
                train_unsuper_loss_dic[year][exp].append(float(ws[10][:-1]))
            if 'Validation' in line:
                ws = line.split()
                valid_loss_dic[year][exp].append(float(ws[3][:-1]))
                valid_super_loss_dic[year][exp].append(float(ws[6][:-1]))
                valid_unsuper_loss_dic[year][exp].append(float(ws[9][:-1]))
                valid_l_n_loss_dic[year][exp].append(float(ws[12][:-1]))
                valid_l_d_loss_dic[year][exp].append(float(ws[15][:-1]))
                valid_l_nd_loss_dic[year][exp].append(float(ws[18][:-1]))
                valid_sn_loss_dic[year][exp].append(float(ws[20][:-1]))
                valid_tn_loss_dic[year][exp].append(float(ws[22][:-1]))
                valid_norm_loss_dic[year][exp].append(float(ws[24][:-1]))
                valid_rmse_dic[year][exp].append(float(ws[26][:-1]))
                valid_r2_dic[year][exp].append(float(ws[28][:-1]))
                valid_corr_dic[year][exp].append(float(ws[30][:-1]))
            if '(Test)' in line and 'epoch' in line:
                ws = line.split()
                test_epochs_dic[year][exp].append(int(ws[3][:-1]))
                test_rmse_dic[year][exp].append(float(ws[5][:-1]))
                test_r2_dic[year][exp].append(float(ws[7][:-1]))
                test_corr_dic[year][exp].append(float(ws[9]))

    for year in train_epochs_dic.keys():
        n_exps = len(train_epochs_dic[year])
        for i in range(n_exps):
            # assert train_epochs_dic[year][i] == test_epochs_dic[year][i], params

            plt.plot(train_epochs_dic[year][i], train_loss_dic[year][i], label='Training')
            plt.plot(train_epochs_dic[year][i], valid_loss_dic[year][i], label='Validation')
            plt.title(params, fontsize=8)
            plt.grid(True)
            plt.legend()
            plt.savefig('{}/{}_{}_total_loss.jpg'.format(out_dir, year, i), dpi=300)
            plt.close()

            plt.plot(train_epochs_dic[year][i], train_super_loss_dic[year][i], label='Training')
            plt.plot(train_epochs_dic[year][i], valid_super_loss_dic[year][i], label='Validation')
            plt.title(params, fontsize=8)
            plt.grid(True)
            plt.legend()
            plt.savefig('{}/{}_{}_supervised_loss.jpg'.format(out_dir, year, i), dpi=300)
            plt.close()

            plt.plot(train_epochs_dic[year][i], train_unsuper_loss_dic[year][i], label='Training')
            plt.plot(train_epochs_dic[year][i], valid_unsuper_loss_dic[year][i], label='Validation')
            plt.title(params, fontsize=8)
            plt.grid(True)
            plt.legend()
            plt.savefig('{}/{}_{}_unsupervised_loss.jpg'.format(out_dir, year, i), dpi=300)
            plt.close()

            # valid_l_n_loss, valid_l_d_loss, valid_l_nd_loss, valid_sn_loss, valid_tn_loss, valid_norm_loss
            plt.plot(train_epochs_dic[year][i], valid_l_n_loss_dic[year][i], label='l_n_loss')
            plt.plot(train_epochs_dic[year][i], valid_l_d_loss_dic[year][i], label='l_d_loss')
            plt.plot(train_epochs_dic[year][i], valid_l_nd_loss_dic[year][i], label='l_nd_loss')
            plt.plot(train_epochs_dic[year][i], valid_sn_loss_dic[year][i], label='spatial_neighbor_loss')
            plt.plot(train_epochs_dic[year][i], valid_tn_loss_dic[year][i], label='temporal_neighbor_loss')
            plt.plot(train_epochs_dic[year][i], valid_norm_loss_dic[year][i], label='l2_norm_loss')
            plt.title(params, fontsize=8)
            plt.grid(True)
            plt.legend()
            plt.savefig('{}/{}_{}_validation_various_losses.jpg'.format(out_dir, year, i), dpi=300)
            plt.close()

            plt.plot(train_epochs_dic[year][i], valid_rmse_dic[year][i], label='Validation')
            plt.plot(test_epochs_dic[year][i], test_rmse_dic[year][i], label='Test')
            plt.title(params, fontsize=8)
            plt.grid(True)
            plt.legend()
            plt.savefig('{}/{}_{}_rmse.jpg'.format(out_dir, year, i), dpi=300)
            plt.close()

            plt.plot(train_epochs_dic[year][i], valid_r2_dic[year][i], label='Validation')
            plt.plot(test_epochs_dic[year][i], test_r2_dic[year][i], label='Test')
            plt.title(params, fontsize=8)
            plt.grid(True)
            plt.legend()
            plt.savefig('{}/{}_{}_r2.jpg'.format(out_dir, year, i), dpi=300)
            plt.close()

            plt.plot(train_epochs_dic[year][i], valid_corr_dic[year][i], label='Validation')
            plt.plot(test_epochs_dic[year][i], test_corr_dic[year][i], label='Test')
            plt.title(params, fontsize=8)
            plt.grid(True)
            plt.legend()
            plt.savefig('{}/{}_{}_corr.jpg'.format(out_dir, year, i), dpi=300)
            plt.close()


if __name__ == '__main__':
    for prediction_log in os.listdir('../../results/spatial_temporal/prediction_logs'):
        if prediction_log.endswith('.txt'):
            print(prediction_log)
            plot_loss(prediction_log)

