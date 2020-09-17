#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from crop_yield_prediction.dataloader import c3d_dataloader

import time
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats.stats import pearsonr
import numpy as np
import torch
from torch.autograd import Variable


def prep_data(batch_X, batch_y, cuda):
    batch_X, batch_y = Variable(batch_X), Variable(batch_y)
    if cuda:
        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

    return batch_X, batch_y


def train_epoch(model, train_dataloader, optimizer, cuda):
    ''' Epoch operation in training phase'''

    model.train()
    if cuda:
        model.cuda()

    n_batches = len(train_dataloader)
    sum_loss = 0

    for batch_X, batch_y in train_dataloader:
        batch_X, batch_y = prep_data(batch_X, batch_y, cuda)

        # forward
        optimizer.zero_grad()

        pred = model(batch_X)

        loss_func = torch.nn.MSELoss()
        loss = loss_func(pred, batch_y)

        loss.backward()
        optimizer.step()

        # note keeping
        sum_loss += loss.item()

    avg_loss = sum_loss / n_batches

    return avg_loss


def cal_performance(prediction, y):
    rmse = np.around(sqrt(mean_squared_error(y, prediction)), 3)
    r2 = np.around(r2_score(y, prediction), 3)
    corr = tuple(map(lambda x: np.around(x, 3), pearsonr(y, prediction)))[0]

    return rmse, r2, corr


def eval_epoch(model, validation_dataloader, cuda):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    if cuda:
        model.cuda()

    n_batches = len(validation_dataloader)
    n_samples = len(validation_dataloader.dataset)
    batch_size = validation_dataloader.batch_size

    predictions = torch.zeros(n_samples)
    # collect y as batch_y has been shuffled
    y = torch.zeros(n_samples)

    sum_loss = 0

    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(validation_dataloader):
            batch_X, batch_y = prep_data(batch_X, batch_y, cuda)

            # forward
            pred = model(batch_X)

            loss_func = torch.nn.MSELoss()
            loss = loss_func(pred, batch_y)

            start = i * batch_size
            end = start + batch_size if i != n_batches - 1 else n_samples
            predictions[start:end] = pred
            y[start:end] = batch_y

            sum_loss += loss.item()

    if cuda:
        predictions, y = predictions.cpu(), y.cpu()
        predictions, y = predictions.data.numpy(), y.data.numpy()

    rmse, r2, corr = cal_performance(predictions, y)
    avg_loss = sum_loss / n_batches

    return avg_loss, rmse, r2, corr


def eval_test(X_dir, X_test_indices, y_test, n_tsteps, max_index, n_triplets_per_file, batch_size, model_dir, model, epochs, year,
              exp_idx, log_file):
    with open(log_file, 'a') as f:
        print('Predict year {}'.format(year), file=f, flush=True)
        print('Test size {}'.format(y_test.shape[0]), file=f, flush=True)
        print('Experiment {}'.format(exp_idx), file=f, flush=True)

        cuda = torch.cuda.is_available()
        models = []
        for epoch_i in range(epochs):
            models.append('{}/{}_{}_epoch{}.tar'.format(model_dir, exp_idx, year, epoch_i))
        best_model = '{}/{}_{}_best.tar'.format(model_dir, exp_idx, year)
        models.append(best_model)

        for model_file in models:
            checkpoint = torch.load(model_file) if cuda else torch.load(model_file, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            if cuda:
                model.cuda()

            test_dataloader = c3d_dataloader(X_dir, X_test_indices[0], X_test_indices[1], y_test, n_tsteps,
                                             max_index, n_triplets_per_file, batch_size, shuffle=False, num_workers=4)
            n_batches = len(test_dataloader)
            n_samples = len(y_test)

            predictions = torch.zeros(n_samples)

            with torch.no_grad():
                for i, (batch_X, batch_y) in enumerate(test_dataloader):
                    batch_X, batch_y = prep_data(batch_X, batch_y, cuda)

                    # forward
                    pred = model(batch_X)

                    start = i * batch_size
                    end = start + batch_size if i != n_batches - 1 else n_samples
                    predictions[start:end] = pred

            if cuda:
                predictions = predictions.cpu()
                predictions = predictions.data.numpy()

            rmse, r2, corr = cal_performance(predictions, y_test)

            if 'epoch' in model_file:
                print('  - {header:12} epoch: {epoch: 5}, rmse: {rmse: 8.3f}, r2: {r2: 8.3f}, corr: {corr: 8.3f}'.
                      format(header=f"({'Test'})", epoch=checkpoint['epoch'], rmse=rmse, r2=r2, corr=corr), file=f, flush=True)
            else:
                print('  - {header:12} best selected based on validation set, '
                      'rmse: {rmse: 8.3f}, r2: {r2: 8.3f}, corr: {corr: 8.3f}'.
                      format(header=f"({'Test'})", rmse=rmse, r2=r2, corr=corr), file=f, flush=True)

    return predictions, rmse, r2, corr


def eval_test_best_only(test_dataloader, y_test, batch_size, model, epoch, log_file):
    cuda = torch.cuda.is_available()

    model.eval()
    if cuda:
        model.cuda()

    n_batches = len(test_dataloader)
    n_samples = len(y_test)

    predictions = torch.zeros(n_samples)

    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(test_dataloader):
            batch_X, batch_y = prep_data(batch_X, batch_y, cuda)

            # forward
            pred = model(batch_X)

            start = i * batch_size
            end = start + batch_size if i != n_batches - 1 else n_samples
            predictions[start:end] = pred

    if cuda:
        predictions = predictions.cpu()
        predictions = predictions.data.numpy()

    rmse, r2, corr = cal_performance(predictions, y_test)

    print('  - {header:12} epoch: {epoch: 5}, rmse: {rmse: 8.3f}, r2: {r2: 8.3f}, corr: {corr: 8.3f}'.
          format(header=f"({'Test_Best'})", epoch=epoch, rmse=rmse, r2=r2, corr=corr), file=log_file, flush=True)


def train_c3d(model, X_dir, X_train_indices, y_train, X_valid_indices, y_valid, X_test_indices, y_test, n_tsteps,
              max_index, n_triplets_per_file, patience, optimizer, batch_size, test_batch_size, n_epochs, out_dir, year,
              exp_idx, log_file):
    with open(log_file, 'a') as f:
        print('Predict year {}......'.format(year), file=f, flush=True)
        print('Train size {}, valid size {}'.format(y_train.shape[0], y_valid.shape[0]), file=f, flush=True)
        print('Experiment {}'.format(exp_idx), file=f, flush=True)

        cuda = torch.cuda.is_available()

        train_dataloader = c3d_dataloader(X_dir, X_train_indices[0], X_train_indices[1], y_train, n_tsteps,
                                          max_index, n_triplets_per_file, batch_size, shuffle=True, num_workers=4)
        validation_dataloader = c3d_dataloader(X_dir, X_valid_indices[0], X_valid_indices[1], y_valid, n_tsteps,
                                               max_index, n_triplets_per_file, batch_size, shuffle=False, num_workers=4)
        test_dataloader = c3d_dataloader(X_dir, X_test_indices[0], X_test_indices[1], y_test, n_tsteps,
                                         max_index, n_triplets_per_file, test_batch_size, shuffle=False, num_workers=4)

        valid_rmse_min = np.inf

        if patience is not None:
            epochs_without_improvement = 0
        for epoch_i in range(n_epochs):
            print('[ Epoch', epoch_i, ']', file=f, flush=True)

            start = time.time()
            train_loss = train_epoch(model, train_dataloader, optimizer, cuda)
            print('  - {header:12} avg loss: {loss: 8.3f}, elapse: {elapse:3.3f} min'.
                  format(header=f"({'Training'})", loss=train_loss,
                         elapse=(time.time() - start) / 60), file=f, flush=True)

            # if epoch_i in [20, 40]:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] /= 10

            start = time.time()
            valid_loss, valid_rmse, valid_r2, valid_corr = eval_epoch(model, validation_dataloader, cuda)
            print('  - {header:12} loss: {loss: 8.3f}, rmse: {rmse: 8.3f}, r2: {r2: 8.3f}, corr: {corr: 8.3f}, '
                  'elapse: {elapse:3.3f} min'.
                  format(header=f"({'Validation'})", loss=valid_loss,
                         rmse=valid_rmse, r2=valid_r2, corr=valid_corr, elapse=(time.time() - start) / 60), file=f,
                  flush=True)

            checkpoint = {'epoch': epoch_i, 'model_state_dict': model.state_dict()}
            torch.save(checkpoint, '{}/{}_{}_epoch{}.tar'.format(out_dir, exp_idx, year, epoch_i))

            if valid_rmse < valid_rmse_min:
                eval_test_best_only(test_dataloader, y_test, test_batch_size, model, epoch_i, f)

                torch.save(checkpoint, '{}/{}_{}_best.tar'.format(out_dir, exp_idx, year))
                print('    - [Info] The checkpoint file has been updated at epoch {}.'.format(epoch_i), file=f, flush=True)
                valid_rmse_min = valid_rmse

                if patience is not None:
                    epochs_without_improvement = 0
            elif patience is not None:
                epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    print('Early stopping!')
                    return epoch_i + 1

        return n_epochs
