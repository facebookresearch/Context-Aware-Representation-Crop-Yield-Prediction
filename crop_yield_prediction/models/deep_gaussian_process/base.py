#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapt code from https://github.com/gabrieltseng/pycrop-yield-prediction


import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from datetime import datetime

from .gp import GaussianProcess
from .loss import l1_l2_loss


class ModelBase:
    """
    Base class for all models
    """
    def __init__(self, model, model_weight, model_bias, model_type, savedir, use_gp=True,
                 sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.savedir = savedir / model_type
        self.savedir.mkdir(parents=True, exist_ok=True)

        print(f'Using {device.type}')
        if device.type != 'cpu':
            model = model.cuda()
        self.model = model
        self.model_type = model_type
        self.model_weight = model_weight
        self.model_bias = model_bias

        self.device = device

        # # for reproducability
        # torch.manual_seed(42)
        # torch.cuda.manual_seed_all(42)

        self.gp = None
        if use_gp:
            self.gp = GaussianProcess(sigma, r_loc, r_year, sigma_e, sigma_b)

    def run(self, times, train_years, path_to_histogram=Path('data/deep_gaussian/data.npz'),
            pred_years=None, num_runs=2, train_steps=25000, batch_size=64,
            starter_learning_rate=1e-3, weight_decay=0, l1_weight=0, patience=10):
        """
        Train the models. Note that multiple models are trained: as per the paper, a model
        is trained for each year, with all preceding years used as training values. In addition,
        for each year, 2 models are trained to account for random initialization.

        Parameters
        ----------
        path_to_histogram: pathlib Path, default=Path('data/img_output/histogram_all_full.npz')
            The location of the training data
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int, list or None, default=None
            Which years to build models for. If None, the default values from the paper (range(2009, 2016))
            are used.
        num_runs: int, default=2
            The number of runs to do per year. Default taken from the paper
        train_steps: int, default=25000
            The number of steps for which to train the model. Default taken from the paper.
        batch_size: int, default=32
            Batch size when training. Default taken from the paper
        starter_learning_rate: float, default=1e-3
            Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
            steps. Default taken from the paper
        weight_decay: float, default=1
            Weight decay (L2 regularization) on the model weights
        l1_weight: float, default=0
            In addition to MSE, L1 loss is also used (sometimes). This is the weight to assign to this L1 loss.
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.
        """

        with np.load(path_to_histogram) as hist:
            images = hist['output_image']
            locations = hist['output_locations']
            yields = hist['output_yield']
            years = hist['output_year']
            indices = hist['output_index']

        # to collect results
        years_list, run_numbers, corr_list, r2_list, rmse_list, me_list, times_list = [], [], [], [], [], [], []
        if self.gp is not None:
            corr_gp_list, r2_gp_list, rmse_gp_list, me_gp_list = [], [], [], []

        if pred_years is None:
            pred_years = range(2014, 2019)
        elif type(pred_years) is int:
            pred_years = [pred_years]

        for pred_year in pred_years:
            for run_number in range(1, num_runs + 1):
                for time in times:
                    print(f'Training to predict on {pred_year}, Run number {run_number}')

                    results = self._run_1_year(train_years, images, yields,
                                               years, locations,
                                               indices, pred_year,
                                               time, run_number,
                                               train_steps, batch_size,
                                               starter_learning_rate,
                                               weight_decay, l1_weight,
                                               patience)

                    years_list.append(pred_year)
                    run_numbers.append(run_number)
                    times_list.append(time)

                    if self.gp is not None:
                        corr, r2, rmse, me, corr_gp, r2_gp, rmse_gp, me_gp = results
                        corr_gp_list.append(corr_gp)
                        r2_gp_list.append(r2_gp)
                        rmse_gp_list.append(rmse_gp)
                        me_gp_list.append(me_gp)
                    else:
                        corr, r2, rmse, me = results
                    corr_list.append(corr)
                    r2_list.append(r2)
                    rmse_list.append(rmse)
                    me_list.append(me)
                print('-----------')

        # save results to a csv file
        data = {'year': years_list, 'run_number': run_numbers, 'time_idx': times_list,
                'Corr': corr_list, 'R2': r2_list, 'RMSE': rmse_list, 'ME': me_list}
        if self.gp is not None:
            data['Corr_GP'] = corr_gp_list
            data['R2_GP'] = r2_gp_list
            data['RMSE_GP'] = rmse_gp_list
            data['ME_GP'] = me_gp_list
        results_df = pd.DataFrame(data=data)
        results_df.to_csv(self.savedir / f'{str(datetime.now())}.csv', index=False)

    def _run_1_year(self, train_years, images, yields, years, locations, indices, predict_year, time, run_number,
                    train_steps, batch_size, starter_learning_rate, weight_decay, l1_weight, patience):
        """
        Train one model on one year of data, and then save the model predictions.
        To be called by run().
        """
        train_data, val_data, test_data = self.prepare_arrays(train_years, images, yields, locations, indices, years, predict_year, time)

        # reinitialize the model, since self.model may be trained multiple
        # times in one call to run()
        self.reinitialize_model(time=time)

        train_scores, val_scores = self._train(train_data.images, train_data.yields,
                                               val_data.images, val_data.yields,
                                               train_steps, batch_size,
                                               starter_learning_rate,
                                               weight_decay, l1_weight,
                                               patience)

        results = self._predict(*train_data, *test_data, batch_size)

        model_information = {
            'state_dict': self.model.state_dict(),
            'val_loss': val_scores['loss'],
            'train_loss': train_scores['loss'],
        }
        for key in results:
            model_information[key] = results[key]

        # finally, get the relevant weights for the Gaussian Process
        model_weight = self.model.state_dict()[self.model_weight]
        model_bias = self.model.state_dict()[self.model_bias]

        if self.model.state_dict()[self.model_weight].device != 'cpu':
            model_weight, model_bias = model_weight.cpu(), model_bias.cpu()

        model_information['model_weight'] = model_weight.numpy()
        model_information['model_bias'] = model_bias.numpy()

        if self.gp is not None:
            print("Running Gaussian Process!")
            gp_pred = self.gp.run(model_information['train_feat'],
                                  model_information['test_feat'],
                                  model_information['train_loc'],
                                  model_information['test_loc'],
                                  model_information['train_years'],
                                  model_information['test_years'],
                                  model_information['train_real'],
                                  model_information['model_weight'],
                                  model_information['model_bias'])
            model_information['test_pred_gp'] = gp_pred.squeeze(1)

        filename = f'{predict_year}_{run_number}_{time}_{"gp" if (self.gp is not None) else ""}.pth.tar'
        torch.save(model_information, self.savedir / filename)
        return self.analyze_results(model_information['test_real'], model_information['test_pred'],
                                    model_information['test_pred_gp'] if self.gp is not None else None)

    def _train(self, train_images, train_yields, val_images, val_yields, train_steps,
               batch_size, starter_learning_rate, weight_decay, l1_weight, patience):
        """Defines the training loop for a model
        """

        train_dataset, val_dataset = TensorDataset(train_images, train_yields), TensorDataset(val_images, val_yields)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.Adam([pam for pam in self.model.parameters()],
                                     lr=starter_learning_rate,
                                     weight_decay=weight_decay)

        num_epochs = 50
        print(f'Training for {num_epochs} epochs')

        train_scores = defaultdict(list)
        val_scores = defaultdict(list)

        step_number = 0
        min_loss = np.inf
        best_state = self.model.state_dict()

        if patience is not None:
            epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.model.train()

            # running train and val scores are only for printing out
            # information
            running_train_scores = defaultdict(list)

            for train_x, train_y in train_dataloader:
                optimizer.zero_grad()
                pred_y = self.model(train_x)

                loss, running_train_scores = l1_l2_loss(pred_y, train_y, l1_weight,
                                                        running_train_scores)
                loss.backward()
                optimizer.step()

                train_scores['loss'].append(loss.item())

                step_number += 1

                if step_number in [4000, 20000]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10

            train_output_strings = []
            for key, val in running_train_scores.items():
                train_output_strings.append('{}: {}'.format(key, round(np.array(val).mean(), 5)))

            running_val_scores = defaultdict(list)
            self.model.eval()
            with torch.no_grad():
                for val_x, val_y, in val_dataloader:
                    val_pred_y = self.model(val_x)

                    val_loss, running_val_scores = l1_l2_loss(val_pred_y, val_y, l1_weight,
                                                              running_val_scores)

                    val_scores['loss'].append(val_loss.item())

            val_output_strings = []
            for key, val in running_val_scores.items():
                val_output_strings.append('{}: {}'.format(key, round(np.array(val).mean(), 5)))

            print('TRAINING: {}'.format(', '.join(train_output_strings)))
            print('VALIDATION: {}'.format(', '.join(val_output_strings)))

            epoch_val_loss = np.array(running_val_scores['loss']).mean()

            if epoch_val_loss < min_loss:
                best_state = self.model.state_dict()
                min_loss = epoch_val_loss

                if patience is not None:
                    epochs_without_improvement = 0
            elif patience is not None:
                epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    # revert to the best state dict
                    self.model.load_state_dict(best_state)
                    print('Early stopping!')
                    break

        self.model.load_state_dict(best_state)
        return train_scores, val_scores

    def _predict(self, train_images, train_yields, train_locations, train_indices,
                 train_years, test_images, test_yields, test_locations, test_indices,
                 test_years, batch_size):
        """
        Predict on the training and validation data. Optionally, return the last
        feature vector of the model.
        """
        train_dataset = TensorDataset(train_images, train_yields,
                                      train_locations, train_indices,
                                      train_years)

        test_dataset = TensorDataset(test_images, test_yields,
                                     test_locations, test_indices,
                                     test_years)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_yield, train_loc, train_idx, train_year in train_dataloader:
                model_output = self.model(train_im,
                                          return_last_dense=True if (self.gp is not None) else False)
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != 'cpu':
                        feat = feat.cpu()
                    results['train_feat'].append(feat.numpy())
                else:
                    pred = model_output
                results['train_pred'].extend(pred.squeeze(1).tolist())
                results['train_real'].extend(train_yield.squeeze(1).tolist())
                results['train_loc'].append(train_loc.numpy())
                results['train_indices'].append(train_idx.numpy())
                results['train_years'].extend(train_year.tolist())

            for test_im, test_yield, test_loc, test_idx, test_year in test_dataloader:
                model_output = self.model(test_im,
                                          return_last_dense=True if (self.gp is not None) else False)
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != 'cpu':
                        feat = feat.cpu()
                    results['test_feat'].append(feat.numpy())
                else:
                    pred = model_output
                results['test_pred'].extend(pred.squeeze(1).tolist())
                results['test_real'].extend(test_yield.squeeze(1).tolist())
                results['test_loc'].append(test_loc.numpy())
                results['test_indices'].append(test_idx.numpy())
                results['test_years'].extend(test_year.tolist())

        for key in results:
            if key in ['train_feat', 'test_feat', 'train_loc',
                       'test_loc', 'train_indices', 'test_indices']:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])
        return results

    def prepare_arrays(self, train_years, images, yields, locations, indices, years, predict_year,
                       time):
        """Prepares the inputs for the model, in the following way:
        - normalizes the images
        - splits into a train and val set
        - turns the numpy arrays into tensors
        - removes excess months, if monthly predictions are being made
        """
        train_idx = np.nonzero((years >= (predict_year - (train_years + 1))) & (years < (predict_year-1)))[0]
        val_idx = np.nonzero(years == predict_year - 1)[0]
        test_idx = np.nonzero(years == predict_year)[0]

        train_images, val_images, test_images = self._normalize(images[train_idx], images[val_idx], images[test_idx])

        print(f'Train set size: {train_idx.shape[0]}, Validation set size {val_idx.shape[0]}, Test set size: {test_idx.shape[0]}')

        Data = namedtuple('Data', ['images', 'yields', 'locations', 'indices', 'years'])

        train_data_images = torch.tensor(train_images[:, :, :time, :]).float()
        train_data_yield = torch.tensor(yields[train_idx]).float().unsqueeze(1)
        if self.device.type != 'cpu':
            train_data_images = train_data_images.cuda()
            train_data_yield = train_data_yield.cuda()
        train_data = Data(
            images=train_data_images,
            yields=train_data_yield,
            locations=torch.tensor(locations[train_idx]),
            indices=torch.tensor(indices[train_idx]),
            years=torch.tensor(years[train_idx])
        )

        val_data_images = torch.tensor(val_images[:, :, :time, :]).float()
        val_data_yield = torch.tensor(yields[val_idx]).float().unsqueeze(1)
        if self.device.type != 'cpu':
            val_data_images = val_data_images.cuda()
            val_data_yield = val_data_yield.cuda()
        val_data = Data(
            images=val_data_images,
            yields=val_data_yield,
            locations=torch.tensor(locations[val_idx]),
            indices=torch.tensor(indices[val_idx]),
            years=torch.tensor(years[val_idx])
        )

        test_data_images = torch.tensor(test_images[:, :, :time, :]).float()
        test_data_yield = torch.tensor(yields[test_idx]).float().unsqueeze(1)
        if self.device.type != 'cpu':
            test_data_images = test_data_images.cuda()
            test_data_yield = test_data_yield.cuda()
        test_data = Data(
            images=test_data_images,
            yields=test_data_yield,
            locations=torch.tensor(locations[test_idx]),
            indices=torch.tensor(indices[test_idx]),
            years=torch.tensor(years[test_idx])
        )

        return train_data, val_data, test_data

    @staticmethod
    def _normalize(train_images, val_images, test_images):
        """
        Find the mean values of the bands in the train images. Use these values
        to normalize both the training and validation images.

        A little awkward, since transpositions are necessary to make array broadcasting work
        """
        mean = np.mean(train_images, axis=(0, 2, 3))

        train_images = (train_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)
        val_images = (val_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)
        test_images = (test_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)

        return train_images, val_images, test_images

    @staticmethod
    def analyze_results(true, pred, pred_gp):
        """Calculate ME and RMSE
        """
        corr = pearsonr(true, pred)[0]
        r2 = r2_score(true, pred)
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        me = np.mean(true - pred)

        print(f'Without GP: Corr: {corr}, R2: {r2}, RMSE: {rmse}, ME: {me}')

        if pred_gp is not None:
            corr_gp = pearsonr(true, pred_gp)[0]
            r2_gp = r2_score(true, pred_gp)
            rmse_gp = np.sqrt(np.mean((true - pred_gp) ** 2))
            me_gp = np.mean(true - pred_gp)
            print(f'With GP: Corr: {corr_gp}, R2: {r2_gp}, RMSE: {rmse_gp}, ME: {me_gp}')
            return corr, r2, rmse, me, corr_gp, r2_gp, rmse_gp, me_gp
        return corr, r2, rmse, me

    def reinitialize_model(self, time=None):
        raise NotImplementedError
